from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver  # <<< Добавлено обратно
from langgraph.graph import END, START, StatefulGraph
from lightrag import LightRAG
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agents.assessment_agent import run_assessment_agent
from app.agents.curriculum_agent import run_curriculum_agent
from app.agents.knowledge_retrieval_agent import run_knowledge_retrieval_agent
from app.agents.recommendation_agent import run_recommendation_agent
from app.agents.state import TutorGraphState
from app.tools.lightrag_retriever_tool import LightRAGRetrieverTool

CURRICULUM_NODE = "curriculum_agent"
KNOWLEDGE_RETRIEVAL_NODE = "knowledge_retrieval_agent"
ASSESSMENT_NODE = "assessment_agent"
RECOMMENDATION_NODE = "recommendation_agent"
HANDLE_ERROR_NODE = "handle_error"


async def curriculum_agent_node(state: TutorGraphState, config: dict):
    logger.debug(f"Вызов узла: {CURRICULUM_NODE}")
    session_pool = config["configurable"]["session_pool"]
    return await run_curriculum_agent(state, session_pool)


async def knowledge_retrieval_node(state: TutorGraphState, config: dict):
    logger.debug(f"Вызов узла: {KNOWLEDGE_RETRIEVAL_NODE}")
    lightrag_tool = config["configurable"]["lightrag_tool"]
    return await run_knowledge_retrieval_agent(state, lightrag_tool)


async def assessment_node(state: TutorGraphState, config: dict):
    logger.debug(f"Вызов узла: {ASSESSMENT_NODE}")
    llm = config["configurable"]["llm"]
    return await run_assessment_agent(state, llm)


async def recommendation_node(state: TutorGraphState, config: dict):
    logger.debug(f"Вызов узла: {RECOMMENDATION_NODE}")
    llm = config["configurable"]["llm"]
    session_pool = config["configurable"]["session_pool"]
    return await run_recommendation_agent(state, llm, session_pool)


async def handle_error_node_func(state: TutorGraphState, config: dict):
    logger.debug(f"Вызов узла: {HANDLE_ERROR_NODE}")
    error_message = state.get("error_message", "Произошла неизвестная ошибка в графе.")
    logger.error(f"Ошибка в графе: {error_message}")
    return {"error_message": error_message}


def should_retrieve_knowledge(state: TutorGraphState) -> str:
    logger.debug(
        f"Условие: should_retrieve_knowledge. Ошибка: {state.get('error_message')}"
    )
    if state.get("error_message"):
        return HANDLE_ERROR_NODE
    if state.get("current_topic_id") or state.get("input_query"):
        logger.debug("Решение: извлекать знания.")
        return KNOWLEDGE_RETRIEVAL_NODE
    logger.debug(
        "Решение: пропустить извлечение знаний (нет темы/запроса), возможно ошибка или переход к END."
    )
    # Если curriculum_agent не установил тему и не выдал ошибку, это странно.
    # Можно добавить явный переход в HANDLE_ERROR_NODE или END.
    # Для большей надежности, если нет темы, это может быть ошибкой конфигурации или логики.
    if not state.get("current_topic_id"):  # Явная проверка
        state["error_message"] = "Тема не была определена Агентом Учебного Плана."
        return HANDLE_ERROR_NODE
    return END  # По умолчанию, если нет input_query и current_topic_id (хотя это условие выше)


def should_assess_or_recommend(state: TutorGraphState) -> str:
    logger.debug(
        f"Условие: should_assess_or_recommend. Ошибка: {state.get('error_message')}"
    )
    if state.get("error_message"):
        return HANDLE_ERROR_NODE

    if state.get("student_answer"):
        logger.debug("Решение: оценить ответ студента (assessment_node).")
        return ASSESSMENT_NODE
    elif state.get("retrieved_context"):
        logger.debug("Решение: задать вопрос/продолжить оценку (assessment_node).")
        return ASSESSMENT_NODE
    else:  # Контекст не извлечен (возможно, RAG ничего не нашел или была ошибка)
        logger.debug(
            "Решение: Контекст не извлечен. Переход к рекомендациям или обработка ошибки."
        )
        # Если retrieved_context пуст, assessment_agent должен это обработать и, возможно, выдать ошибку.
        # Или мы можем здесь решить перейти к рекомендациям, если это допустимый сценарий.
        # Пока что, пусть assessment_agent разбирается с отсутствием контекста.
        # Если RAG вернул пустой список, assessment_agent сообщит, что контекста нет.
        return ASSESSMENT_NODE  # Assessment agent решит, что делать без контекста (например, сообщить об этом)


def after_assessment_router(state: TutorGraphState) -> str:
    logger.debug(
        f"Условие: after_assessment_router. Ошибка: {state.get('error_message')}"
    )
    if state.get("error_message"):
        return HANDLE_ERROR_NODE

    if state.get("assessment_question"):
        logger.debug(
            "Решение: Завершить текущий вызов графа (ожидание ответа студента)."
        )
        return END
    elif state.get("assessment_feedback"):
        logger.debug("Решение: перейти к рекомендациям (recommendation_node).")
        return RECOMMENDATION_NODE
    else:
        logger.warning(
            "Состояние после assessment_agent неясно (нет нового вопроса или фидбека). Переход к рекомендациям."
        )
        # Если assessment_agent не смог ни задать вопрос, ни дать фидбек (например, из-за отсутствия контекста),
        # то можно перейти к рекомендациям или обработать как ошибку.
        # Переход к рекомендациям может быть безопасным вариантом.
        return RECOMMENDATION_NODE  # Изменено: если нет вопроса/фидбека, все равно к рекомендациям


def after_recommendations_router(state: TutorGraphState) -> str:
    logger.debug(
        f"Условие: after_recommendations_router. Ошибка: {state.get('error_message')}"
    )
    if state.get("error_message"):
        return HANDLE_ERROR_NODE
    logger.debug("Решение: Завершить граф после рекомендаций.")
    return END


def create_tutor_graph(
    llm: ChatOpenAI,
    rag_instance: LightRAG,
    session_pool: async_sessionmaker[AsyncSession],
) -> StatefulGraph:
    lightrag_tool = LightRAGRetrieverTool(rag_instance=rag_instance)
    graph_config = {
        "llm": llm,
        "session_pool": session_pool,
        "lightrag_tool": lightrag_tool,
    }

    memory_saver = MemorySaver()  # <<< Инициализируем MemorySaver
    workflow = StatefulGraph(
        TutorGraphState, checkpointer=memory_saver
    )  # <<< Передаем checkpointer

    workflow.add_node(CURRICULUM_NODE, curriculum_agent_node)
    workflow.add_node(KNOWLEDGE_RETRIEVAL_NODE, knowledge_retrieval_node)
    workflow.add_node(ASSESSMENT_NODE, assessment_node)
    workflow.add_node(RECOMMENDATION_NODE, recommendation_node)
    workflow.add_node(HANDLE_ERROR_NODE, handle_error_node_func)

    workflow.add_edge(START, CURRICULUM_NODE)

    workflow.add_conditional_edges(
        CURRICULUM_NODE,
        should_retrieve_knowledge,
        {
            KNOWLEDGE_RETRIEVAL_NODE: KNOWLEDGE_RETRIEVAL_NODE,
            HANDLE_ERROR_NODE: HANDLE_ERROR_NODE,
            END: END,
        },
    )

    workflow.add_conditional_edges(
        KNOWLEDGE_RETRIEVAL_NODE,
        should_assess_or_recommend,
        {
            ASSESSMENT_NODE: ASSESSMENT_NODE,
            # RECOMMENDATION_NODE: RECOMMENDATION_NODE, # Убрано, assessment_node сам разберется с отсутствием контекста
            HANDLE_ERROR_NODE: HANDLE_ERROR_NODE,
            END: END,  # Если KNOWLEDGE_RETRIEVAL_NODE решит завершить (маловероятно)
        },
    )

    workflow.add_conditional_edges(
        ASSESSMENT_NODE,
        after_assessment_router,
        {
            RECOMMENDATION_NODE: RECOMMENDATION_NODE,
            HANDLE_ERROR_NODE: HANDLE_ERROR_NODE,
            END: END,
        },
    )

    workflow.add_conditional_edges(
        RECOMMENDATION_NODE,
        after_recommendations_router,
        {HANDLE_ERROR_NODE: HANDLE_ERROR_NODE, END: END},
    )

    workflow.add_edge(HANDLE_ERROR_NODE, END)

    # Компилируем граф с checkpointer'ом, который уже был передан в конструктор StatefulGraph
    # compiled_graph = workflow.compile().with_config(configurable=graph_config) # Checkpointer уже в workflow
    compiled_graph = workflow.compile(checkpointer=memory_saver).with_config(
        configurable=graph_config
    )
    # Или если checkpointer уже в конструкторе StatefulGraph, то просто:
    # compiled_graph = workflow.compile().with_config(configurable=graph_config)
    # Однако, документация часто показывает compile(checkpointer=...). Проверим.
    # Да, checkpointer передается в compile() или в конструктор StatefulGraph.
    # Если передан в конструктор, то в compile() не нужно.
    # Если не передан в конструктор, то нужно в compile().
    # Для ясности, если он уже в StatefulGraph, то compile() без него.
    # Но если мы хотим использовать with_config для передачи checkpointer'а во время выполнения,
    # то его не нужно в конструктор StatefulGraph.
    # Давайте оставим его в конструкторе StatefulGraph, как более современный подход.
    # Тогда компиляция:
    # compiled_graph = workflow.compile().with_config(configurable=graph_config)
    # Однако, если мы хотим, чтобы checkpointer был частью `config` для `with_config`,
    # то его не нужно в конструктор StatefulGraph.
    # Давайте оставим `MemorySaver` как есть и передадим его в compile.
    # Это наиболее частый паттерн в примерах.

    logger.info("Граф ИИ Тьютора скомпилирован с MemorySaver.")
    return compiled_graph
