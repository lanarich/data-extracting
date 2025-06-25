from agents.assessment_agent import run_assessment_agent
from agents.curriculum_agent import run_curriculum_agent
from agents.knowledge_retrieval_agent import run_knowledge_retrieval_agent
from agents.recommendation_agent import run_recommendation_agent
from agents.state import TutorGraphState
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from lightrag import LightRAG
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from tools.lightrag_retriever_tool import LightRAGRetrieverTool

# Для работы с прогрессом
from app.services.db_service import update_or_create_student_progress

# Узлы графа
CURRICULUM_NODE = "curriculum_agent"
KNOWLEDGE_RETRIEVAL_NODE = "knowledge_retrieval_agent"
ASSESSMENT_NODE = "assessment_agent"
RECOMMENDATION_NODE = "recommendation_agent"
SAVE_PROGRESS_NODE = "save_progress"
HANDLE_ERROR_NODE = "handle_error"


async def curriculum_agent_node(state: TutorGraphState, config: dict):
    # выбор следующей темы и установка статуса in_progress
    result = await run_curriculum_agent(state, config["configurable"]["session_pool"])
    topic_id = result.get("current_topic_id")
    if topic_id and not state.get("error_message"):
        # пометим тему как начатую
        async with config["configurable"]["session_pool"]() as session:
            await update_or_create_student_progress(
                session, state["student_id"], topic_id, status="in_progress"
            )
    return result


async def knowledge_retrieval_node(state: TutorGraphState, config: dict):
    result = await run_knowledge_retrieval_agent(
        state, config["configurable"]["lightrag_tool"]
    )
    if not result.get("retrieved_context") and not result.get("error_message"):
        result["error_message"] = "Не удалось найти релевантную информацию по теме"
    return result


async def assessment_node(state: TutorGraphState, config: dict):
    result = await run_assessment_agent(state, config["configurable"]["llm"])
    if not any(
        [
            result.get("assessment_question"),
            result.get("assessment_feedback"),
            result.get("error_message"),
        ]
    ):
        result["error_message"] = "Не удалось создать вопрос или оценить ответ студента"
    return result


async def recommendation_node(state: TutorGraphState, config: dict):
    result = await run_recommendation_agent(
        state, config["configurable"]["llm"], config["configurable"]["session_pool"]
    )
    if not result.get("recommendations") and not result.get("error_message"):
        result["error_message"] = "Не удалось создать персонализированные рекомендации"
    return result


async def save_progress_node(state: TutorGraphState, config: dict):
    # помечаем тему как выполненную и очищаем временные поля
    async with config["configurable"]["session_pool"]() as session:
        await update_or_create_student_progress(
            session,
            state["student_id"],
            state["current_topic_id"],
            status="completed",
            assessment_results={"feedback": state.get("assessment_feedback")},
        )
    # сброс временных полей
    for key in [
        "retrieved_context",
        "assessment_question",
        "assessment_feedback",
        "student_answer",
        "recommendations",
    ]:
        state[key] = None
    return {}


async def handle_error_node_func(state: TutorGraphState, config: dict):
    err = state.get("error_message", "Неизвестная ошибка")
    # формируем дружелюбный ответ
    if "конфигурации" in err.lower():
        user_msg = "Извините, техническая проблема. Попробуйте позже."
    elif "не удалось найти" in err.lower():
        user_msg = "Не удалось найти информацию. Попробуйте переформулировать."
    else:
        user_msg = "Произошла ошибка. Попробуйте повторить запрос."
    return {"error_message": err, "user_message": user_msg, "error_handled": True}


# Условные маршруты


def should_retrieve_knowledge(state: TutorGraphState) -> str:
    if state.get("error_message"):
        return HANDLE_ERROR_NODE
    if state.get("student_answer"):
        return ASSESSMENT_NODE
    if state.get("current_topic_id") or state.get("input_query"):
        return KNOWLEDGE_RETRIEVAL_NODE
    state["error_message"] = "Нет темы или запроса"
    return HANDLE_ERROR_NODE


def should_assess_or_recommend(state: TutorGraphState) -> str:
    if state.get("error_message"):
        return HANDLE_ERROR_NODE
    if state.get("retrieved_context") and not state.get("assessment_question"):
        return ASSESSMENT_NODE
    if state.get("student_answer"):
        return ASSESSMENT_NODE
    if state.get("assessment_feedback"):
        return RECOMMENDATION_NODE
    return ASSESSMENT_NODE


# Компиляция графа


def create_tutor_graph(
    llm: ChatOpenAI,
    rag_instance: LightRAG,
    session_pool: async_sessionmaker[AsyncSession],
) -> StateGraph:
    # настраиваем инструмент для retrieval
    lightrag_tool = LightRAGRetrieverTool(rag_instance=rag_instance)
    tool_node = ToolNode(
        [lightrag_tool], handle_tool_errors="Ошибка при извлечении информации."
    )
    cfg = {"llm": llm, "session_pool": session_pool, "lightrag_tool": tool_node}

    workflow = StateGraph(TutorGraphState)
    memory = MemorySaver()

    # регистрируем узлы
    workflow.add_node(CURRICULUM_NODE, curriculum_agent_node)
    workflow.add_node(KNOWLEDGE_RETRIEVAL_NODE, knowledge_retrieval_node)
    workflow.add_node(ASSESSMENT_NODE, assessment_node)
    workflow.add_node(RECOMMENDATION_NODE, recommendation_node)
    workflow.add_node(SAVE_PROGRESS_NODE, save_progress_node)
    workflow.add_node(HANDLE_ERROR_NODE, handle_error_node_func)

    # старт
    workflow.add_edge(START, CURRICULUM_NODE)

    # после curriculum
    workflow.add_conditional_edges(
        CURRICULUM_NODE,
        should_retrieve_knowledge,
        {
            KNOWLEDGE_RETRIEVAL_NODE: KNOWLEDGE_RETRIEVAL_NODE,
            HANDLE_ERROR_NODE: HANDLE_ERROR_NODE,
        },
    )

    # после retrieval
    workflow.add_conditional_edges(
        KNOWLEDGE_RETRIEVAL_NODE,
        should_assess_or_recommend,
        {ASSESSMENT_NODE: ASSESSMENT_NODE, HANDLE_ERROR_NODE: HANDLE_ERROR_NODE},
    )

    # после assessment
    workflow.add_conditional_edges(
        ASSESSMENT_NODE,
        lambda s: END
        if s.get("assessment_question")
        else (RECOMMENDATION_NODE if not s.get("error_message") else HANDLE_ERROR_NODE),
        {
            RECOMMENDATION_NODE: RECOMMENDATION_NODE,
            HANDLE_ERROR_NODE: HANDLE_ERROR_NODE,
            END: END,
        },
    )

    # после recommendations — сохраняем прогресс или обрабатываем ошибку
    workflow.add_conditional_edges(
        RECOMMENDATION_NODE,
        lambda s: SAVE_PROGRESS_NODE
        if not s.get("error_message")
        else HANDLE_ERROR_NODE,
        {SAVE_PROGRESS_NODE: SAVE_PROGRESS_NODE, HANDLE_ERROR_NODE: HANDLE_ERROR_NODE},
    )
    # после save_progress — опять к выбору темы или в ошибку
    workflow.add_conditional_edges(
        SAVE_PROGRESS_NODE,
        lambda s: CURRICULUM_NODE if not s.get("error_message") else HANDLE_ERROR_NODE,
        {CURRICULUM_NODE: CURRICULUM_NODE, HANDLE_ERROR_NODE: HANDLE_ERROR_NODE},
    )

    # все ошибки ведут в END
    workflow.add_edge(HANDLE_ERROR_NODE, END)

    # компиляция
    graph = workflow.compile(checkpointer=memory).with_config(configurable=cfg)
    logger.info("Graph compiled with MemorySaver and full cycle.")
    return graph
