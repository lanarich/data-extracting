from typing import Dict, List, Optional

from loguru import logger

from app.agents.state import TutorGraphState
from app.tools.lightrag_retriever_tool import (  # Импортируем наш инструмент
    LightRAGRetrieverTool,
)


async def run_knowledge_retrieval_agent(
    state: TutorGraphState, lightrag_tool: LightRAGRetrieverTool
) -> Dict:
    """
    Узел LangGraph: Агент Извлечения Знаний.
    Использует LightRAGRetrieverTool для поиска релевантного контекста.
    """
    logger.info("Агент Извлечения Знаний запущен.")

    query_for_retrieval: Optional[str] = None
    retrieved_context: Optional[List[str]] = None
    error_message: Optional[str] = None

    # Определяем, что использовать для запроса: input_query или current_topic_name
    if state.get("input_query"):
        query_for_retrieval = state["input_query"]
        logger.info(f"Запрос для извлечения из input_query: '{query_for_retrieval}'")
    elif state.get("current_topic_name"):
        query_for_retrieval = state["current_topic_name"]
        logger.info(
            f"Запрос для извлечения из current_topic_name: '{query_for_retrieval}'"
        )
    else:
        error_message = "Отсутствует тема или запрос для извлечения знаний."
        logger.warning(error_message)

    if query_for_retrieval:
        try:
            # Используем LightRAGRetrieverTool
            # top_k можно брать из state или задать по умолчанию
            top_k_retrieval = state.get(
                "retrieval_top_k", 5
            )  # Пример получения top_k из состояния

            # Инструмент ожидает именованные аргументы, если его args_schema это Pydantic модель
            # В нашем случае LightRAGRetrieverInput ожидает 'query' и опционально 'top_k'
            retrieved_context = await lightrag_tool.arun(
                tool_input={"query": query_for_retrieval, "top_k": top_k_retrieval}
            )
            # Или если arun напрямую принимает kwargs:
            # retrieved_context = await lightrag_tool.arun(query=query_for_retrieval, top_k=top_k_retrieval)
            # Проверьте, как ваш BaseTool.arun сконфигурирован для приема аргументов.
            # Стандартно для LangChain, если args_schema определен, arun(tool_input: Dict) или arun(query="...", top_k=...)

            if retrieved_context and not (
                len(retrieved_context) == 1
                and "Ошибка при извлечении документов:" in retrieved_context[0]
            ):
                logger.info(f"Извлечено {len(retrieved_context)} фрагментов контекста.")
            elif not retrieved_context:
                logger.warning(
                    f"Контекст не был извлечен для запроса: '{query_for_retrieval}'."
                )
                retrieved_context = []  # Возвращаем пустой список, а не None
            else:  # Если вернулась ошибка от инструмента
                logger.error(f"Ошибка от LightRAGRetrieverTool: {retrieved_context[0]}")
                error_message = retrieved_context[0]
                retrieved_context = []

        except Exception as e:
            logger.error(f"Ошибка при вызове LightRAGRetrieverTool: {e}", exc_info=True)
            error_message = f"Внутренняя ошибка при извлечении знаний: {e}"
            retrieved_context = []
    else:
        # Если не было запроса, контекст остается None или пустым списком
        retrieved_context = []

    update_data = {
        "retrieved_context": retrieved_context,
        "error_message": state.get("error_message") or error_message,
    }
    logger.debug(
        f"Агент Извлечения Знаний обновил состояние: retrieved_context (кол-во: {len(retrieved_context or [])})"
    )
    return update_data
