import asyncio
from typing import Dict, List, Optional, Union

from loguru import logger

from app.agents.state import TutorGraphState
from app.tools.lightrag_retriever_tool import LightRAGRetrieverTool

# Константы для конфигурации
MAX_RETRIES = 3
RETRY_DELAY = 1.0
DEFAULT_TOP_K = 5
MAX_QUERY_LENGTH = 1000


def validate_inputs(
    state: TutorGraphState, lightrag_tool: LightRAGRetrieverTool
) -> Dict[str, Union[str, List[str]]]:
    """Валидация входных данных"""
    errors = []

    # Проверяем наличие инструмента
    if not lightrag_tool or not isinstance(lightrag_tool, LightRAGRetrieverTool):
        errors.append("LightRAG инструмент отсутствует или имеет неправильный тип")

    # Проверяем наличие запроса или темы
    input_query = state.get("input_query")
    current_topic_name = state.get("current_topic_name")

    if not input_query and not current_topic_name:
        errors.append("Отсутствует тема или запрос для извлечения знаний")

    # Определяем запрос для поиска
    query_for_retrieval = None
    if input_query:
        if not isinstance(input_query, str) or not input_query.strip():
            errors.append("input_query должен быть непустой строкой")
        else:
            query_for_retrieval = input_query.strip()
    elif current_topic_name:
        if not isinstance(current_topic_name, str) or not current_topic_name.strip():
            errors.append("current_topic_name должен быть непустой строкой")
        else:
            query_for_retrieval = current_topic_name.strip()

    # Проверяем длину запроса
    if query_for_retrieval and len(query_for_retrieval) > MAX_QUERY_LENGTH:
        errors.append(f"Запрос слишком длинный (максимум {MAX_QUERY_LENGTH} символов)")
        query_for_retrieval = query_for_retrieval[:MAX_QUERY_LENGTH]

    # Проверяем top_k
    top_k = state.get("retrieval_top_k", DEFAULT_TOP_K)
    if not isinstance(top_k, int) or top_k <= 0:
        logger.warning(
            f"Некорректное значение top_k: {top_k}, используем значение по умолчанию"
        )
        top_k = DEFAULT_TOP_K

    return {
        "errors": errors,
        "query_for_retrieval": query_for_retrieval,
        "top_k": top_k,
        "query_source": "input_query" if input_query else "current_topic_name",
    }


async def retrieve_with_retry(
    lightrag_tool: LightRAGRetrieverTool,
    query: str,
    top_k: int,
    max_retries: int = MAX_RETRIES,
) -> List[str]:
    """Извлечение контекста с retry логикой"""
    for attempt in range(max_retries):
        try:
            # Используем правильный API инструмента
            retrieved_context = await lightrag_tool._arun(query=query, top_k=top_k)

            # Проверяем результат
            if isinstance(retrieved_context, list):
                return retrieved_context
            elif isinstance(retrieved_context, str):
                return [retrieved_context]
            else:
                logger.warning(f"Неожиданный тип результата: {type(retrieved_context)}")
                return []

        except Exception as e:
            logger.warning(f"Попытка {attempt + 1} извлечения не удалась: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(RETRY_DELAY * (attempt + 1))
            else:
                logger.error(f"Все {max_retries} попыток извлечения не удались")
                raise e


def filter_and_validate_context(retrieved_context: List[str]) -> List[str]:
    """Фильтрация и валидация извлеченного контекста"""
    if not retrieved_context:
        return []

    # Проверяем на ошибки в результатах
    filtered_context = []
    for item in retrieved_context:
        if isinstance(item, str) and item.strip():
            # Проверяем, не является ли это сообщением об ошибке
            if not item.startswith("Ошибка при"):
                filtered_context.append(item.strip())
            else:
                logger.warning(f"Обнаружена ошибка в результате: {item}")

    return filtered_context


async def run_knowledge_retrieval_agent(
    state: TutorGraphState, lightrag_tool: LightRAGRetrieverTool
) -> Dict:
    """
    Узел LangGraph: Агент Извлечения Знаний.
    Использует LightRAGRetrieverTool для поиска релевантного контекста.
    """
    logger.info("Агент Извлечения Знаний запущен.")

    # Валидация входных данных
    validation_result = validate_inputs(state, lightrag_tool)

    if validation_result["errors"]:
        error_message = f"Ошибки валидации: {'; '.join(validation_result['errors'])}"
        logger.error(error_message)
        return {"retrieved_context": [], "error_message": error_message}

    query_for_retrieval = validation_result["query_for_retrieval"]
    top_k = validation_result["top_k"]
    query_source = validation_result["query_source"]

    logger.info(
        f"Запрос для извлечения из {query_source}: '{query_for_retrieval[:100]}...'"
    )

    retrieved_context: List[str] = []
    error_message: Optional[str] = None

    try:
        # Извлекаем контекст с retry логикой
        raw_context = await retrieve_with_retry(
            lightrag_tool, query_for_retrieval, top_k
        )

        # Фильтруем и валидируем результат
        retrieved_context = filter_and_validate_context(raw_context)

        if retrieved_context:
            logger.info(
                f"Успешно извлечено {len(retrieved_context)} фрагментов контекста"
            )
        else:
            logger.warning(
                f"Контекст не был извлечен для запроса: '{query_for_retrieval}'"
            )
            error_message = "Не удалось найти релевантную информацию по запросу"

    except Exception as e:
        logger.error(f"Критическая ошибка при извлечении знаний: {e}", exc_info=True)
        error_message = f"Внутренняя ошибка при извлечении знаний: {str(e)}"
        retrieved_context = []

    # Формирование результата
    update_data = {
        "retrieved_context": retrieved_context,
        "error_message": state.get("error_message") or error_message,
        "retrieval_query": query_for_retrieval,  # Сохраняем запрос для отладки
        "retrieval_source": query_source,  # Сохраняем источник запроса
    }

    logger.debug(
        f"Агент Извлечения Знаний завершен. Контекст: {len(retrieved_context)} фрагментов"
    )
    return update_data
