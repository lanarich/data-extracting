import asyncio
from typing import Dict, List, Optional, Union

from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from loguru import logger
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agents.state import TutorGraphState
from app.services.db_service import (
    get_student_progress_summary_for_recommendations_agent,
)

# Константы для конфигурации
MAX_CONTEXT_LENGTH = 8000
MAX_RETRIES = 3
RETRY_DELAY = 1.0

GENERATE_RECOMMENDATIONS_SYSTEM_PROMPT = """
Ты - ИИ-Ассистент в роли опытного преподавателя. Твоя задача - проанализировать успеваемость студента по теме и дать ему персонализированные рекомендации для дальнейшего обучения.
Учитывай цели обучения, результаты предыдущей оценки (если есть) и возможные пробелы в знаниях.
Предложи конкретные шаги: что повторить, какие материалы изучить дополнительно, на что обратить особое внимание.
Если есть информация о пререквизитах, и студент показал слабые знания по текущей теме, порекомендуй также повторить пререквизиты.
Отвечай на русском языке, будь ободряющим и конструктивным.

Информация о текущей теме:
Название: {topic_name}
Цели обучения: {learning_objectives}

Контекст (учебный материал по теме, если есть):
{context}

Обратная связь по последнему ответу студента (если была оценка):
{assessment_feedback}

Данные о прогрессе студента (если есть, например, предыдущие попытки, статус по пререквизитам):
{student_progress_summary}

Твои рекомендации для студента:
"""


def validate_state_inputs(state: TutorGraphState) -> Dict[str, Union[str, List[str]]]:
    """Валидация входных данных из состояния"""
    errors = []

    student_id = state.get("student_id")
    if not student_id:
        errors.append("student_id отсутствует в состоянии")
    elif isinstance(student_id, str) and not student_id.strip():
        errors.append("student_id является пустой строкой")

    current_topic_name = state.get("current_topic_name")
    if not current_topic_name or not isinstance(current_topic_name, str):
        errors.append("Отсутствует или некорректное название темы")

    learning_objectives = state.get("current_learning_objectives")
    if not learning_objectives or not isinstance(learning_objectives, str):
        errors.append("Отсутствуют или некорректные цели обучения")

    return {
        "errors": errors,
        "student_id": student_id,
        "current_topic_name": current_topic_name or "Текущая тема",
        "learning_objectives": learning_objectives or "Цели обучения не указаны.",
    }


def prepare_context(context_list: Optional[List[str]]) -> str:
    """Подготовка контекста с ограничением длины"""
    if not context_list:
        return "Учебный материал по теме не загружен."

    full_context = "\n\n".join(context_list)

    if len(full_context) <= MAX_CONTEXT_LENGTH:
        return full_context

    # Обрезаем контекст по элементам
    truncated_context = ""
    for item in context_list:
        if len(truncated_context + item) <= MAX_CONTEXT_LENGTH:
            truncated_context += item + "\n\n"
        else:
            break

    if not truncated_context:
        truncated_context = context_list[0][:MAX_CONTEXT_LENGTH] + "..."

    logger.warning(
        f"Контекст обрезан с {len(full_context)} до {len(truncated_context)} символов"
    )
    return truncated_context


async def get_progress_summary_with_retry(
    session_pool: async_sessionmaker[AsyncSession],
    student_id: Union[str, int],
    current_topic_id: Optional[str],
    max_retries: int = MAX_RETRIES,
) -> str:
    """Получение сводки прогресса с retry логикой"""
    for attempt in range(max_retries):
        try:
            async with session_pool() as session:
                summary = await get_student_progress_summary_for_recommendations_agent(
                    session, student_id, current_topic_id
                )
                return summary or "Информация о прогрессе не загружена."

        except SQLAlchemyError as e:
            logger.warning(f"Попытка {attempt + 1} получения прогресса не удалась: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(RETRY_DELAY * (attempt + 1))
            else:
                logger.error(
                    f"Все {max_retries} попыток получения прогресса не удались"
                )
                raise e


async def generate_recommendations_with_retry(
    chain, inputs: Dict, max_retries: int = MAX_RETRIES
) -> Optional[str]:
    """Генерация рекомендаций с retry логикой"""
    for attempt in range(max_retries):
        try:
            response = await chain.ainvoke(inputs)
            return response.content
        except Exception as e:
            logger.warning(
                f"Попытка {attempt + 1} генерации рекомендаций не удалась: {e}"
            )
            if attempt < max_retries - 1:
                await asyncio.sleep(RETRY_DELAY * (attempt + 1))
            else:
                logger.error(
                    f"Все {max_retries} попыток генерации рекомендаций не удались"
                )
                raise e


def parse_recommendations(generated_text: str) -> List[str]:
    """Умная обработка сгенерированных рекомендаций"""
    if not generated_text or not generated_text.strip():
        return ["Не удалось сгенерировать рекомендации в данный момент."]

    # Разбиваем по строкам и фильтруем
    lines = [line.strip() for line in generated_text.split("\n") if line.strip()]

    # Если получили только одну длинную строку, пытаемся разбить по предложениям
    if len(lines) == 1 and len(lines[0]) > 200:
        sentences = [s.strip() + "." for s in lines[0].split(".") if s.strip()]
        if len(sentences) > 1:
            return sentences

    return (
        lines if lines else ["Не удалось сгенерировать рекомендации в данный момент."]
    )


async def run_recommendation_agent(
    state: TutorGraphState,
    llm: ChatOpenAI,
    session_pool: async_sessionmaker[AsyncSession],
) -> Dict:
    """
    Узел LangGraph: Агент Рекомендаций.
    Генерирует персонализированные рекомендации для студента, используя db_service.
    """
    logger.info("Агент Рекомендаций запущен.")

    # Валидация входных данных
    validation_result = validate_state_inputs(state)
    if validation_result["errors"]:
        error_message = f"Ошибки валидации: {'; '.join(validation_result['errors'])}"
        logger.error(error_message)
        return {
            "recommendations": [
                "К сожалению, произошла ошибка при подготовке рекомендаций."
            ],
            "error_message": error_message,
        }

    student_id = validation_result["student_id"]
    current_topic_name = validation_result["current_topic_name"]
    learning_objectives = validation_result["learning_objectives"]

    current_topic_id = state.get("current_topic_id")
    retrieved_context_list = state.get("retrieved_context")
    assessment_feedback = state.get(
        "assessment_feedback", "Оценка предыдущего ответа отсутствует."
    )

    # Подготовка контекста
    context_str = prepare_context(retrieved_context_list)

    recommendations_list: List[str] = []
    error_message: Optional[str] = None

    try:
        # 1. Получаем сводку о прогрессе студента с retry логикой
        try:
            student_progress_summary_str = await get_progress_summary_with_retry(
                session_pool, student_id, current_topic_id
            )
            logger.info("Сводка прогресса успешно получена")
        except SQLAlchemyError as e:
            logger.error(f"Ошибка БД при получении прогресса: {e}")
            student_progress_summary_str = "Информация о прогрессе временно недоступна."

        # 2. Генерируем рекомендации с помощью LLM
        prompt_template_recommend = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=GENERATE_RECOMMENDATIONS_SYSTEM_PROMPT),
            ]
        )
        recommendation_chain = prompt_template_recommend | llm

        inputs = {
            "topic_name": current_topic_name,
            "learning_objectives": learning_objectives,
            "context": context_str,
            "assessment_feedback": assessment_feedback,
            "student_progress_summary": student_progress_summary_str,
        }

        generated_recommendations = await generate_recommendations_with_retry(
            recommendation_chain, inputs
        )

        if generated_recommendations:
            recommendations_list = parse_recommendations(generated_recommendations)
            logger.info(
                f"Успешно сгенерировано {len(recommendations_list)} рекомендаций"
            )
        else:
            recommendations_list = [
                "Не удалось сгенерировать рекомендации в данный момент."
            ]
            logger.warning("LLM не вернул текст рекомендаций.")

    except Exception as e:
        logger.error(f"Критическая ошибка в Агенте Рекомендаций: {e}", exc_info=True)
        error_message = f"Внутренняя ошибка при генерации рекомендаций: {str(e)}"
        recommendations_list = [
            "К сожалению, произошла ошибка при подготовке рекомендаций."
        ]

    # Формирование результата
    update_data = {
        "recommendations": recommendations_list,
        "error_message": state.get("error_message") or error_message,
        "assessment_question": None,
        "assessment_feedback": None,
    }

    logger.debug(
        f"Агент Рекомендаций завершен. Рекомендации: {len(recommendations_list)}"
    )
    return update_data
