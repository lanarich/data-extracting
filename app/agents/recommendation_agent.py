from typing import Dict, List, Optional

from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agents.state import TutorGraphState

# Импортируем реальную функцию из db_service
from app.services.db_service import (
    get_student_progress_summary_for_recommendations_agent,
)

# Системный промпт для LLM (остается как был)
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

# Заглушка get_student_progress_summary_for_recommendations больше не нужна здесь


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

    student_id = state["student_id"]
    current_topic_id = state.get("current_topic_id")
    current_topic_name = state.get("current_topic_name", "Текущая тема")
    learning_objectives = state.get(
        "current_learning_objectives", "Цели обучения не указаны."
    )
    retrieved_context_list = state.get("retrieved_context")
    assessment_feedback = state.get(
        "assessment_feedback", "Оценка предыдущего ответа отсутствует."
    )

    context_str = (
        "\n\n".join(retrieved_context_list)
        if retrieved_context_list
        else "Учебный материал по теме не загружен."
    )

    recommendations_list: Optional[List[str]] = None
    error_message: Optional[str] = None
    student_progress_summary_str = "Информация о прогрессе не загружена."

    try:
        # 1. Получаем сводку о прогрессе студента из db_service
        async with session_pool() as session:
            student_progress_summary_str = (
                await get_student_progress_summary_for_recommendations_agent(
                    session, student_id, current_topic_id
                )
            )
        logger.info(
            f"Сводка прогресса для рекомендаций: {student_progress_summary_str}"
        )

        # 2. Генерируем рекомендации с помощью LLM
        prompt_template_recommend = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=GENERATE_RECOMMENDATIONS_SYSTEM_PROMPT),
            ]
        )
        recommendation_chain = prompt_template_recommend | llm

        response = await recommendation_chain.ainvoke(
            {
                "topic_name": current_topic_name,
                "learning_objectives": learning_objectives,
                "context": context_str,
                "assessment_feedback": assessment_feedback,
                "student_progress_summary": student_progress_summary_str,
            }
        )

        generated_recommendations = response.content
        if generated_recommendations:
            recommendations_list = [
                rec.strip()
                for rec in generated_recommendations.split("\n")
                if rec.strip()
            ]
            logger.info(f"Сгенерированы рекомендации LLM: {recommendations_list}")
        else:
            recommendations_list = [
                "Не удалось сгенерировать рекомендации в данный момент."
            ]
            logger.warning("LLM не вернул текст рекомендаций.")

    except Exception as e:
        logger.error(f"Ошибка в Агенте Рекомендаций: {e}", exc_info=True)
        error_message = f"Внутренняя ошибка при генерации рекомендаций: {str(e)}"
        recommendations_list = [
            "К сожалению, произошла ошибка при подготовке рекомендаций."
        ]

    update_data = {
        "recommendations": recommendations_list,
        "error_message": state.get("error_message") or error_message,
        "assessment_question": None,
        "assessment_feedback": None,
    }
    logger.debug(
        f"Агент Рекомендаций обновил состояние: рекомендации (кол-во: {len(recommendations_list or [])})"
    )
    return update_data
