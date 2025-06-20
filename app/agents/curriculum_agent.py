from typing import Dict, Optional

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agents.state import TutorGraphState

# Импортируем реальные функции из db_service
from app.services.db_service import (  # Было get_user_by_id, теперь get_student_by_id; Новая функция из db_service
    get_student_by_id,
    get_student_curriculum_info_for_agent,
)

# Заглушки get_student_profile_from_db и determine_current_topic_for_student больше не нужны здесь,
# так как их логика перенесена или будет в get_student_curriculum_info_for_agent и get_student_by_id.


async def run_curriculum_agent(
    state: TutorGraphState, session_pool: async_sessionmaker[AsyncSession]
) -> Dict:
    """
    Узел LangGraph: Агент Учебного Плана.
    Определяет текущее положение студента в учебном плане, используя db_service.
    """
    logger.info(f"Агент Учебного Плана запущен для student_id: {state['student_id']}")

    student_id = state["student_id"]
    student_profile_data: Optional[Dict] = None
    current_topic_info: Dict = {}
    error_message: Optional[str] = None

    try:
        async with session_pool() as session:  # Используем session_pool напрямую
            # 1. Получить профиль студента
            student_model = await get_student_by_id(session, student_id)
            if student_model:
                student_profile_data = {
                    "user_id": student_model.user_id,
                    "username": student_model.username,
                    "first_name": student_model.first_name,
                    "last_name": student_model.last_name,
                    "course": student_model.course,
                    "group": student_model.group,
                    "specialization": student_model.specialization,
                    "enrollment_date": str(student_model.enrollment_date)
                    if student_model.enrollment_date
                    else None,
                    "is_admin": student_model.is_admin
                    # Добавьте другие поля из модели Student, если они нужны в состоянии графа
                }
                logger.info(
                    f"Профиль студента {student_id} получен: {student_profile_data.get('first_name')}"
                )
            else:
                logger.warning(f"Профиль студента {student_id} не найден в БД.")
                error_message = f"Профиль студента {student_id} не найден."
                # Устанавливаем значения по умолчанию или специфичные для ошибки
                current_topic_info = {
                    "current_topic_id": "student_profile_not_found",
                    "current_topic_name": "Ошибка: Профиль не найден",
                    "current_learning_objectives": "Невозможно определить цели обучения.",
                }

            if not error_message:
                # 2. Определить текущую тему, используя функцию из db_service
                # Эта функция должна вернуть словарь с current_topic_id, current_topic_name, current_learning_objectives
                current_topic_info_from_db = (
                    await get_student_curriculum_info_for_agent(session, student_id)
                )

                if current_topic_info_from_db and current_topic_info_from_db.get(
                    "current_topic_id"
                ):
                    current_topic_info = current_topic_info_from_db
                    logger.info(
                        f"Определена текущая тема для студента {student_id}: {current_topic_info.get('current_topic_name')}"
                    )
                else:
                    logger.warning(
                        f"Не удалось определить текущую тему для студента {student_id} из db_service."
                    )
                    error_message = (
                        f"Не удалось определить текущую тему для студента {student_id}."
                    )
                    current_topic_info = {  # Заполняем, чтобы избежать KeyError ниже
                        "current_topic_id": "topic_determination_failed",
                        "current_topic_name": "Ошибка определения темы",
                        "current_learning_objectives": "Не удалось загрузить цели обучения.",
                    }

    except Exception as e:
        logger.error(
            f"Ошибка в Агенте Учебного Плана для student_id {student_id}: {e}",
            exc_info=True,
        )
        error_message = f"Внутренняя ошибка при определении учебного плана: {str(e)}"
        current_topic_info = {
            "current_topic_id": "error_topic_curriculum_agent",
            "current_topic_name": "Ошибка определения темы (Agent Exception)",
            "current_learning_objectives": "Не удалось загрузить цели обучения из-за ошибки.",
        }

    update_data = {
        "student_profile": student_profile_data,
        "current_topic_id": current_topic_info.get("current_topic_id"),
        "current_topic_name": current_topic_info.get("current_topic_name"),
        "current_learning_objectives": current_topic_info.get(
            "current_learning_objectives"
        ),
        "error_message": state.get("error_message") or error_message,
    }
    logger.debug(f"Агент Учебного Плана обновил состояние: {update_data}")
    return update_data
