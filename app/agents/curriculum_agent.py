import asyncio
from typing import Dict, Optional, Union

from loguru import logger
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agents.state import TutorGraphState
from app.services.db_service import (
    get_student_by_id,
    get_student_curriculum_info_for_agent,
    update_or_create_student_progress,
)

MAX_RETRIES = 3
RETRY_DELAY = 1.0


def validate_student_id(
    student_id: Union[str, int, None]
) -> tuple[bool, Optional[str]]:
    """Валидация student_id"""
    if not student_id:
        return False, "student_id отсутствует в состоянии"

    if isinstance(student_id, str) and not student_id.strip():
        return False, "student_id является пустой строкой"

    try:
        int(student_id)
        return True, None
    except (ValueError, TypeError):
        return False, f"student_id имеет некорректный формат: {student_id}"


def create_error_topic_info(error_type: str, student_id: Union[str, int]) -> Dict:
    """Создание стандартизированной информации о теме при ошибках"""
    error_configs = {
        "student_not_found": {
            "current_topic_id": "student_profile_not_found",
            "current_topic_name": "Ошибка: Профиль не найден",
            "current_learning_objectives": "Невозможно определить цели обучения.",
        },
        "topic_determination_failed": {
            "current_topic_id": "topic_determination_failed",
            "current_topic_name": "Ошибка определения темы",
            "current_learning_objectives": "Не удалось загрузить цели обучения.",
        },
        "database_error": {
            "current_topic_id": "error_topic_curriculum_agent",
            "current_topic_name": "Ошибка определения темы (Database Error)",
            "current_learning_objectives": "Не удалось загрузить цели обучения из-за ошибки БД.",
        },
        "general_error": {
            "current_topic_id": "error_topic_curriculum_agent",
            "current_topic_name": "Ошибка определения темы (Agent Exception)",
            "current_learning_objectives": "Не удалось загрузить цели обучения из-за ошибки.",
        },
    }

    return error_configs.get(error_type, error_configs["general_error"])


async def get_student_profile_with_retry(
    session: AsyncSession, student_id: Union[str, int], max_retries: int = MAX_RETRIES
) -> Optional[Dict]:
    """Получение профиля студента с retry логикой"""
    for attempt in range(max_retries):
        try:
            student_model = await get_student_by_id(session, student_id)
            if student_model:
                return {
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
                    "is_admin": student_model.is_admin,
                }
            return None

        except SQLAlchemyError as e:
            logger.warning(
                f"Попытка {attempt + 1} получения профиля студента не удалась: {e}"
            )
            if attempt < max_retries - 1:
                await asyncio.sleep(RETRY_DELAY * (attempt + 1))
            else:
                logger.error(
                    f"Все {max_retries} попыток получения профиля студента не удались"
                )
                raise e


async def get_curriculum_info_with_retry(
    session: AsyncSession, student_id: Union[str, int], max_retries: int = MAX_RETRIES
) -> Optional[Dict]:
    """Получение информации о учебном плане с retry логикой"""
    for attempt in range(max_retries):
        try:
            curriculum_info = await get_student_curriculum_info_for_agent(
                session, student_id
            )
            return (
                curriculum_info
                if curriculum_info and curriculum_info.get("current_topic_id")
                else None
            )

        except SQLAlchemyError as e:
            logger.warning(
                f"Попытка {attempt + 1} получения учебного плана не удалась: {e}"
            )
            if attempt < max_retries - 1:
                await asyncio.sleep(RETRY_DELAY * (attempt + 1))
            else:
                logger.error(
                    f"Все {max_retries} попыток получения учебного плана не удались"
                )
                raise e


async def run_curriculum_agent(
    state: TutorGraphState, session_pool: async_sessionmaker[AsyncSession]
) -> Dict:
    """
    Узел LangGraph: Агент Учебного Плана.
    Определяет текущее положение студента в учебном плане, используя db_service.
    """
    logger.info("Агент Учебного Плана запущен.")

    student_id = state.get("student_id")
    is_valid, validation_error = validate_student_id(student_id)

    if not is_valid:
        error_message = f"Ошибка валидации: {validation_error}"
        logger.error(error_message)
        return {
            "error_message": error_message,
            **create_error_topic_info("general_error", student_id),
        }

    logger.info(f"Агент Учебного Плана запущен для student_id: {student_id}")

    student_profile_data: Optional[Dict] = None
    current_topic_info: Dict = {}
    error_message: Optional[str] = None

    try:
        async with session_pool() as session:

            try:
                student_profile_data = await get_student_profile_with_retry(
                    session, student_id
                )

                if student_profile_data:
                    logger.info(
                        f"Профиль студента {student_id} получен: {student_profile_data.get('first_name')}"
                    )
                else:
                    logger.warning(f"Профиль студента {student_id} не найден в БД.")
                    error_message = f"Профиль студента {student_id} не найден."
                    current_topic_info = create_error_topic_info(
                        "student_not_found", student_id
                    )

            except SQLAlchemyError as e:
                logger.error(
                    f"Ошибка БД при получении профиля студента {student_id}: {e}"
                )
                error_message = (
                    f"Ошибка базы данных при получении профиля студента: {str(e)}"
                )
                current_topic_info = create_error_topic_info(
                    "database_error", student_id
                )

            # 2. Определить текущую тему, если профиль найден
            if not error_message:
                try:
                    current_topic_info_from_db = await get_curriculum_info_with_retry(
                        session, student_id
                    )

                    if current_topic_info_from_db:
                        current_topic_info = current_topic_info_from_db
                        await update_or_create_student_progress(
                            session,
                            student_id,
                            current_topic_info["current_topic_id"],
                            status="in_progress",
                        )
                        logger.info(
                            f"Определена текущая тема для студента {student_id}: {current_topic_info.get('current_topic_name')}"
                        )
                    else:
                        logger.warning(
                            f"Не удалось определить текущую тему для студента {student_id} из db_service."
                        )
                        error_message = f"Не удалось определить текущую тему для студента {student_id}."
                        current_topic_info = create_error_topic_info(
                            "topic_determination_failed", student_id
                        )

                except SQLAlchemyError as e:
                    logger.error(
                        f"Ошибка БД при получении учебного плана для студента {student_id}: {e}"
                    )
                    error_message = (
                        f"Ошибка базы данных при получении учебного плана: {str(e)}"
                    )
                    current_topic_info = create_error_topic_info(
                        "database_error", student_id
                    )

    except Exception as e:
        logger.error(
            f"Неожиданная ошибка в Агенте Учебного Плана для student_id {student_id}: {e}",
            exc_info=True,
        )
        error_message = f"Внутренняя ошибка при определении учебного плана: {str(e)}"
        current_topic_info = create_error_topic_info("general_error", student_id)

    # Формирование результата
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
