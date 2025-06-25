import datetime
from contextlib import asynccontextmanager
from typing import (  # Dict, Any добавлены
    Any,
    AsyncGenerator,
    Dict,
    List,
    Optional,
    Tuple,
)

from loguru import logger
from models.models import (  # CurriculumTopic, StudentProgress добавлены
    Base,
    CurriculumTopic,
    Document,
    Student,
    StudentProgress,
)
from sqlalchemy import delete, select  # and_ добавлен
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)


def get_db_url(db_config) -> str:
    try:
        return f"postgresql+asyncpg://{db_config.DB_USER}:{db_config.DB_PASSWORD}@{db_config.DB_HOST}:{db_config.DB_PORT}/{db_config.DB_NAME}"
    except AttributeError as e:
        logger.error(f"Ошибка конфигурации БД: {e}")
        raise ValueError(f"Неполная конфигурация для подключения к БД: {e}")


async def create_db_engine_and_session_pool(
    db_url: str, echo: bool = False
) -> Tuple[AsyncEngine, async_sessionmaker[AsyncSession]]:
    try:
        engine = create_async_engine(db_url, echo=echo)
        session_pool = async_sessionmaker(
            bind=engine, class_=AsyncSession, expire_on_commit=False
        )
        logger.info("Движок SQLAlchemy и пул сессий успешно созданы.")
        return engine, session_pool
    except Exception as e:
        logger.critical(
            f"Не удалось создать движок или пул сессий для БД: {e}", exc_info=True
        )
        raise


async def create_tables(engine: AsyncEngine):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Все таблицы успешно созданы/проверены (если их не было).")


@asynccontextmanager
async def get_session(
    session_pool: async_sessionmaker[AsyncSession],
) -> AsyncGenerator[AsyncSession, None]:
    session = session_pool()
    try:
        yield session
        await session.commit()
    except SQLAlchemyError as e:
        await session.rollback()
        logger.error(f"Ошибка SQLAlchemy во время сессии: {e}", exc_info=True)
        raise
    except Exception as e:
        await session.rollback()
        logger.error(f"Неожиданная ошибка во время сессии БД: {e}", exc_info=True)
        raise
    finally:
        await session.close()


async def get_or_create_student(
    session: AsyncSession,
    user_id: int,
    username: Optional[str] = None,
    first_name: Optional[str] = None,
    last_name: Optional[str] = None,
    middle_name: Optional[str] = None,
    full_name: Optional[str] = None,
    program: Optional[str] = None,
    direction: Optional[str] = None,
    course: Optional[int] = None,
    group: Optional[str] = None,
    specialization: Optional[str] = None,
    enrollment_date: Optional[datetime.datetime] = None,
) -> Tuple[Student, bool]:
    try:
        stmt = select(Student).where(Student.user_id == user_id)
        result = await session.execute(stmt)
        student = result.scalar_one_or_none()
        created = False
        if student is None:
            student = Student(
                user_id=user_id,
                username=username,
                first_name=first_name,
                last_name=last_name,
                middle_name=middle_name,
                full_name=full_name,
                program=program,
                direction=direction,
                course=course,
                group=group,
                specialization=specialization,
                enrollment_date=enrollment_date,
                is_admin=False,
            )
            session.add(student)
            await session.flush()
            created = True
            logger.info(f"Создан новый студент: {user_id}")
        else:
            updated_fields = False
            if username is not None and student.username != username:
                student.username = username
                updated_fields = True
            if first_name is not None and student.first_name != first_name:
                student.first_name = first_name
                updated_fields = True
            if last_name is not None and student.last_name != last_name:
                student.last_name = last_name
                updated_fields = True
            if middle_name is not None and student.middle_name != middle_name:
                student.middle_name = middle_name
                updated_fields = True
            if full_name is not None and student.full_name != full_name:
                student.full_name = full_name
                updated_fields = True
            if program is not None and student.program != program:
                student.program = program
                updated_fields = True
            if direction is not None and student.direction != direction:
                student.direction = direction
                updated_fields = True
            if course is not None and student.course != course:
                student.course = course
                updated_fields = True
            if group is not None and student.group != group:
                student.group = group
                updated_fields = True
            if specialization is not None and student.specialization != specialization:
                student.specialization = specialization
                updated_fields = True
            if (
                enrollment_date is not None
                and student.enrollment_date != enrollment_date
            ):
                student.enrollment_date = enrollment_date
                updated_fields = True
            if updated_fields:
                await session.flush()
                logger.info(f"Данные студента {user_id} обновлены.")
        return student, created
    except IntegrityError as e:
        await session.rollback()
        logger.error(f"Ошибка целостности (студент {user_id}): {e}")
        raise
    except Exception as e:
        await session.rollback()
        logger.error(f"Ошибка get_or_create_student ({user_id}): {e}", exc_info=True)
        raise


async def get_student_by_id(session: AsyncSession, user_id: int) -> Optional[Student]:
    stmt = select(Student).where(Student.user_id == user_id)
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def set_user_admin_status(
    session: AsyncSession, user_id: int, is_admin_status: bool
) -> Optional[Student]:
    student = await get_student_by_id(session, user_id)
    if student:
        student.is_admin = is_admin_status
        await session.flush()
        logger.info(
            f"Статус администратора для студента {user_id} изменен на: {is_admin_status}."
        )
        return student
    logger.warning(
        f"Попытка изменить статус админа для несуществующего студента: {user_id}"
    )
    return None


async def get_admin_user_ids_from_db(session: AsyncSession) -> List[int]:
    try:
        stmt = select(Student.user_id).where(Student.is_admin == True)
        result = await session.execute(stmt)
        return result.scalars().all()
    except Exception as e:
        logger.error(f"Ошибка при получении ID администраторов: {e}", exc_info=True)
        return []


async def add_document(
    session: AsyncSession,
    lightrag_id: str,
    file_name: str,
    uploaded_by_tg_id: int,
    status: str = "pending",
) -> Document:
    try:
        existing_doc = await get_document_by_lightrag_id(session, lightrag_id)
        if existing_doc:
            existing_doc.file_name = file_name
            existing_doc.status = status
            await session.flush()
            logger.info(f"Данные документа '{lightrag_id}' обновлены.")
            return existing_doc
        new_document = Document(
            lightrag_id=lightrag_id,
            file_name=file_name,
            status=status,
            uploaded_by_tg_id=uploaded_by_tg_id,
        )
        session.add(new_document)
        await session.flush()
        logger.info(
            f"Документ '{file_name}' (ID: {new_document.lightrag_id}) добавлен студентом {uploaded_by_tg_id}."
        )
        return new_document
    except IntegrityError as e:
        await session.rollback()
        logger.error(
            f"Ошибка целостности (документ '{lightrag_id}', студент {uploaded_by_tg_id}): {e}"
        )
        raise
    except Exception as e:
        await session.rollback()
        logger.error(
            f"Ошибка add_document ('{file_name}', ID: {lightrag_id}): {e}",
            exc_info=True,
        )
        raise


async def get_document_by_lightrag_id(
    session: AsyncSession, lightrag_id: str
) -> Optional[Document]:
    stmt = select(Document).where(Document.lightrag_id == lightrag_id)
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def update_document_status(
    session: AsyncSession, lightrag_id: str, new_status: str
) -> Optional[Document]:
    document = await get_document_by_lightrag_id(session, lightrag_id)
    if document:
        document.status = new_status
        await session.flush()
        logger.info(
            f"Статус документа ID: {lightrag_id} в БД обновлен на '{new_status}'."
        )
        return document
    logger.warning(
        f"Попытка обновить статус несуществующего документа: ID {lightrag_id}"
    )
    return None


async def mark_document_as_deleted_by_admin(
    session: AsyncSession, lightrag_id: str
) -> Optional[Document]:
    return await update_document_status(session, lightrag_id, "deleted_by_admin")


async def get_documents_for_admin_list(
    session: AsyncSession, limit: int = 1000, offset: int = 0
) -> List[Document]:
    stmt = (
        select(Document)
        .where(Document.status != "deleted_by_admin")
        .order_by(Document.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    result = await session.execute(stmt)
    return result.scalars().all()


async def get_all_documents(
    session: AsyncSession, limit: int = 100, offset: int = 0
) -> List[Document]:
    stmt = (
        select(Document)
        .order_by(Document.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    result = await session.execute(stmt)
    return result.scalars().all()


async def get_user_documents(
    session: AsyncSession,
    user_id: int,
    limit: int = 100,
    offset: int = 0,
    include_deleted_by_admin: bool = False,
) -> List[Document]:
    stmt = select(Document).where(Document.uploaded_by_tg_id == user_id)
    if not include_deleted_by_admin:
        stmt = stmt.where(Document.status != "deleted_by_admin")
    stmt = stmt.order_by(Document.created_at.desc()).limit(limit).offset(offset)
    result = await session.execute(stmt)
    return result.scalars().all()


async def hard_delete_document(session: AsyncSession, lightrag_id: str) -> bool:
    stmt = delete(Document).where(Document.lightrag_id == lightrag_id)
    result = await session.execute(stmt)
    if result.rowcount > 0:
        logger.info(f"Документ ID: {lightrag_id} физически удален из БД.")
        return True
    logger.warning(
        f"Попытка физически удалить несуществующий документ: ID {lightrag_id}"
    )
    return False


async def get_curriculum_topic_by_id(
    session: AsyncSession, topic_id: str
) -> Optional[CurriculumTopic]:
    stmt = select(CurriculumTopic).where(CurriculumTopic.topic_id == topic_id)
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def get_student_progress(
    session: AsyncSession, student_id: int, topic_id: str
) -> Optional[StudentProgress]:
    stmt = select(StudentProgress).where(
        StudentProgress.student_id == student_id, StudentProgress.topic_id == topic_id
    )
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def update_or_create_student_progress(
    session: AsyncSession,
    student_id: int,
    topic_id: str,
    status: str,
    assessment_results: Optional[dict] = None,
) -> StudentProgress:
    progress = await get_student_progress(session, student_id, topic_id)
    if progress:
        progress.status = status
        progress.last_accessed = datetime.datetime.utcnow()
        if assessment_results is not None:
            progress.assessment_results = assessment_results
        logger.info(
            f"Прогресс студента {student_id} по теме {topic_id} обновлен: {status}."
        )
    else:
        student = await get_student_by_id(session, student_id)
        topic = await get_curriculum_topic_by_id(session, topic_id)
        if not student:
            raise ValueError(f"Студент {student_id} не найден.")
        if not topic:
            raise ValueError(f"Тема {topic_id} не найдена.")
        progress = StudentProgress(
            student_id=student_id,
            topic_id=topic_id,
            status=status,
            assessment_results=assessment_results,
        )
        session.add(progress)
        logger.info(
            f"Создан прогресс для студента {student_id} по теме {topic_id}: {status}."
        )
    await session.flush()
    return progress


async def get_student_curriculum_info_for_agent(
    session: AsyncSession, student_id: int
) -> Dict[str, Any]:
    """
    Определяет текущую/следующую тему для студента.
    Логика: ищет первую тему без статуса 'completed', учитывая порядок и (опционально) пререквизиты.
    """
    logger.debug(f"Определение текущей темы для студента {student_id}")

    # 1. Получить все темы, отсортированные по порядку (предполагаем, что поле 'order' существует и заполнено)
    all_topics_stmt = select(CurriculumTopic).order_by(
        CurriculumTopic.parent_topic_id.asc(),
        CurriculumTopic.order.asc(),
        CurriculumTopic.topic_id.asc(),
    )
    all_topics_result = await session.execute(all_topics_stmt)
    all_topics: List[CurriculumTopic] = all_topics_result.scalars().all()

    if not all_topics:
        logger.warning("Учебный план пуст. Невозможно определить тему.")
        return {
            "current_topic_id": "no_topics_available",
            "current_topic_name": "Учебный план пуст",
            "current_learning_objectives": "Обратитесь к администратору.",
        }

    # 2. Получить весь прогресс студента по всем темам
    progress_stmt = select(StudentProgress).where(
        StudentProgress.student_id == student_id
    )
    progress_result = await session.execute(progress_stmt)
    student_progress_list: List[StudentProgress] = progress_result.scalars().all()
    progress_map: Dict[str, StudentProgress] = {
        p.topic_id: p for p in student_progress_list
    }

    # 3. Найти первую не пройденную тему
    current_topic: Optional[CurriculumTopic] = None
    for topic in all_topics:
        topic_progress = progress_map.get(topic.topic_id)
        if not topic_progress or topic_progress.status != "completed":
            # Дополнительно: Проверка пререквизитов (если они есть)
            # prerequisites_ok = True
            # if topic.prerequisites:
            #     for prereq_id in topic.prerequisites:
            #         prereq_progress = progress_map.get(prereq_id)
            #         if not prereq_progress or prereq_progress.status != "completed":
            #             prerequisites_ok = False
            #             break
            # if prerequisites_ok:
            current_topic = topic
            break

    if current_topic:
        logger.info(
            f"Найдена текущая тема для студента {student_id}: {current_topic.topic_name} (ID: {current_topic.topic_id})"
        )
        return {
            "current_topic_id": current_topic.topic_id,
            "current_topic_name": current_topic.topic_name,
            "current_learning_objectives": current_topic.learning_objectives
            or "Цели обучения не указаны.",
        }
    else:
        logger.info(
            f"Все темы для студента {student_id} пройдены или не удалось определить следующую."
        )
        return {
            "current_topic_id": "all_topics_completed",
            "current_topic_name": "Все темы пройдены!",
            "current_learning_objectives": "Поздравляем с завершением учебного плана!",
        }


async def get_student_progress_summary_for_recommendations_agent(
    session: AsyncSession, student_id: int, current_topic_id: Optional[str]
) -> str:
    """Собирает сводку прогресса студента для Агента Рекомендаций."""
    logger.debug(
        f"Сбор сводки прогресса для студента {student_id}, тема: {current_topic_id}"
    )
    summary_parts = []

    student = await get_student_by_id(session, student_id)
    if not student:
        return "Профиль студента не найден."

    summary_parts.append(
        f"Студент: {student.first_name or student.username or student.user_id}."
    )
    if student.full_name:
        summary_parts.append(f"Полное имя: {student.full_name}.")
    if student.course:
        summary_parts.append(f"Курс: {student.course}.")
    if student.group:
        summary_parts.append(f"Группа: {student.group}.")
    if student.program:
        summary_parts.append(f"Программа: {student.program}.")
    if student.direction:
        summary_parts.append(f"Направление: {student.direction}.")

    if current_topic_id:
        topic = await get_curriculum_topic_by_id(session, current_topic_id)
        if topic:
            summary_parts.append(f'Текущая тема для анализа: "{topic.topic_name}".')
            progress = await get_student_progress(session, student_id, current_topic_id)
            if progress:
                summary_parts.append(f"Статус по этой теме: '{progress.status}'.")
                if progress.assessment_results:
                    summary_parts.append(
                        f"Последние результаты оценки: {progress.assessment_results}."
                    )
                summary_parts.append(
                    f"Последнее обращение к теме: {progress.last_accessed.strftime('%Y-%m-%d %H:%M') if progress.last_accessed else 'нет данных'}."
                )
            else:
                summary_parts.append("Прогресс по данной теме еще не начат.")

            # Информация о пререквизитах
            if topic.prerequisites and isinstance(topic.prerequisites, list):
                summary_parts.append(f'Пререквизиты для "{topic.topic_name}":')
                for prereq_id in topic.prerequisites:
                    prereq_topic = await get_curriculum_topic_by_id(session, prereq_id)
                    if prereq_topic:
                        prereq_name = prereq_topic.topic_name
                        prereq_progress = await get_student_progress(
                            session, student_id, prereq_id
                        )
                        if prereq_progress:
                            summary_parts.append(
                                f"  - \"{prereq_name}\" (ID: {prereq_id}): статус '{prereq_progress.status}'."
                            )
                        else:
                            summary_parts.append(
                                f'  - "{prereq_name}" (ID: {prereq_id}): прогресс не начат.'
                            )
                    else:
                        summary_parts.append(
                            f"  - Тема-пререквизит с ID '{prereq_id}' не найдена."
                        )
        else:
            summary_parts.append(
                f"Тема с ID '{current_topic_id}' не найдена в учебном плане."
            )
    else:
        summary_parts.append("Конкретная текущая тема для анализа не указана.")

    # Можно добавить общую успеваемость или количество пройденных тем
    # completed_topics_count_stmt = select(func.count(StudentProgress.id)).where(
    #     StudentProgress.student_id == student_id,
    #     StudentProgress.status == "completed"
    # )
    # completed_count_res = await session.execute(completed_topics_count_stmt)
    # completed_count = completed_count_res.scalar_one_or_none()
    # if completed_count is not None:
    #     summary_parts.append(f"Всего пройденных тем: {completed_count}.")

    if not summary_parts:
        return "Детальная информация о прогрессе для рекомендаций отсутствует."
    return "\n".join(summary_parts)
