from contextlib import asynccontextmanager
from typing import AsyncGenerator, List, Optional, Tuple

from loguru import logger
from models.models import Base, Document, User
from sqlalchemy import delete, select
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
        logger.error(
            f"Ошибка конфигурации БД: отсутствует один из необходимых атрибутов (DB_USER, DB_PASSWORD, etc.). {e}"
        )
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
    logger.info("Все таблицы успешно созданы (если их не было).")


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


async def get_or_create_user(
    session: AsyncSession,
    user_id: int,
    username: Optional[str] = None,
    first_name: Optional[str] = None,
    last_name: Optional[str] = None
    # Параметр role удален
) -> Tuple[User, bool]:
    """
    Получает пользователя по user_id или создает нового.
    Новые пользователи по умолчанию НЕ являются администраторами (is_admin=False).
    """
    try:
        stmt = select(User).where(User.user_id == user_id)
        result = await session.execute(stmt)
        user = result.scalar_one_or_none()

        created = False
        if user is None:
            user = User(
                user_id=user_id,
                username=username,
                first_name=first_name,
                last_name=last_name,
                is_admin=False,
            )
            session.add(user)
            await session.flush()
            created = True
            logger.info(
                f"Создан новый пользователь: {user_id}, username: {username}, is_admin: {user.is_admin}"
            )
        else:
            updated_fields = False
            if username is not None and user.username != username:
                user.username = username
                updated_fields = True
            if first_name is not None and user.first_name != first_name:
                user.first_name = first_name
                updated_fields = True
            if last_name is not None and user.last_name != last_name:
                user.last_name = last_name
                updated_fields = True
            if updated_fields:
                await session.flush()
                logger.info(
                    f"Данные пользователя {user_id} обновлены. Его текущий статус is_admin: {user.is_admin}"
                )
        return user, created
    except IntegrityError as e:
        await session.rollback()
        logger.error(
            f"Ошибка целостности при создании/получении пользователя {user_id}: {e}"
        )
        raise
    except Exception as e:
        await session.rollback()
        logger.error(
            f"Ошибка при создании/получении пользователя {user_id}: {e}", exc_info=True
        )
        raise


async def get_user_by_id(session: AsyncSession, user_id: int) -> Optional[User]:
    """Получает пользователя по его Telegram ID, включая его статус is_admin."""
    stmt = select(User).where(User.user_id == user_id)
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def set_user_admin_status(
    session: AsyncSession, user_id: int, is_admin_status: bool
) -> Optional[User]:
    """
    Устанавливает или снимает статус администратора для пользователя.
    (Может использоваться программно, если потребуется, помимо pgAdmin)
    """
    user = await get_user_by_id(session, user_id)
    if user:
        user.is_admin = is_admin_status
        await session.flush()
        logger.info(
            f"Статус администратора для пользователя {user_id} изменен на: {is_admin_status}."
        )
        return user
    logger.warning(
        f"Попытка изменить статус администратора для несуществующего пользователя: {user_id}"
    )
    return None


async def get_admin_user_ids_from_db(session: AsyncSession) -> List[int]:
    """
    Получает список ID всех пользователей, у которых is_admin = True.
    """
    try:
        stmt = select(User.user_id).where(User.is_admin == True)
        result = await session.execute(stmt)
        admin_ids = result.scalars().all()
        logger.debug(f"Получены ID администраторов из БД: {admin_ids}")
        return admin_ids
    except Exception as e:
        logger.error(
            f"Ошибка при получении ID администраторов из БД: {e}", exc_info=True
        )
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
            logger.warning(
                f"Документ с lightrag_id='{lightrag_id}' уже существует. Обновление существующего."
            )
            existing_doc.file_name = file_name
            existing_doc.status = status
            existing_doc.uploaded_by_tg_id = uploaded_by_tg_id
            await session.flush()
            logger.info(f"Данные существующего документа '{lightrag_id}' обновлены.")
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
            f"Документ '{file_name}' (lightrag_id: {new_document.lightrag_id}, status: {status}) добавлен пользователем {uploaded_by_tg_id}."
        )
        return new_document
    except IntegrityError as e:
        await session.rollback()
        logger.error(
            f"Ошибка целостности при добавлении документа '{lightrag_id}': {e}"
        )
        raise
    except Exception as e:
        await session.rollback()
        logger.error(
            f"Ошибка при добавлении документа '{file_name}' (lightrag_id: {lightrag_id}): {e}",
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
    allowed_statuses = [
        "pending",
        "processing",
        "processed",
        "failed",
        "deleted_by_admin",
    ]
    if new_status not in allowed_statuses:
        logger.error(
            f"Недопустимый статус '{new_status}' для документа {lightrag_id}. Допустимые: {', '.join(allowed_statuses)}."
        )
        raise ValueError(f"Недопустимый статус для БД: {new_status}")

    document = await get_document_by_lightrag_id(session, lightrag_id)
    if document:
        document.status = new_status
        await session.flush()
        logger.info(
            f"Статус документа lightrag_id: {lightrag_id} в БД обновлен на '{new_status}'."
        )
        return document
    logger.warning(
        f"Попытка обновить статус несуществующего документа в БД: lightrag_id {lightrag_id}"
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
        logger.info(f"Документ lightrag_id: {lightrag_id} физически удален из БД.")
        return True
    logger.warning(
        f"Попытка физически удалить несуществующий документ: lightrag_id {lightrag_id}"
    )
    return False
