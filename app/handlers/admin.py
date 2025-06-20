import asyncio  # Добавлено для фоновых задач
import uuid
from typing import Dict, List, Optional, Union

from aiogram import Bot, F, Router
from aiogram.enums import ChatAction
from aiogram.filters import Filter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import CallbackQuery, Message
from keyboards.admin_kb import (
    ADMIN_CALLBACK_PREFIX,
    DELETE_DOC_START_ACTION,
    DOC_DELETE_CANCEL_ACTION,
    DOC_DELETE_CONFIRM_PREFIX,
    DOC_PAGE_PREFIX,
    DOC_SELECT_PREFIX,
    LIST_DOCS_ACTION,
    UPLOAD_DOC_ACTION,
    get_admin_main_menu_keyboard,
    get_confirm_delete_document_keyboard,
    get_documents_list_keyboard,
)
from langfuse import Langfuse

# Langfuse imports, если langfuse_handler передается
from langfuse.callback import (
    CallbackHandler as LangfuseCallbackHandler,  # Переименовано во избежание конфликта
)
from lightrag import LightRAG
from loguru import logger
from models.models import Document as DB_Document  # Оставляем для типизации, если нужно
from omegaconf import DictConfig  # Для доступа к cfg
from services.db_service import add_document as db_add_document  # Уже асинхронный
from services.db_service import mark_document_as_deleted_by_admin  # Уже асинхронный
from services.db_service import update_document_status  # Уже асинхронный
from services.db_service import (
    get_document_by_lightrag_id,
    get_documents_for_admin_list,
    get_session,
    get_user_by_id,
)
from services.document_service import parse_document_to_markdown  # Уже асинхронный
from services.document_service import save_uploaded_file_temp  # Уже асинхронный
from services.document_service import DocumentParsingError, cleanup_temp_file
from services.rag_service import (  # Уже асинхронный
    RAGServiceError,
    add_document_contents_to_rag,
)
from services.rag_service import (
    delete_document_from_rag as rag_delete_document,  # Уже асинхронный
)
from services.rag_service import (  # Уже асинхронный
    get_multiple_document_statuses_from_rag,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

DOCS_PER_PAGE = 5


class AdminFilter(Filter):
    async def __call__(
        self,
        message_or_call: Union[Message, CallbackQuery],
        session_pool: async_sessionmaker[AsyncSession],
    ) -> bool:
        user_id = message_or_call.from_user.id
        if not session_pool:
            logger.error("Session pool не доступен в AdminFilter!")
            return False
        try:
            async with get_session(session_pool) as session:
                user_db_obj = await get_user_by_id(session, user_id)
                if user_db_obj and user_db_obj.is_admin:
                    return True
                else:
                    logger.debug(
                        f"Пользователь {user_id} не является администратором. Доступ запрещен."
                    )
                    return False
        except Exception as e:
            logger.error(
                f"Ошибка БД в AdminFilter при проверке пользователя {user_id}: {e}",
                exc_info=True,
            )
            return False


class AdminUploadDocument(StatesGroup):
    waiting_for_document = State()


admin_router = Router(name="admin_handlers")
admin_router.message.filter(AdminFilter())
admin_router.callback_query.filter(AdminFilter())


async def _process_document_background(
    bot: Bot,  # Добавлен bot для возможной отправки уведомлений
    chat_id: int,  # Добавлен chat_id для возможной отправки уведомлений
    user_id: int,  # Добавлен user_id для логирования и связи с БД
    temp_file_path: str,
    original_file_name: str,
    lightrag_doc_id: str,
    session_pool: async_sessionmaker[AsyncSession],
    rag_instance: LightRAG,
    bot_texts: dict,
    cfg: DictConfig,  # Добавлена конфигурация Hydra
    langfuse_handler: Optional[
        LangfuseCallbackHandler
    ] = None,  # Опциональный Langfuse handler
):
    """Фоновая задача для обработки документа."""
    langfuse_trace = None
    if cfg.langfuse.enabled:
        try:
            # Инициализируем Langfuse клиента здесь, если он не был передан
            # Или используем глобальный/переданный клиент
            # Для простоты, предположим, что cfg содержит все необходимое для инициализации
            langfuse_client = Langfuse(
                public_key=cfg.langfuse.public_key,
                secret_key=cfg.langfuse.secret_key,
                host=cfg.langfuse.host,
                release=cfg.langfuse.get("release"),
                debug=cfg.langfuse.get("debug", False),
            )
            langfuse_trace = langfuse_client.trace(
                name="document-processing-pipeline",
                user_id=str(user_id),  # Langfuse ожидает user_id как строку
                metadata={
                    "file_name": original_file_name,
                    "lightrag_doc_id": lightrag_doc_id,
                    "trigger": "admin_upload",
                },
            )
            logger.info(
                f"[Langfuse] Трейс создан для обработки документа: {lightrag_doc_id}"
            )
        except Exception as e_fuse:
            logger.error(f"Ошибка инициализации Langfuse в фоновой задаче: {e_fuse}")
            # Продолжаем без Langfuse, если инициализация не удалась

    db_doc_created = False
    processing_status = "unknown_error"  # Статус по умолчанию

    try:
        logger.info(
            f"Фоновая обработка документа '{original_file_name}' (ID: {lightrag_doc_id}) началась."
        )

        # 1. Парсинг документа
        parsing_span = (
            langfuse_trace.span(
                name="parse-document", input={"file_path": temp_file_path}
            )
            if langfuse_trace
            else None
        )
        markdown_content_str = await parse_document_to_markdown(temp_file_path)
        if parsing_span:
            parsing_span.end(
                output={
                    "markdown_length": len(markdown_content_str)
                    if markdown_content_str
                    else 0
                }
            )

        if not markdown_content_str:
            logger.error(
                f"Не удалось извлечь Markdown из документа '{original_file_name}' (ID: {lightrag_doc_id})."
            )
            processing_status = "parsing_failed"
            raise DocumentParsingError("Markdown content is empty.")

        # 2. Добавление записи в БД (статус pending)
        async with get_session(session_pool) as session:
            await db_add_document(
                session=session,
                lightrag_id=lightrag_doc_id,
                file_name=original_file_name,
                uploaded_by_tg_id=user_id,
                status="processing",  # Сразу ставим processing, т.к. парсинг прошел
            )
            db_doc_created = True
        logger.info(
            f"Запись о документе '{original_file_name}' (ID: {lightrag_doc_id}) обновлена/добавлена в БД со статусом 'processing'."
        )

        # 3. Индексация в LightRAG
        indexing_span = (
            langfuse_trace.span(
                name="index-in-lightrag",
                input={
                    "doc_id": lightrag_doc_id,
                    "content_length": len(markdown_content_str),
                },
            )
            if langfuse_trace
            else None
        )
        success_rag_add = await add_document_contents_to_rag(
            rag_instance=rag_instance,
            file_path=[original_file_name],  # LightRAG ожидает список
            documents_content=[markdown_content_str],  # LightRAG ожидает список
            document_ids=[lightrag_doc_id],  # LightRAG ожидает список
        )
        if indexing_span:
            indexing_span.end(output={"rag_add_successful": success_rag_add})

        # 4. Обновление статуса в БД
        final_db_status = "processed" if success_rag_add else "indexing_failed"
        async with get_session(session_pool) as session:
            await update_document_status(session, lightrag_doc_id, final_db_status)

        processing_status = final_db_status
        logger.info(
            f"Документ '{original_file_name}' (ID: {lightrag_doc_id}): финальный статус в БД '{final_db_status}'."
        )

        # Опционально: уведомить администратора об успешной обработке
        # await bot.send_message(chat_id, f"Документ '{original_file_name}' успешно обработан.")

    except DocumentParsingError as e:
        logger.error(
            f"Ошибка парсинга в фоновой задаче для '{original_file_name}': {e}",
            exc_info=True,
        )
        processing_status = "parsing_failed"
        if (
            db_doc_created
        ):  # Если запись была создана, но парсинг упал позже (маловероятно с текущей логикой)
            async with get_session(session_pool) as session:
                await update_document_status(session, lightrag_doc_id, "failed_parsing")
    except RAGServiceError as e:
        logger.error(
            f"Ошибка сервиса RAG в фоновой задаче для '{original_file_name}': {e}",
            exc_info=True,
        )
        processing_status = "rag_error"
        if db_doc_created:
            async with get_session(session_pool) as session:
                await update_document_status(session, lightrag_doc_id, "failed_rag")
    except Exception as e:
        logger.error(
            f"Неожиданная ошибка в фоновой задаче для '{original_file_name}': {e}",
            exc_info=True,
        )
        processing_status = "unknown_background_error"
        if db_doc_created:
            async with get_session(session_pool) as session:
                await update_document_status(session, lightrag_doc_id, "failed_unknown")
    finally:
        cleanup_temp_file(temp_file_path)
        logger.info(
            f"Фоновая обработка документа '{original_file_name}' (ID: {lightrag_doc_id}) завершена со статусом: {processing_status}."
        )
        if langfuse_trace:
            langfuse_trace.update(
                output={"final_processing_status": processing_status},
                level="ERROR"
                if "failed" in processing_status or "error" in processing_status
                else "DEFAULT",
            )
            # Убедимся, что все данные отправлены
            if hasattr(langfuse_trace, "client") and langfuse_trace.client:
                langfuse_trace.client.flush()


@admin_router.callback_query(F.data == f"{ADMIN_CALLBACK_PREFIX}main_menu")
async def handle_admin_main_menu_cb(
    callback: CallbackQuery, state: FSMContext, bot_texts: dict
):
    await state.clear()
    text_key = "admin_panel_title"
    text = bot_texts.get("admin", {}).get(
        text_key, "Панель администратора. Выберите действие:"
    )
    keyboard = get_admin_main_menu_keyboard()
    await callback.message.edit_text(text, reply_markup=keyboard)
    await callback.answer()


@admin_router.callback_query(F.data == UPLOAD_DOC_ACTION)
async def handle_upload_document_action(
    callback: CallbackQuery, state: FSMContext, bot_texts: dict
):
    upload_prompt_key = "admin_doc_upload_prompt"
    prompt_text = bot_texts.get("admin", {}).get(
        upload_prompt_key,
        "Пожалуйста, отправьте файл документа для загрузки.",
    )
    await callback.message.edit_text(prompt_text)
    await state.set_state(AdminUploadDocument.waiting_for_document)
    await callback.answer()


async def send_admin_main_menu(
    target: Union[Message, CallbackQuery],
    bot_texts: dict,
    state: Optional[FSMContext] = None,
):
    if state:
        await state.clear()
    text_key = "admin_panel_title"
    text = bot_texts.get("admin", {}).get(
        text_key, "Панель администратора. Выберите действие:"
    )
    keyboard = get_admin_main_menu_keyboard()
    if isinstance(target, Message):
        await target.answer(text, reply_markup=keyboard)
    elif isinstance(target, CallbackQuery):
        if target.message:
            try:
                await target.message.edit_text(text, reply_markup=keyboard)
            except Exception as e:
                logger.warning(
                    f"Не удалось отредактировать сообщение для админ-меню: {e}."
                )
                await target.message.answer(text, reply_markup=keyboard)
        else:
            await target.bot.send_message(
                chat_id=target.from_user.id, text=text, reply_markup=keyboard
            )
        await target.answer()


@admin_router.message(AdminUploadDocument.waiting_for_document, F.document)
async def handle_document_received(
    message: Message,
    state: FSMContext,
    bot: Bot,
    session_pool: async_sessionmaker[AsyncSession],
    rag_instance: LightRAG,
    bot_texts: dict,
    cfg: DictConfig,  # Добавлено для передачи в фоновую задачу
    langfuse_handler: Optional[LangfuseCallbackHandler] = None,  # Добавлено
):
    if not message.document or not message.from_user:
        await message.reply(
            "Произошла ошибка: документ или информация о пользователе не получены."
        )
        return

    document_file = message.document
    original_file_name = (
        document_file.file_name or f"unknown_document_{uuid.uuid4().hex[:8]}"
    )
    lightrag_doc_id = f"doc_{uuid.uuid4().hex}"

    logger.info(
        f"Администратор {message.from_user.id} загрузил файл: {original_file_name}. Присвоен lightrag_id: {lightrag_doc_id}"
    )

    # Немедленный ответ администратору
    await message.reply(
        f"Файл '{original_file_name}' получен и поставлен в очередь на обработку."
    )
    await state.clear()  # Сбрасываем состояние ожидания документа

    temp_file_path = None
    downloaded_file_stream = None
    try:
        file_info = await bot.get_file(document_file.file_id)
        downloaded_file_stream = await bot.download_file(file_info.file_path)

        if not downloaded_file_stream:
            await message.reply(f"Не удалось скачать файл '{original_file_name}'.")
            return

        downloaded_file_bytes = downloaded_file_stream.read()
        temp_file_path, _ = await save_uploaded_file_temp(
            downloaded_file_bytes, original_file_name
        )

        # Запуск фоновой задачи
        asyncio.create_task(
            _process_document_background(
                bot=bot,
                chat_id=message.chat.id,
                user_id=message.from_user.id,
                temp_file_path=temp_file_path,
                original_file_name=original_file_name,
                lightrag_doc_id=lightrag_doc_id,
                session_pool=session_pool,
                rag_instance=rag_instance,
                bot_texts=bot_texts,
                cfg=cfg,  # Передаем cfg
                langfuse_handler=langfuse_handler,  # Передаем langfuse_handler
            )
        )
        # Не нужно вызывать send_admin_main_menu здесь, т.к. ответ уже дан

    except Exception as e:
        logger.error(
            f"Ошибка на начальном этапе обработки файла '{original_file_name}' (до запуска фоновой задачи): {e}",
            exc_info=True,
        )
        await message.reply(
            f"Произошла ошибка при подготовке файла '{original_file_name}' к обработке."
        )
        if temp_file_path:  # Очистка, если временный файл был создан до ошибки
            cleanup_temp_file(temp_file_path)
    finally:
        if downloaded_file_stream:
            downloaded_file_stream.close()
        # Не очищаем temp_file_path здесь, это делает фоновая задача


@admin_router.message(AdminUploadDocument.waiting_for_document)
async def handle_wrong_content_for_upload(message: Message, bot_texts: dict):
    wrong_content_key = "admin_doc_upload_not_a_document"
    text = bot_texts.get("admin", {}).get(
        wrong_content_key, "Пожалуйста, отправьте файл документа или отмените загрузку."
    )
    await message.reply(text)


async def show_documents_page(
    callback_or_message: Union[CallbackQuery, Message],
    session_pool: async_sessionmaker[AsyncSession],
    rag_instance: LightRAG,  # rag_instance нужен для get_multiple_document_statuses_from_rag
    bot_texts: dict,
    page: int = 1,
    action_context: str = "list",
):
    async with get_session(session_pool) as session:
        db_docs_query_result: List[DB_Document] = await get_documents_for_admin_list(
            session, limit=1000  # Можно сделать лимит настраиваемым
        )

    total_db_docs = len(db_docs_query_result)
    doc_ids_from_db = [doc.lightrag_id for doc in db_docs_query_result]
    rag_statuses: Dict[str, Optional[str]] = {}
    if doc_ids_from_db:
        try:
            rag_statuses = await get_multiple_document_statuses_from_rag(
                rag_instance, doc_ids_from_db
            )
        except Exception as e_rag_status:
            logger.error(
                f"Ошибка при получении статусов из RAG для списка документов: {e_rag_status}"
            )
            # Заполняем статусы ошибкой, чтобы UI это отразил
            rag_statuses = {doc_id: "Ошибка RAG" for doc_id in doc_ids_from_db}

    enriched_documents = []
    for db_doc in db_docs_query_result:
        # Статус из БД - основной источник правды об обработке
        db_status_display = db_doc.status

        # Статус из RAG - дополнительная информация о наличии в вектороном хранилище
        raw_rag_status = rag_statuses.get(db_doc.lightrag_id)
        rag_status_for_display: str

        if (
            db_doc.status == "processed"
        ):  # Если в БД "processed", ожидаем, что в RAG тоже "processed"
            rag_status_for_display = (
                raw_rag_status if raw_rag_status else "Нет в RAG (ожидался)"
            )
            if raw_rag_status and raw_rag_status.lower() != "processed":
                rag_status_for_display = f"RAG: {raw_rag_status} (ожидался processed)"
        elif db_doc.status in ["pending", "processing"]:
            rag_status_for_display = "Обрабатывается"
        elif "failed" in db_doc.status:
            rag_status_for_display = "Ошибка обработки"
        elif db_doc.status == "deleted_by_admin":
            rag_status_for_display = "Удален админом"
        else:  # Неизвестный статус из БД
            rag_status_for_display = raw_rag_status if raw_rag_status else "N/A"

        setattr(
            db_doc,
            "display_status_line",
            f"БД: {db_status_display}, RAG: {rag_status_for_display}",
        )
        enriched_documents.append(db_doc)

    total_pages = (total_db_docs + DOCS_PER_PAGE - 1) // DOCS_PER_PAGE
    if total_pages == 0:
        page = 1
    if page < 1:
        page = 1
    if page > total_pages and total_pages > 0:
        page = total_pages

    offset = (page - 1) * DOCS_PER_PAGE
    documents_on_page = enriched_documents[offset : offset + DOCS_PER_PAGE]

    text: str
    if not documents_on_page and total_db_docs == 0:
        text_key = "admin_docs_list_empty"
        text = bot_texts.get("admin", {}).get(
            text_key, "Список загруженных документов пуст."
        )
        keyboard = get_admin_main_menu_keyboard()
    else:
        text_template_key = (
            "admin_docs_list_title_v2"
            if action_context == "list"
            else "admin_docs_select_for_delete_v2"
        )
        default_text_template = (
            "Документы (Стр. {page}/{total_pages}):"
            if action_context == "list"
            else "Удаление (Стр. {page}/{total_pages}):"
        )

        text_template = bot_texts.get("admin", {}).get(
            text_template_key, default_text_template
        )
        text = text_template.format(
            page=page, total_pages=total_pages if total_pages > 0 else 1
        )

        # Передаем display_status_line в клавиатуру
        keyboard = get_documents_list_keyboard(
            documents_on_page,
            current_page=page,
            total_pages=total_pages if total_pages > 0 else 1,
            context=action_context,  # Передаем контекст для callback_data кнопок
        )

    if isinstance(callback_or_message, CallbackQuery):
        try:
            await callback_or_message.message.edit_text(text, reply_markup=keyboard)
        except Exception as e:
            logger.warning(
                f"Не удалось отредактировать сообщение для списка документов: {e}."
            )
            if callback_or_message.message:
                await callback_or_message.message.answer(text, reply_markup=keyboard)
        await callback_or_message.answer()
    elif isinstance(callback_or_message, Message):
        await callback_or_message.answer(text, reply_markup=keyboard)


@admin_router.callback_query(F.data == LIST_DOCS_ACTION)
async def handle_list_documents_action(
    callback: CallbackQuery,
    session_pool: async_sessionmaker[AsyncSession],
    rag_instance: LightRAG,
    bot_texts: dict,
):
    await show_documents_page(
        callback, session_pool, rag_instance, bot_texts, page=1, action_context="list"
    )


@admin_router.callback_query(F.data == DELETE_DOC_START_ACTION)
async def handle_delete_document_start_action(
    callback: CallbackQuery,
    session_pool: async_sessionmaker[AsyncSession],
    rag_instance: LightRAG,
    bot_texts: dict,
):
    await show_documents_page(
        callback, session_pool, rag_instance, bot_texts, page=1, action_context="delete"
    )


@admin_router.callback_query(F.data.startswith(DOC_PAGE_PREFIX))
async def handle_document_page_action(
    callback: CallbackQuery,
    session_pool: async_sessionmaker[AsyncSession],
    rag_instance: LightRAG,
    bot_texts: dict,
):
    try:
        parts = callback.data.split(DOC_PAGE_PREFIX)[1].split(
            "_"
        )  # page_1_list -> ["1", "list"]
        page = int(parts[0])
        action_ctx = (
            parts[1] if len(parts) > 1 else "list"
        )  # По умолчанию 'list', если контекст не передан

        await show_documents_page(
            callback,
            session_pool,
            rag_instance,
            bot_texts,
            page=page,
            action_context=action_ctx,
        )
    except (ValueError, IndexError) as e:
        logger.error(
            f"Ошибка парсинга страницы/контекста из callback_data '{callback.data}': {e}"
        )
        await callback.answer("Ошибка навигации.", show_alert=True)


@admin_router.callback_query(
    F.data.startswith(DOC_SELECT_PREFIX)
)  # Используется для выбора документа для удаления
async def handle_document_select_for_delete(
    callback: CallbackQuery,
    session_pool: async_sessionmaker[AsyncSession],
    bot_texts: dict,
    state: FSMContext,  # state может быть не нужен здесь, если мы не устанавливаем FSM
):
    try:
        doc_lightrag_id = callback.data.split(DOC_SELECT_PREFIX)[1]
    except IndexError:
        logger.error(
            f"Не удалось извлечь ID документа из callback_data: {callback.data}"
        )
        await callback.answer("Ошибка: неверный ID документа.", show_alert=True)
        return

    async with get_session(session_pool) as session:
        document = await get_document_by_lightrag_id(session, doc_lightrag_id)

    if not document:
        logger.warning(
            f"Попытка удалить несуществующий документ (ID: {doc_lightrag_id})."
        )
        await callback.answer("Документ не найден.", show_alert=True)
        # Вместо вызова handle_admin_main_menu_cb, можно просто закрыть текущее сообщение или обновить список
        await show_documents_page(
            callback,
            session_pool,
            callback.bot.rag_instance,
            bot_texts,
            page=1,
            action_context="delete",
        )  # callback.bot.rag_instance - нужно передать rag_instance
        return

    confirm_text_key = "admin_doc_delete_confirm_prompt"
    confirm_text_template = bot_texts.get("admin", {}).get(
        confirm_text_key,
        "Вы уверены, что хотите удалить документ '{file_name}' (ID: {doc_id})?",
    )
    text = confirm_text_template.format(
        file_name=document.file_name, doc_id=document.lightrag_id
    )
    keyboard = get_confirm_delete_document_keyboard(
        document.lightrag_id, document.file_name
    )

    await callback.message.edit_text(text, reply_markup=keyboard)
    await callback.answer()


@admin_router.callback_query(F.data.startswith(DOC_DELETE_CONFIRM_PREFIX))
async def handle_document_delete_confirm(
    callback: CallbackQuery,
    session_pool: async_sessionmaker[AsyncSession],
    rag_instance: LightRAG,
    bot_texts: dict,
    bot: Bot,  # bot нужен для send_chat_action и edit_text
    # state: FSMContext, # state здесь не используется
):
    try:
        doc_lightrag_id = callback.data.split(DOC_DELETE_CONFIRM_PREFIX)[1]
    except IndexError:
        logger.error(
            f"Не удалось извлечь ID из callback_data для подтверждения удаления: {callback.data}"
        )
        await callback.answer("Ошибка: неверный ID документа.", show_alert=True)
        return

    if callback.message:  # Проверка что callback.message существует
        await callback.message.edit_text("Удаление документа, пожалуйста, подождите...")
        await bot.send_chat_action(
            chat_id=callback.message.chat.id, action=ChatAction.TYPING
        )

    doc_name_for_message = "Неизвестный документ"
    success_db_mark = False
    success_rag_delete = False

    langfuse_trace_delete = None
    # cfg должен быть доступен, если он передается в dispatcher и затем в хендлеры
    # Для этого нужно добавить cfg в аргументы функции или получить из bot.cfg, если он там есть
    # В данном примере предполагаем, что cfg доступен через callback.bot.cfg или аналогично
    current_cfg = getattr(callback.bot, "cfg", None)
    if current_cfg and current_cfg.langfuse.enabled:
        try:
            langfuse_client_delete = Langfuse(
                public_key=current_cfg.langfuse.public_key,
                secret_key=current_cfg.langfuse.secret_key,
                host=current_cfg.langfuse.host,
            )
            langfuse_trace_delete = langfuse_client_delete.trace(
                name="document-delete-pipeline",
                user_id=str(callback.from_user.id),
                metadata={"lightrag_doc_id": doc_lightrag_id},
            )
        except Exception as e_lf_del:
            logger.error(f"Ошибка инициализации Langfuse для удаления: {e_lf_del}")

    try:
        async with get_session(session_pool) as session:
            db_doc = await get_document_by_lightrag_id(session, doc_lightrag_id)
            if db_doc:
                doc_name_for_message = db_doc.file_name
                await mark_document_as_deleted_by_admin(session, doc_lightrag_id)
                success_db_mark = True
                logger.info(
                    f"Документ '{doc_name_for_message}' (ID: {doc_lightrag_id}) помечен как 'deleted_by_admin' в БД."
                )
            else:
                logger.warning(
                    f"Документ с ID '{doc_lightrag_id}' не найден в БД при подтверждении удаления."
                )

        if success_db_mark:  # Удаляем из RAG только если в БД успешно помечен
            rag_delete_span = (
                langfuse_trace_delete.span(
                    name="delete-from-lightrag", input={"doc_id": doc_lightrag_id}
                )
                if langfuse_trace_delete
                else None
            )
            success_rag_delete = await rag_delete_document(
                rag_instance, doc_lightrag_id
            )
            if rag_delete_span:
                rag_delete_span.end(
                    output={"rag_delete_successful": success_rag_delete}
                )

            if success_rag_delete:
                logger.info(f"Документ '{doc_lightrag_id}' успешно удален из LightRAG.")
            else:
                logger.error(
                    f"Ошибка при удалении документа '{doc_lightrag_id}' из LightRAG."
                )

        final_message_text: str
        if success_db_mark and success_rag_delete:
            text_key = "admin_doc_delete_success"
            final_message_text = (
                bot_texts.get("admin", {})
                .get(text_key, "Документ '{file_name}' успешно удален.")
                .format(file_name=doc_name_for_message)
            )
        elif success_db_mark and not success_rag_delete:
            text_key = "admin_doc_delete_db_ok_rag_fail"
            final_message_text = (
                bot_texts.get("admin", {})
                .get(
                    text_key,
                    "Документ '{file_name}' помечен как удаленный, но ошибка при удалении из RAG. Проверьте логи.",
                )
                .format(file_name=doc_name_for_message)
            )
        else:  # Документ не найден в БД или другая ошибка на этапе БД
            text_key = "admin_doc_delete_fail_db"
            final_message_text = (
                bot_texts.get("admin", {})
                .get(text_key, "Не удалось удалить документ '{file_name}' (ошибка БД).")
                .format(file_name=doc_name_for_message)
            )

        if callback.message:  # Проверка что callback.message существует
            await callback.message.edit_text(final_message_text)

        # Обновляем список документов после удаления
        await show_documents_page(
            callback,
            session_pool,
            rag_instance,
            bot_texts,
            page=1,
            action_context="delete",
        )

    except Exception as e:
        logger.error(
            f"Ошибка при подтверждении удаления документа {doc_lightrag_id}: {e}",
            exc_info=True,
        )
        error_key = "error_generic_delete"
        error_text = bot_texts.get("errors", {}).get(
            error_key, "Произошла ошибка при удалении."
        )
        if callback.message:  # Проверка что callback.message существует
            try:
                await callback.message.edit_text(error_text)
            except Exception:  # noqa
                logger.error("Не удалось отправить сообщение об ошибке при удалении.")
    finally:
        if langfuse_trace_delete:
            langfuse_trace_delete.update(
                output={
                    "db_marked_deleted": success_db_mark,
                    "rag_deleted": success_rag_delete,
                }
            )
            if (
                hasattr(langfuse_trace_delete, "client")
                and langfuse_trace_delete.client
            ):
                langfuse_trace_delete.client.flush()
        await callback.answer()


@admin_router.callback_query(F.data == DOC_DELETE_CANCEL_ACTION)
async def handle_document_delete_cancel(
    callback: CallbackQuery,
    session_pool: async_sessionmaker[AsyncSession],
    rag_instance: LightRAG,
    bot_texts: dict,
    # state: FSMContext, # state здесь не используется
):
    await callback.answer("Удаление отменено.")
    await show_documents_page(
        callback, session_pool, rag_instance, bot_texts, page=1, action_context="delete"
    )
