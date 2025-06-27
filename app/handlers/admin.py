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
from lightrag import LightRAG
from loguru import logger
from models.models import Document as DB_Document
from services.db_service import add_document as db_add_document
from services.db_service import (
    get_document_by_lightrag_id,
    get_documents_for_admin_list,
    get_session,
    get_user_by_id,
    mark_document_as_deleted_by_admin,
    update_document_status,
)
from services.document_service import (
    DocumentParsingError,
    cleanup_temp_file,
    parse_document_to_markdown,
    save_uploaded_file_temp,
)
from services.rag_service import RAGServiceError, add_document_contents_to_rag
from services.rag_service import delete_document_from_rag as rag_delete_document
from services.rag_service import get_multiple_document_statuses_from_rag
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

DOCS_PER_PAGE = 5


class AdminFilter(Filter):
    """
    Фильтр для проверки, является ли пользователь администратором,
    проверяя поле is_admin в базе данных.
    """

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
                        f"Пользователь {user_id} не является администратором (is_admin=False или не найден в БД). Доступ запрещен."
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
        "Пожалуйста, отправьте файл документа для загрузки, формат .pdf",
    )

    await callback.message.edit_text(prompt_text)
    await state.set_state(AdminUploadDocument.waiting_for_document)
    await callback.answer()


async def send_admin_main_menu(
    target: Union[Message, CallbackQuery],
    bot_texts: dict,
    state: Optional[FSMContext] = None,
):
    """Отправляет или редактирует сообщение, показывая главное меню администратора."""
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
                    f"Не удалось отредактировать сообщение для админ-меню: {e}. Отправка нового."
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
):
    if not message.document:
        await message.reply("Произошла ошибка: документ не получен.")
        return

    document_file = message.document
    original_file_name = (
        document_file.file_name or f"unknown_document_{uuid.uuid4().hex[:8]}"
    )
    lightrag_doc_id = f"doc_{uuid.uuid4().hex}"

    logger.info(
        f"Администратор {message.from_user.id} загрузил файл: {original_file_name} (MIME: {document_file.mime_type}). Присвоен lightrag_id: {lightrag_doc_id}"
    )

    await message.reply(f"Файл '{original_file_name}' получен. Начинаю обработку...")
    await bot.send_chat_action(chat_id=message.chat.id, action=ChatAction.TYPING)

    temp_file_path = None
    downloaded_file_stream = None
    db_doc_created = False
    try:
        file_info = await bot.get_file(document_file.file_id)
        downloaded_file_stream = await bot.download_file(file_info.file_path)

        if not downloaded_file_stream:
            await message.reply("Не удалось скачать файл.")
            await state.clear()
            return

        downloaded_file_bytes = downloaded_file_stream.read()

        temp_file_path, _ = await save_uploaded_file_temp(
            downloaded_file_bytes, original_file_name
        )

        logger.info(f"Конвертация документа в Markdown: {temp_file_path}")
        markdown_content_str = await parse_document_to_markdown(temp_file_path)

        if not markdown_content_str:
            await message.reply(
                f"Не удалось извлечь Markdown из документа '{original_file_name}'."
            )
            await state.clear()
            return

        logger.info(
            f"Общая длина извлеченного Markdown: {len(markdown_content_str)} символов."
        )

        async with get_session(session_pool) as session:
            await db_add_document(
                session=session,
                lightrag_id=lightrag_doc_id,
                file_name=original_file_name,
                uploaded_by_tg_id=message.from_user.id,
                status="pending",
            )
            db_doc_created = True
            logger.info(
                f"Запись о документе '{original_file_name}' (ID: {lightrag_doc_id}, status: pending) добавлена в БД."
            )

        logger.info(
            f"Добавление Markdown содержимого документа '{lightrag_doc_id}' в LightRAG..."
        )
        success_rag_add = await add_document_contents_to_rag(
            rag_instance=rag_instance,
            file_path=[original_file_name],
            documents_content=[markdown_content_str],
            document_ids=[lightrag_doc_id],
        )

        final_db_status = "processed" if success_rag_add else "failed"
        async with get_session(session_pool) as session:
            await update_document_status(session, lightrag_doc_id, final_db_status)

        if success_rag_add:
            success_message_key = "admin_doc_upload_success"
            success_text = bot_texts.get("admin", {}).get(
                success_message_key,
                "Документ '{file_name}' успешно обработан и добавлен в базу знаний.",
            )
            await message.reply(success_text.format(file_name=original_file_name))

        else:
            error_rag_key = "admin_doc_upload_rag_fail"
            error_rag_text = bot_texts.get("admin", {}).get(
                error_rag_key,
                "Документ '{file_name}' был сохранен, но произошла ошибка при его добавлении/обработке в RAG. Статус в БД: '{status}'. Проверьте логи.",
            )
            await message.reply(
                error_rag_text.format(
                    file_name=original_file_name, status=final_db_status
                )
            )
            logger.error(
                f"Ошибка добавления/обработки документа '{lightrag_doc_id}' в LightRAG."
            )

    except DocumentParsingError as e:
        logger.error(
            f"Ошибка парсинга файла '{original_file_name}': {e}", exc_info=True
        )
        await message.reply(
            f"Не удалось обработать файл '{original_file_name}'. Ошибка парсинга: {e}"
        )
        if db_doc_created:
            async with get_session(session_pool) as session:
                await update_document_status(session, lightrag_doc_id, "failed")
    except RAGServiceError as e:
        logger.error(
            f"Ошибка сервиса RAG при обработке файла '{original_file_name}': {e}",
            exc_info=True,
        )
        await message.reply(
            f"Произошла ошибка при добавлении документа '{original_file_name}' в RAG: {e}"
        )
        if db_doc_created:
            async with get_session(session_pool) as session:
                await update_document_status(session, lightrag_doc_id, "failed")
    except ValueError as e:
        logger.error(
            f"Ошибка значения (например, неверный статус) при обработке файла '{original_file_name}': {e}",
            exc_info=True,
        )
        await message.reply(
            f"Произошла ошибка значения при обработке файла '{original_file_name}'."
        )
        if db_doc_created:
            async with get_session(session_pool) as session:
                await update_document_status(session, lightrag_doc_id, "failed")
    except Exception as e:
        logger.error(
            f"Неожиданная ошибка при обработке файла '{original_file_name}': {e}",
            exc_info=True,
        )
        await message.reply(
            f"Произошла непредвиденная ошибка при обработке файла '{original_file_name}'."
        )
        if db_doc_created:
            try:
                async with get_session(session_pool) as session:
                    await update_document_status(session, lightrag_doc_id, "failed")
            except Exception as db_err:
                logger.error(
                    f"Дополнительная ошибка при попытке обновить статус документа {lightrag_doc_id} на ошибку: {db_err}"
                )
    finally:
        if downloaded_file_stream:
            downloaded_file_stream.close()
        if temp_file_path:
            cleanup_temp_file(temp_file_path)
        await state.clear()
        await send_admin_main_menu(message, bot_texts, state)


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
    rag_instance: LightRAG,
    bot_texts: dict,
    page: int = 1,
    action_context: str = "list",
):
    async with get_session(session_pool) as session:
        db_docs_query_result: List[DB_Document] = await get_documents_for_admin_list(
            session, limit=1000
        )

    total_db_docs = len(db_docs_query_result)

    doc_ids_from_db = [doc.lightrag_id for doc in db_docs_query_result]
    rag_statuses: Dict[str, Optional[str]] = {}
    if doc_ids_from_db:
        rag_statuses = await get_multiple_document_statuses_from_rag(
            rag_instance, doc_ids_from_db
        )

    enriched_documents = []
    for db_doc in db_docs_query_result:
        db_status = db_doc.status
        raw_rag_status = rag_statuses.get(db_doc.lightrag_id)
        display_rag_status: str

        if raw_rag_status is None:
            display_rag_status = "Нет в RAG"
        elif raw_rag_status == "ошибка_запроса_статуса":
            display_rag_status = "Ошибка RAG"
        elif raw_rag_status == "N/A":
            display_rag_status = "Метод RAG N/A"
        else:
            display_rag_status = raw_rag_status

        setattr(db_doc, "db_status_display", db_status)
        setattr(db_doc, "rag_status_display", display_rag_status)
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
        if action_context == "delete":
            text_template_key = "admin_docs_select_for_delete"
            default_text_template = (
                "Выберите документ для удаления (Страница {page}/{total_pages}):"
            )
        else:
            text_template_key = "admin_docs_list_title"
            default_text_template = (
                "Список загруженных документов (Страница {page}/{total_pages}):"
            )

        text_template = bot_texts.get("admin", {}).get(
            text_template_key, default_text_template
        )
        text = text_template.format(
            page=page, total_pages=total_pages if total_pages > 0 else 1
        )

        keyboard = get_documents_list_keyboard(documents_on_page, page, total_pages)

    if isinstance(callback_or_message, CallbackQuery):
        try:
            await callback_or_message.message.edit_text(text, reply_markup=keyboard)
        except Exception as e:
            logger.warning(
                f"Не удалось отредактировать сообщение для списка документов: {e}. Попытка отправить новое."
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
        page = int(callback.data.split(DOC_PAGE_PREFIX)[1])
        action_ctx = (
            "delete"
            if callback.message
            and callback.message.text
            and "удаления" in callback.message.text.lower()
            else "list"
        )
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
            f"Ошибка парсинга страницы из callback_data '{callback.data}': {e}"
        )
        await callback.answer("Ошибка навигации.", show_alert=True)


@admin_router.callback_query(F.data.startswith(DOC_SELECT_PREFIX))
async def handle_document_select_for_delete(
    callback: CallbackQuery,
    session_pool: async_sessionmaker[AsyncSession],
    bot_texts: dict,
    state: FSMContext,
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
            f"Попытка удалить несуществующий документ (ID: {doc_lightrag_id}) из callback."
        )
        await callback.answer("Документ не найден.", show_alert=True)
        await handle_admin_main_menu_cb(callback, state, bot_texts)
        return

    confirm_text_key = "admin_doc_delete_confirm_prompt"
    confirm_text_template = bot_texts.get("admin", {}).get(
        confirm_text_key,
        "Вы уверены, что хотите удалить документ '{file_name}' (ID: {doc_id})?\nЭто действие нельзя будет отменить.",
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
    bot: Bot,
    state: FSMContext,
):
    try:
        doc_lightrag_id = callback.data.split(DOC_DELETE_CONFIRM_PREFIX)[1]
    except IndexError:
        logger.error(
            f"Не удалось извлечь ID документа из callback_data для подтверждения удаления: {callback.data}"
        )
        await callback.answer("Ошибка: неверный ID документа.", show_alert=True)
        return

    await callback.message.edit_text("Удаление документа, пожалуйста, подождите...")
    await bot.send_chat_action(
        chat_id=callback.message.chat.id, action=ChatAction.TYPING
    )

    doc_name_for_message = "Неизвестный документ"
    success = False
    try:
        async with get_session(session_pool) as session:
            db_doc = await get_document_by_lightrag_id(session, doc_lightrag_id)
            if db_doc:
                doc_name_for_message = db_doc.file_name
                await mark_document_as_deleted_by_admin(session, doc_lightrag_id)
                logger.info(
                    f"Документ '{doc_name_for_message}' (ID: {doc_lightrag_id}) помечен как 'deleted_by_admin' в БД."
                )

                logger.info(f"Удаление документа '{doc_lightrag_id}' из LightRAG...")
                rag_delete_success = await rag_delete_document(
                    rag_instance, doc_lightrag_id
                )
                if rag_delete_success:
                    logger.info(
                        f"Документ '{doc_lightrag_id}' успешно удален из LightRAG."
                    )
                    success = True
                else:
                    logger.error(
                        f"Ошибка при удалении документа '{doc_lightrag_id}' из LightRAG. Статус в БД 'deleted_by_admin'."
                    )
            else:
                logger.warning(
                    f"Документ с ID '{doc_lightrag_id}' не найден в БД при подтверждении удаления."
                )

        if success:
            text_key = "admin_doc_delete_success"
            text = (
                bot_texts.get("admin", {})
                .get(text_key, "Документ '{file_name}' успешно удален.")
                .format(file_name=doc_name_for_message)
            )
        else:
            text_key = "admin_doc_delete_fail"
            text = (
                bot_texts.get("admin", {})
                .get(
                    text_key,
                    "Не удалось удалить документ '{file_name}'. Проверьте логи.",
                )
                .format(file_name=doc_name_for_message)
            )

        await callback.message.edit_text(text)
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
        error_key = "error_generic"
        error_text = bot_texts.get("errors", {}).get(
            error_key, "Произошла ошибка при удалении."
        )
        try:
            await callback.message.edit_text(error_text)
        except Exception:
            logger.error("Не удалось отправить сообщение об ошибке при удалении.")
    finally:
        await callback.answer()


@admin_router.callback_query(F.data == DOC_DELETE_CANCEL_ACTION)
async def handle_document_delete_cancel(
    callback: CallbackQuery,
    session_pool: async_sessionmaker[AsyncSession],
    rag_instance: LightRAG,
    bot_texts: dict,
    state: FSMContext,
):
    await callback.answer("Удаление отменено.")
    await show_documents_page(
        callback, session_pool, rag_instance, bot_texts, page=1, action_context="delete"
    )
