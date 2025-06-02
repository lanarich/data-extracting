from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
from aiogram.utils.keyboard import InlineKeyboardBuilder

ADMIN_CALLBACK_PREFIX = "admin_"
UPLOAD_DOC_ACTION = f"{ADMIN_CALLBACK_PREFIX}upload_document"
LIST_DOCS_ACTION = f"{ADMIN_CALLBACK_PREFIX}list_documents"
DELETE_DOC_START_ACTION = f"{ADMIN_CALLBACK_PREFIX}delete_doc_start"
DOC_PAGE_PREFIX = f"{ADMIN_CALLBACK_PREFIX}doc_page_"
DOC_SELECT_PREFIX = f"{ADMIN_CALLBACK_PREFIX}doc_select_"
DOC_DELETE_CONFIRM_PREFIX = f"{ADMIN_CALLBACK_PREFIX}doc_delete_confirm_"
DOC_DELETE_CANCEL_ACTION = f"{ADMIN_CALLBACK_PREFIX}doc_delete_cancel"


def get_admin_main_menu_keyboard() -> InlineKeyboardMarkup:
    """
    Возвращает клавиатуру главного меню администратора.
    """
    builder = InlineKeyboardBuilder()
    builder.button(text="📤 Загрузить документ", callback_data=UPLOAD_DOC_ACTION)
    builder.button(text="📄 Список документов", callback_data=LIST_DOCS_ACTION)
    builder.button(text="🗑️ Удалить документ", callback_data=DELETE_DOC_START_ACTION)
    builder.adjust(1)
    return builder.as_markup()


def get_documents_list_keyboard(
    documents_with_rag_status: list,
    current_page: int,
    total_pages: int,
) -> InlineKeyboardMarkup:
    """
    Создает клавиатуру для отображения списка документов с пагинацией.
    Каждый документ - кнопка с callback_data для его выбора (например, для удаления).
    Отображает имя файла и его статус из RAG.
    """
    builder = InlineKeyboardBuilder()

    for doc in documents_with_rag_status:
        doc_id = getattr(doc, "lightrag_id", "unknown_id")
        file_name = getattr(doc, "file_name", "Unknown Document")
        rag_status_display = getattr(doc, "rag_status_display", "N/A")
        display_name = f"{file_name[:40]}... (RAG: {rag_status_display})"

        builder.button(text=display_name, callback_data=f"{DOC_SELECT_PREFIX}{doc_id}")

    pagination_buttons = []
    if current_page > 1:
        pagination_buttons.append(
            InlineKeyboardButton(
                text="⬅️ Назад", callback_data=f"{DOC_PAGE_PREFIX}{current_page - 1}"
            )
        )
    if current_page < total_pages:
        pagination_buttons.append(
            InlineKeyboardButton(
                text="Вперед ➡️", callback_data=f"{DOC_PAGE_PREFIX}{current_page + 1}"
            )
        )

    builder.adjust(1)
    if pagination_buttons:
        builder.row(*pagination_buttons)

    builder.row(
        InlineKeyboardButton(
            text="↩️ В админ. меню", callback_data=f"{ADMIN_CALLBACK_PREFIX}main_menu"
        )
    )
    return builder.as_markup()


def get_confirm_delete_document_keyboard(
    document_id: str, document_name: str
) -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    builder.button(
        text=f"🗑️ Да, удалить '{document_name[:30]}...'",
        callback_data=f"{DOC_DELETE_CONFIRM_PREFIX}{document_id}",
    )
    builder.button(text="❌ Нет, отмена", callback_data=DOC_DELETE_CANCEL_ACTION)
    builder.adjust(1)
    return builder.as_markup()
