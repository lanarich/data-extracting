from typing import Any, Dict, List

from aiogram import Bot, F, Router, types
from aiogram.enums import ChatAction
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from keyboards.admin_kb import get_admin_main_menu_keyboard
from lightrag import LightRAG
from loguru import logger
from services.db_service import get_or_create_user, get_session
from services.rag_service import RAGServiceError, get_rag_answer
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

user_router = Router(name="user_handlers")

MAX_HISTORY_MESSAGES = 5
DEFAULT_LLM_MODE = "standard"


@user_router.message(CommandStart())
async def handle_start_command(
    message: types.Message,
    state: FSMContext,
    session_pool: async_sessionmaker[AsyncSession],
    bot_texts: dict,
):
    user = message.from_user
    if not user:
        logger.warning(
            "Получено сообщение без информации о пользователе в CommandStart."
        )
        await message.answer(
            "Не удалось получить информацию о пользователе. Попробуйте еще раз."
        )
        return

    logger.info(
        f"Пользователь {user.id} ({user.username or 'N/A'}) вызвал команду /start."
    )

    await state.clear()
    await state.update_data(chat_history=[], llm_mode=DEFAULT_LLM_MODE)
    logger.info(
        f"Для пользователя {user.id} установлен режим LLM по умолчанию: {DEFAULT_LLM_MODE}."
    )

    try:
        async with get_session(session_pool) as session:
            db_user, created = await get_or_create_user(
                session=session,
                user_id=user.id,
                username=user.username,
                first_name=user.first_name,
                last_name=user.last_name,
            )

            if created:
                logger.info(
                    f"Новый пользователь {db_user.user_id} зарегистрирован с is_admin={db_user.is_admin}."
                )
            else:
                logger.info(
                    f"Пользователь {db_user.user_id} уже существует (is_admin={db_user.is_admin}), данные обновлены (если требовалось)."
                )

        if db_user and db_user.is_admin:
            logger.info(
                f"Пользователь {user.id} является администратором (is_admin=True в БД). Показываем админ-меню."
            )
            admin_menu_text_key = "admin_welcome"
            admin_text = bot_texts.get("admin", {}).get(
                admin_menu_text_key,
                "Добро пожаловать, Администратор! Ваша панель управления:",
            )
            keyboard = get_admin_main_menu_keyboard()
            await message.answer(text=admin_text, reply_markup=keyboard)
        else:
            logger.info(
                f"Пользователь {user.id} является обычным пользователем (is_admin=False или не найден в БД)."
            )
            greeting_message_key = "user_greeting"

            start_texts = bot_texts.get("start", {})
            if not isinstance(start_texts, dict):
                logger.warning(
                    f"Секция 'start' в bot_texts не является словарем: {start_texts}"
                )
                start_texts = {}

            if greeting_message_key not in start_texts:
                logger.warning(
                    f"Ключ '{greeting_message_key}' не найден в bot_texts['start']. Используется сообщение по умолчанию."
                )
                greeting_text = f"Здравствуйте, {user.first_name or user.username}!\nЯ ваш помощник по учебным материалам."
            else:
                greeting_template = start_texts.get(greeting_message_key, "Привет!")
                greeting_text = greeting_template.format(
                    user_first_name=user.first_name or "",
                    user_username=user.username or "пользователь",
                )

            user_commands_texts = bot_texts.get("user_commands", {})
            if not isinstance(user_commands_texts, dict):
                logger.warning(
                    f"Секция 'user_commands' в bot_texts не является словарем: {user_commands_texts}"
                )
                user_commands_texts = {}

            mode_display_key = "current_llm_mode_display"
            mode_display_template = user_commands_texts.get(
                mode_display_key, "\nТекущий режим ответов: {mode_name}."
            )

            fsm_data = await state.get_data()
            current_llm_mode_for_display = fsm_data.get("llm_mode", DEFAULT_LLM_MODE)
            mode_name = (
                "Стандартный"
                if current_llm_mode_for_display == "standard"
                else "Размышление"
            )
            greeting_text += mode_display_template.format(mode_name=mode_name)

            await message.answer(text=greeting_text)

    except Exception as e:
        logger.error(
            f"Ошибка при обработке команды /start для пользователя {user.id}: {e}",
            exc_info=True,
        )
        error_message_key = "error_generic"
        error_text_template = bot_texts.get("errors", {}).get(
            error_message_key, "Произошла ошибка. Пожалуйста, попробуйте позже."
        )
        error_text = str(error_text_template)
        await message.answer(error_text)


@user_router.message(Command("toggle_mode"))
async def handle_toggle_llm_mode(
    message: types.Message, state: FSMContext, bot_texts: dict
):
    user = message.from_user
    if not user:
        return

    current_data = await state.get_data()
    current_mode = current_data.get("llm_mode", DEFAULT_LLM_MODE)

    new_mode = "thinking" if current_mode == "standard" else "standard"
    await state.update_data(llm_mode=new_mode)

    mode_switched_key = "llm_mode_switched"
    mode_name_display = "Стандартный" if new_mode == "standard" else "Размышление"

    text_template = bot_texts.get("user_commands", {}).get(
        mode_switched_key, "Режим LLM изменен на: {mode_name}."
    )
    response_text = text_template.format(mode_name=mode_name_display)

    await message.answer(response_text)
    logger.info(f"Пользователь {user.id} переключил режим LLM на: {new_mode}.")


@user_router.message(
    F.text, ~CommandStart(), ~Command("toggle_mode"), ~Command("clear_history")
)
async def handle_text_message(
    message: types.Message,
    state: FSMContext,
    rag_instance: LightRAG,
    bot_texts: dict,
    bot: Bot,
    llm_base_config: Dict[str, Any],
):
    user = message.from_user
    if not user or not message.text:
        logger.warning(
            "Получено пустое текстовое сообщение или нет информации о пользователе."
        )
        return

    query_text = message.text
    logger.info(
        f"Пользователь {user.id} отправил текстовый запрос: '{query_text[:50]}...'"
    )

    await bot.send_chat_action(chat_id=message.chat.id, action=ChatAction.TYPING)

    try:
        data = await state.get_data()
        current_chat_history: List[Dict[str, str]] = data.get("chat_history", [])
        user_llm_mode = data.get("llm_mode", DEFAULT_LLM_MODE)

        logger.debug(
            f"Пользователь {user.id}: режим LLM '{user_llm_mode}' для запроса."
        )
        logger.debug(
            f"Текущая история чата для пользователя {user.id} (перед запросом): {current_chat_history}"
        )
        logger.debug(
            f"Длина текущая история чата для пользователя {user.id} (перед запросом): {len(current_chat_history)}"
        )

        rag_answer = await get_rag_answer(
            rag_instance=rag_instance,
            query_text=query_text,
            chat_history=current_chat_history,
            user_llm_mode=user_llm_mode,
            llm_base_config=llm_base_config,
        )

        if rag_answer:
            await message.answer(rag_answer)
            current_chat_history.append({"role": "user", "content": query_text})
            current_chat_history.append({"role": "assistant", "content": rag_answer})
            if len(current_chat_history) > MAX_HISTORY_MESSAGES * 2:
                current_chat_history = current_chat_history[
                    -(MAX_HISTORY_MESSAGES * 2) :
                ]

            await state.update_data(chat_history=current_chat_history)
            logger.debug(
                f"История чата для пользователя {user.id} (после ответа): {current_chat_history}"
            )
            logger.debug(
                f"Длина история чата для пользователя {user.id} (после ответа): {len(current_chat_history)}"
            )
        else:
            logger.warning(f"RAG вернул пустой ответ на запрос: '{query_text}'")
            no_answer_key = "rag_no_answer"
            no_answer_text = bot_texts.get("rag", {}).get(
                no_answer_key, "К сожалению, я не смог найти ответ на ваш вопрос."
            )
            await message.answer(no_answer_text)

    except RAGServiceError as e:
        logger.error(
            f"Ошибка RAG сервиса для пользователя {user.id} с запросом '{query_text}': {e}",
            exc_info=True,
        )
        error_message_key = "error_rag_processing"
        error_text = bot_texts.get("errors", {}).get(
            error_message_key,
            "Возникла проблема при обработке вашего запроса. Попробуйте еще раз.",
        )
        await message.answer(error_text)
    except Exception as e:
        logger.error(
            f"Неожиданная ошибка при обработке текстового сообщения от {user.id}: {e}",
            exc_info=True,
        )
        error_message_key = "error_generic"
        error_text = bot_texts.get("errors", {}).get(
            error_message_key,
            "Произошла непредвиденная ошибка. Пожалуйста, попробуйте позже.",
        )
        await message.answer(error_text)


@user_router.message(Command("clear_history"))
async def handle_clear_history_command(
    message: types.Message, state: FSMContext, bot_texts: dict
):
    user = message.from_user
    if not user:
        return

    await state.update_data(chat_history=[])
    logger.info(f"История чата для пользователя {user.id} очищена.")

    clear_confirm_key = "chat_history_cleared"
    clear_confirm_text = bot_texts.get("user_commands", {}).get(
        clear_confirm_key, "История вашей переписки очищена."
    )
    await message.answer(clear_confirm_text)
