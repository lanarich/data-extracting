from typing import Optional

from aiogram import Bot, F, Router, types
from aiogram.enums import ChatAction
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from langfuse.callback import CallbackHandler as LangfuseCallbackHandler
from langgraph.graph import StatefulGraph
from loguru import logger
from omegaconf import DictConfig
from services.db_service import (  # Используем get_or_create_student
    get_or_create_student,
    get_session,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agents.state import TutorGraphState

user_router = Router(name="user_handlers")


async def handle_graph_output(
    output_state: TutorGraphState, message: types.Message, bot_texts: dict
):
    """Обрабатывает состояние, полученное от графа, и отправляет ответ пользователю."""
    if output_state.get("error_message"):
        logger.error(
            f"Ошибка от графа для пользователя {message.from_user.id}: {output_state['error_message']}"
        )
        error_text = bot_texts.get("errors", {}).get(
            "error_graph_processing", "Произошла ошибка при обработке вашего запроса."
        )
        await message.answer(error_text)
        return

    response_parts = []
    # Сначала информация о теме, если она обновилась и нет вопроса/фидбека
    # Это может быть полезно после curriculum_agent
    if (
        output_state.get("current_topic_name")
        and not output_state.get("assessment_question")
        and not output_state.get("assessment_feedback")
        and not output_state.get("recommendations")
    ):
        response_parts.append(
            f"Текущая тема: \"{output_state['current_topic_name']}\"."
        )
        if output_state.get("current_learning_objectives"):
            response_parts.append(
                f"Цели обучения: {output_state['current_learning_objectives']}"
            )
        # Можно добавить предложение начать, если это первый вход в тему
        # response_parts.append("Готовы начать или есть вопросы по этой теме?")

    if output_state.get("assessment_feedback"):
        response_parts.append(
            f"**Оценка вашего ответа:**\n{output_state['assessment_feedback']}"
        )

    if output_state.get("assessment_question"):
        # Если есть и фидбек, и новый вопрос, они будут вместе.
        # Если только вопрос, то только он.
        response_parts.append(
            f"**Вопрос для вас:**\n{output_state['assessment_question']}"
        )

    if output_state.get("recommendations"):
        recommendations_str = "\n".join(
            [f"- {rec}" for rec in output_state["recommendations"]]
        )
        response_parts.append(f"**Рекомендации:**\n{recommendations_str}")

    if not response_parts:
        logger.warning(
            f"Граф вернул состояние без явного текстового ответа для пользователя {message.from_user.id}"
        )
        # Можно отправить дефолтное сообщение, если это не ожидание ввода от пользователя (например, после /start)
        # if not output_state.get("assessment_question"): # Если не ждем ответа на вопрос
        #     response_parts.append(bot_texts.get("user_messages", {}).get("graph_no_specific_output", "Продолжим?"))

    if response_parts:
        await message.answer(
            "\n\n".join(response_parts), parse_mode="Markdown"
        )  # Добавлен parse_mode для Markdown
    elif not output_state.get("error_message"):
        logger.info(
            f"Граф для пользователя {message.from_user.id} завершил обработку без текстового вывода для пользователя на данном шаге."
        )


@user_router.message(CommandStart())
async def handle_start_command(
    message: types.Message,
    state: FSMContext,
    session_pool: async_sessionmaker[AsyncSession],
    bot: Bot,
    bot_texts: dict,
    tutor_graph: StatefulGraph,
    cfg: DictConfig,
    langfuse_handler: Optional[
        LangfuseCallbackHandler
    ] = None,  # Получаем из Dispatcher
):
    user = message.from_user
    if not user:
        await message.answer("Не удалось получить информацию о пользователе.")
        return

    logger.info(
        f"Пользователь {user.id} ({user.username or 'N/A'}) вызвал команду /start."
    )
    await state.clear()

    async with get_session(session_pool) as db_session:
        db_student_model, created = await get_or_create_student(
            session=db_session,
            user_id=user.id,
            username=user.username,
            first_name=user.first_name,
            last_name=user.last_name,
        )
        if created:
            logger.info(f"Новый студент {db_student_model.user_id} зарегистрирован.")
        else:
            logger.info(f"Студент {db_student_model.user_id} уже существует.")

    initial_graph_input = TutorGraphState(
        student_id=user.id,
        student_profile=None,
        input_query=None,
        chat_history=[],
        current_topic_id=None,
        current_topic_name=None,
        current_learning_objectives=None,
        retrieved_context=None,
        assessment_question=None,
        student_answer=None,
        assessment_feedback=None,
        recommendations=None,
        error_message=None,
    )

    try:
        await message.answer(
            bot_texts.get("user_messages", {}).get(
                "start_graph_processing", "Инициализация вашего учебного плана..."
            )
        )
        await bot.send_chat_action(chat_id=message.chat.id, action=ChatAction.TYPING)

        config_for_graph_call = {"configurable": {"thread_id": str(user.id)}}
        if cfg.langfuse.enabled and langfuse_handler:
            # Langfuse трейсинг для вызовов LangChain компонентов внутри графа
            config_for_graph_call["callbacks"] = [langfuse_handler]
            logger.info(
                f"Langfuse callback handler добавлен для вызова графа пользователя {user.id}"
            )

        final_output_state = None
        # Используем .invoke() для получения только финального состояния после всех шагов,
        # если граф спроектирован так, чтобы выдать все необходимое за один "проход" до первого END (вопроса студенту)
        # Или astream, если хотим обрабатывать промежуточные ответы/действия

        # output = await tutor_graph.ainvoke(initial_graph_input, config=config_for_graph_call)
        # final_output_state = output # ainvoke возвращает финальное состояние

        # Используем astream, если граф может остановиться на полпути (например, задать вопрос)
        # и мы хотим получить это состояние. stream_mode="values" дает полное состояние на каждом шаге.
        async for event_state in tutor_graph.astream(
            initial_graph_input, config=config_for_graph_call, stream_mode="values"
        ):
            final_output_state = event_state

        if final_output_state:
            await handle_graph_output(final_output_state, message, bot_texts)
        else:
            logger.error(
                f"Граф не вернул финального состояния для пользователя {user.id} при /start."
            )
            await message.answer(
                bot_texts.get("errors", {}).get(
                    "error_graph_start_failed", "Не удалось начать сессию."
                )
            )

    except Exception as e:
        logger.error(
            f"Ошибка при первом вызове графа для пользователя {user.id}: {e}",
            exc_info=True,
        )
        await message.answer(
            bot_texts.get("errors", {}).get("error_generic", "Произошла ошибка.")
        )


@user_router.message(
    F.text, ~CommandStart(), ~Command("toggle_mode"), ~Command("clear_history")
)
async def handle_text_message(
    message: types.Message,
    state: FSMContext,
    bot: Bot,
    tutor_graph: StatefulGraph,
    bot_texts: dict,
    cfg: DictConfig,
    langfuse_handler: Optional[LangfuseCallbackHandler] = None,
):
    user = message.from_user
    if not user or not message.text:
        return

    query_text = message.text
    logger.info(f"Пользователь {user.id} отправил: '{query_text[:50]}...'")
    await bot.send_chat_action(chat_id=message.chat.id, action=ChatAction.TYPING)

    # Входные данные для графа: текущий ответ/запрос пользователя.
    # Граф сам подтянет предыдущее состояние по thread_id благодаря MemorySaver.
    graph_input_update = {
        # "input_query": query_text, # Если это новый независимый запрос
        "student_answer": query_text  # Если это ответ на вопрос графа
    }
    # Можно добавить логику, чтобы различать новый запрос и ответ на вопрос,
    # например, проверяя, есть ли в состоянии графа `assessment_question`.
    # Пока для простоты считаем, что любой текст - это `student_answer`.
    # `input_query` можно устанавливать, если это первый запрос в новой теме или после /start.

    try:
        config_for_graph_call = {"configurable": {"thread_id": str(user.id)}}
        if cfg.langfuse.enabled and langfuse_handler:
            config_for_graph_call["callbacks"] = [langfuse_handler]
            logger.info(
                f"Langfuse callback handler добавлен для вызова графа пользователя {user.id} (текстовое сообщение)"
            )

        final_output_state = None
        logger.debug(f"Вызов графа для пользователя {user.id} с текстом: {query_text}")

        # `ainvoke` передает `graph_input_update` как вход для START узла,
        # но если граф уже был запущен для этого thread_id, он продолжит с сохраненного состояния,
        # и `graph_input_update` будет входными данными для следующего узла, который ожидает ввод.
        # В нашем случае, если граф остановился на END после assessment_question,
        # то `student_answer` из `graph_input_update` должен быть подхвачен.
        # Это зависит от того, как определен вход у узлов после точки ожидания.
        # Обычно, `ainvoke` с новым input перезапускает граф с этим input, если нет активного состояния.
        # Если есть активное состояние (т.е. граф ждет), то новый input должен быть обработан как продолжение.
        # Для диалоговых графов, которые ждут ввода, `astream` или явное управление состоянием может быть надежнее.
        # Однако, `StatefulGraph` с чекпоинтером должен корректно обрабатывать это:
        # он загружает состояние и передает новый ввод.

        # output = await tutor_graph.ainvoke(graph_input_update, config=config_for_graph_call)
        # final_output_state = output

        async for event_state in tutor_graph.astream(
            graph_input_update, config=config_for_graph_call, stream_mode="values"
        ):
            final_output_state = event_state

        if final_output_state:
            await handle_graph_output(final_output_state, message, bot_texts)
        else:
            logger.error(
                f"Граф не вернул финального состояния для {user.id} при обработке сообщения."
            )
            await message.answer(
                bot_texts.get("errors", {}).get(
                    "error_graph_processing", "Не удалось обработать ваш запрос."
                )
            )

    except Exception as e:
        logger.error(
            f"Ошибка при вызове графа для сообщения от {user.id}: {e}", exc_info=True
        )
        await message.answer(
            bot_texts.get("errors", {}).get("error_generic", "Произошла ошибка.")
        )


@user_router.message(Command("clear_history"))
async def handle_clear_history_command(
    message: types.Message,
    state: FSMContext,
    tutor_graph: StatefulGraph,  # Нужен для доступа к checkpointer, если хотим чистить
    bot_texts: dict,
    cfg: DictConfig,
):
    user = message.from_user
    if not user:
        return

    logger.info(f"Пользователь {user.id} вызвал /clear_history.")
    await state.clear()  # Очищаем FSM состояние, если оно для чего-то используется

    # Очистка состояния LangGraph для данного пользователя (thread_id)
    # Для MemorySaver это означает удаление записи из его внутреннего словаря.
    # Доступ к checkpointer'у графа: tutor_graph.checkpointer
    if tutor_graph.checkpointer and hasattr(
        tutor_graph.checkpointer, "storage"
    ):  # Проверяем, что это MemorySaver
        thread_id = str(user.id)
        if thread_id in tutor_graph.checkpointer.storage:
            del tutor_graph.checkpointer.storage[thread_id]
            logger.info(
                f"Состояние графа для thread_id='{thread_id}' очищено из MemorySaver."
            )
            clear_confirm_key = "chat_history_cleared_graph_ok"
            default_text = "История вашего диалога с тьютором очищена. Следующее сообщение начнет новую сессию."
        else:
            logger.info(
                f"Состояние графа для thread_id='{thread_id}' не найдено в MemorySaver для очистки."
            )
            clear_confirm_key = "chat_history_cleared_graph_not_found"
            default_text = "Активная сессия с тьютором не найдена для очистки. Следующее сообщение начнет новую."
    else:
        logger.warning(
            "Не удалось очистить состояние графа: checkpointer не MemorySaver или отсутствует."
        )
        clear_confirm_key = "chat_history_cleared_graph_failed"
        default_text = "Не удалось полностью очистить историю сессии с тьютором. Попробуйте команду /start."

    clear_confirm_text = bot_texts.get("user_commands", {}).get(
        clear_confirm_key, default_text
    )
    await message.answer(clear_confirm_text)


# Команда /toggle_mode - оставляем как есть, она влияет на старый LightRAG режим, если он где-то используется
@user_router.message(Command("toggle_mode"))
async def handle_toggle_llm_mode(
    message: types.Message, state: FSMContext, bot_texts: dict
):
    # ... (код как был, с комментариями о его применимости) ...
    user = message.from_user
    if not user:
        return

    current_data = await state.get_data()
    DEFAULT_LLM_MODE_LIGHTRAG = "standard"
    current_mode_lightrag = current_data.get(
        "llm_mode_lightrag", DEFAULT_LLM_MODE_LIGHTRAG
    )

    new_mode_lightrag = (
        "thinking" if current_mode_lightrag == "standard" else "standard"
    )
    await state.update_data(llm_mode_lightrag=new_mode_lightrag)

    mode_switched_key = "llm_mode_switched"
    mode_name_display = (
        "Стандартный (LightRAG)"
        if new_mode_lightrag == "standard"
        else "Размышление (LightRAG)"
    )

    text_template = bot_texts.get("user_commands", {}).get(
        mode_switched_key, "Режим LLM (для LightRAG) изменен на: {mode_name}."
    )
    response_text = text_template.format(mode_name=mode_name_display)

    await message.answer(response_text)
    logger.info(
        f"Пользователь {user.id} переключил режим LLM для LightRAG на: {new_mode_lightrag}."
    )
