import asyncio
import os
import sys
from typing import Optional

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import BotCommand, BotCommandScopeAllPrivateChats
from handlers.admin import admin_router
from handlers.user import user_router
from hydra import compose, initialize
from langchain_openai import ChatOpenAI
from langfuse import Langfuse
from langfuse.callback import CallbackHandler as LangfuseCallbackHandler
from langgraph.graph import StatefulGraph
from lightrag import LightRAG
from loguru import logger
from omegaconf import DictConfig
from services.db_service import (
    create_db_engine_and_session_pool,
    create_tables,
    get_db_url,
)
from services.rag_service import initialize_lightrag_instance
from utils.config_loader import load_texts_from_config

from app.agents.graph import create_tutor_graph

langfuse_callback_handler: Optional[LangfuseCallbackHandler] = None
langfuse_client: Optional[Langfuse] = None
cfg: Optional[DictConfig] = None

try:
    with initialize(
        version_base=None, config_path="utils/conf", job_name="medical_university_bot"
    ):
        cfg = compose(config_name="config")
    logger.info("Конфигурация Hydra успешно загружена.")

    if not hasattr(cfg, "server") or not cfg.server.API_KEY:
        raise ValueError("Отсутствует API_KEY в конфигурации (server.API_KEY).")
    if not hasattr(cfg, "llm") or not cfg.llm.api_base or not cfg.llm.model_name:
        raise ValueError(
            "Отсутствует или неполная конфигурация LLM (llm.api_base, llm.model_name)."
        )
    if (
        not hasattr(cfg, "embedding")
        or not cfg.embedding.api_base
        or not cfg.embedding.model_name
    ):
        raise ValueError(
            "Отсутствует или неполная конфигурация Embedding (embedding.api_base, embedding.model_name)."
        )

    if (
        not hasattr(cfg, "langfuse")
        or not hasattr(cfg.langfuse, "public_key")
        or not hasattr(cfg.langfuse, "secret_key")
        or not hasattr(cfg.langfuse, "host")
    ):
        logger.warning(
            "Конфигурация Langfuse неполная или отсутствует в config.yaml. Langfuse будет отключен."
        )
        if hasattr(cfg, "langfuse"):
            cfg.langfuse.enabled = False
        else:
            from omegaconf import OmegaConf

            cfg.langfuse = OmegaConf.create(
                {"enabled": False, "public_key": None, "secret_key": None, "host": None}
            )
    elif (
        not cfg.langfuse.public_key
        or not cfg.langfuse.secret_key
        or not cfg.langfuse.host
    ):
        logger.warning(
            "Ключи или хост Langfuse не заданы в config.yaml. Langfuse будет отключен."
        )
        cfg.langfuse.enabled = False


except Exception as e:
    print(
        f"КРИТИЧЕСКАЯ ОШИБКА: Ошибка загрузки конфигурации через Hydra: {e}",
        file=sys.stderr,
    )
    sys.exit("Загрузка конфигурации не удалась. Проверьте логи и настройки Hydra.")


os.environ["QDRANT_URL"] = cfg.qdrant.get("url", "http://127.0.0.1:6333")

logger.remove()
logger.add(
    sys.stdout,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
    level=cfg.logging.console_level
    if hasattr(cfg.logging, "console_level")
    else "INFO",
    colorize=True,
)

if (
    hasattr(cfg, "logging")
    and hasattr(cfg.logging, "file_path")
    and cfg.logging.file_path
):
    log_file_path = cfg.logging.file_path
    logger.add(
        log_file_path,
        rotation=cfg.logging.rotation if hasattr(cfg.logging, "rotation") else "100 MB",
        retention=cfg.logging.retention
        if hasattr(cfg.logging, "retention")
        else "10 days",
        compression=cfg.logging.compression
        if hasattr(cfg.logging, "compression")
        else "zip",
        level=cfg.logging.file_level if hasattr(cfg.logging, "file_level") else "DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        encoding="utf-8",
        enqueue=True,
        backtrace=True,
        diagnose=True,
    )
    logger.info(f"Логирование в файл настроено: {log_file_path}")
else:
    logger.info(
        "Логирование в файл не настроено (не указан logging.file_path в конфигурации)."
    )

logger.info("Логгер сконфигурирован.")


if cfg.langfuse.enabled:
    try:
        langfuse_client = Langfuse(
            public_key=cfg.langfuse.public_key,
            secret_key=cfg.langfuse.secret_key,
            host=cfg.langfuse.host,
            release=cfg.langfuse.get("release"),
            debug=cfg.langfuse.get("debug", False),
        )
        langfuse_callback_handler = LangfuseCallbackHandler(
            public_key=cfg.langfuse.public_key,
            secret_key=cfg.langfuse.secret_key,
            host=cfg.langfuse.host,
            release=cfg.langfuse.get("release"),
        )
        logger.info(
            f"Langfuse успешно инициализирован. Release: {cfg.langfuse.get('release')}"
        )
    except Exception as e:
        logger.error(
            f"Ошибка инициализации Langfuse: {e}. Langfuse будет отключен.",
            exc_info=True,
        )
        cfg.langfuse.enabled = False
        langfuse_callback_handler = None
        langfuse_client = None
else:
    logger.info("Langfuse отключен согласно конфигурации.")
    langfuse_callback_handler = None
    langfuse_client = None


async def set_bot_commands(bot: Bot):
    user_commands = [
        BotCommand(command="start", description="▶️ Запустить / Перезапустить бота"),
        BotCommand(
            command="toggle_mode",
            description="⚙️ Сменить режим LLM (Стандарт/Размышление)",
        ),
        BotCommand(command="clear_history", description="🗑️ Очистить историю диалога"),
    ]
    try:
        await bot.set_my_commands(
            commands=user_commands, scope=BotCommandScopeAllPrivateChats()
        )
        logger.info(
            f"Пользовательские команды {', '.join([c.command for c in user_commands])} успешно установлены."
        )
    except Exception as e:
        logger.error(
            f"Ошибка при установке пользовательских команд бота: {e}", exc_info=True
        )


async def main():
    logger.info("Инициализация бота...")

    bot_texts = {}
    try:
        bot_texts = load_texts_from_config(cfg)
        logger.info("Тексты для бота успешно загружены.")
    except Exception as e:
        logger.critical(
            f"Не удалось загрузить тексты для бота: {e}. Бот не может быть запущен.",
            exc_info=True,
        )
        return

    engine, session_pool = None, None
    try:
        db_url_str = get_db_url(cfg.server)
        engine, session_pool = await create_db_engine_and_session_pool(
            db_url_str,
            echo=cfg.db.echo
            if hasattr(cfg, "db") and hasattr(cfg.db, "echo")
            else False,
        )
        await create_tables(
            engine
        )  # Убедитесь, что create_tables создает новые таблицы
        logger.info(
            f"Соединение с БД ({cfg.server.DB_NAME}@{cfg.server.DB_HOST}) установлено и таблицы созданы/проверены."
        )
    except Exception as e:  # Более общая обработка ошибок
        logger.critical(f"Не удалось инициализировать БД: {e}", exc_info=True)
        return

    langchain_llm = ChatOpenAI(
        model=cfg.llm.model_name,
        openai_api_base=cfg.llm.api_base,
        openai_api_key=cfg.llm.api_key,
        temperature=cfg.llm.get("default_temperature", 0.7),
        max_tokens=cfg.llm.get("default_max_tokens", 2048),
    )
    logger.info(f"Langchain LLM ({cfg.llm.model_name}) инициализирован для агентов.")

    rag_instance: Optional[LightRAG] = None
    try:
        logger.info("Инициализация LightRAG...")
        rag_instance = await initialize_lightrag_instance(cfg)
        logger.info("Экземпляр LightRAG успешно инициализирован.")
    except Exception as e:  # Более общая обработка ошибок
        logger.critical(
            f"Не удалось инициализировать LightRAG: {e}. Бот не может быть запущен.",
            exc_info=True,
        )
        if engine:
            await engine.dispose()
        return

    if not rag_instance:
        logger.critical("Экземпляр rag_instance не был создан. Завершение работы.")
        if engine:
            await engine.dispose()
        return

    # Создание графа ИИ Тьютора
    tutor_graph: Optional[StatefulGraph] = None
    try:
        tutor_graph = create_tutor_graph(
            llm=langchain_llm, rag_instance=rag_instance, session_pool=session_pool
        )
        logger.info("Граф ИИ Тьютора успешно создан.")
    except Exception as e:
        logger.critical(f"Не удалось создать граф ИИ Тьютора: {e}", exc_info=True)
        if engine:
            await engine.dispose()
        return

    storage = MemoryStorage()
    logger.info(f"Хранилище FSM: {type(storage).__name__}")

    dp_kwargs = {
        "storage": storage,
        "session_pool": session_pool,
        "bot_texts": bot_texts,
        "engine": engine,
        "rag_instance": rag_instance,  # Для админских функций и, возможно, старой логики
        "llm_base_config": cfg.llm,  # Конфигурация LLM для LightRAG, если он ее использует
        "langchain_llm": langchain_llm,  # LLM для агентов LangGraph
        "tutor_graph": tutor_graph,  # Скомпилированный граф
        "cfg": cfg,
    }
    if cfg.langfuse.enabled and langfuse_callback_handler:
        dp_kwargs["langfuse_handler"] = langfuse_callback_handler
        logger.info("Langfuse callback handler передан в Dispatcher.")

    dp = Dispatcher(**dp_kwargs)
    logger.info("Диспетчер создан и сконфигурирован.")

    dp.include_router(user_router)
    dp.include_router(admin_router)
    logger.info("Пользовательские и администраторские роутеры зарегистрированы.")

    bot = Bot(
        token=cfg.server.API_KEY,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    # ... (остальная часть main как была) ...
    bot_info = await bot.get_me()
    logger.info(f"Экземпляр бота создан для @{bot_info.username} (ID: {bot_info.id}).")

    await set_bot_commands(bot)

    try:
        await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())
    except Exception as e:
        logger.critical(
            f"Необработанное исключение во время опроса (polling): {e}", exc_info=True
        )
    finally:
        logger.warning("Остановка бота...")
        if engine:
            await engine.dispose()
            logger.info("Соединение с базой данных закрыто.")

        global langfuse_client  # Используем глобальную переменную
        if cfg and cfg.langfuse.enabled and langfuse_client:
            langfuse_client.flush()
            logger.info("Langfuse сессия обработана перед остановкой.")

        if bot and bot.session:
            await bot.session.close()
            logger.info("Сессия Telegram бота закрыта.")
        logger.warning("Бот остановлен.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Бот остановлен вручную (KeyboardInterrupt или SystemExit).")
