import asyncio
import os
import sys

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import BotCommand, BotCommandScopeAllPrivateChats
from handlers.admin import admin_router
from handlers.user import user_router
from hydra import compose, initialize
from loguru import logger
from services.db_service import (
    create_db_engine_and_session_pool,
    create_tables,
    get_db_url,
)
from services.rag_service import RAGServiceError, initialize_lightrag_instance
from utils.config_loader import load_texts_from_config

try:
    with initialize(
        version_base=None, config_path="utils/conf", job_name="medical_university_bot"
    ):
        cfg = compose(config_name="config")
    logger.info("Конфигурация Hydra успешно загружена.")

    if not hasattr(cfg, "server") or not cfg.server.API_KEY:
        raise ValueError("Отсутствует API_KEY в конфигурации (server.API_KEY).")
    if not cfg.server.DB_NAME:
        raise ValueError("Отсутствует DB_NAME в конфигурации (server.DB_NAME).")
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
    if hasattr(cfg, "logging") and hasattr(cfg.logging, "console_level")
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
        await create_tables(engine)
        logger.info(
            f"Соединение с базой данных ({cfg.server.DB_NAME}@{cfg.server.DB_HOST}) установлено и таблицы проверены/созданы."
        )
    except ValueError as e:
        logger.critical(f"Ошибка конфигурации для подключения к БД: {e}", exc_info=True)
        return
    except Exception as e:
        logger.critical(
            f"Не удалось инициализировать соединение с базой данных: {e}", exc_info=True
        )
        return

    llm_base_cfg_dict = {
        "model_name": cfg.llm.get("model_name", "Qwen/Qwen3-8B"),
        "base_url": cfg.llm.get("api_base", "http://localhost:30000/v1"),
        "api_key": cfg.llm.get("api_key", "NO"),
        "default_temperature": cfg.llm.get("default_temperature", 0.7),
        "default_max_tokens": cfg.llm.get("default_max_tokens", 2048),
    }
    logger.info(f"Базовая конфигурация LLM подготовлена: {llm_base_cfg_dict}")

    rag_instance = None
    try:
        logger.info("Инициализация LightRAG...")
        rag_instance = await initialize_lightrag_instance(cfg)
        logger.info("Экземпляр LightRAG успешно инициализирован.")
    except RAGServiceError as e:
        logger.critical(
            f"Не удалось инициализировать LightRAG: {e}. Бот не может быть запущен.",
            exc_info=True,
        )
        if engine:
            await engine.dispose()
        return
    except Exception as e:
        logger.critical(
            f"Неожиданная ошибка при инициализации LightRAG: {e}. Бот не может быть запущен.",
            exc_info=True,
        )
        if engine:
            await engine.dispose()
        return

    storage = MemoryStorage()
    logger.info(f"Хранилище FSM: {type(storage).__name__}")

    dp = Dispatcher(
        storage=storage,
        session_pool=session_pool,
        bot_texts=bot_texts,
        engine=engine,
        rag_instance=rag_instance,
        llm_base_config=llm_base_cfg_dict,
    )
    logger.info("Диспетчер создан и сконфигурирован.")

    dp.include_router(user_router)
    dp.include_router(admin_router)
    logger.info("Пользовательские и администраторские роутеры зарегистрированы.")

    bot = Bot(
        token=cfg.server.API_KEY,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
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
        if "engine" in dp.workflow_data and dp.workflow_data["engine"]:
            await dp.workflow_data["engine"].dispose()
            logger.info("Соединение с базой данных закрыто.")

        if bot and bot.session:
            await bot.session.close()
            logger.info("Сессия Telegram бота закрыта.")
        logger.warning("Бот остановлен.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Бот остановлен вручную (KeyboardInterrupt или SystemExit).")
    except Exception as e:
        print(f"КРИТИЧЕСКАЯ ОШИБКА (во время запуска/остановки): {e}", file=sys.stderr)
        logger.critical(
            f"Критическая ошибка во время запуска или глобальной остановки бота: {e}",
            exc_info=True,
        )
        sys.exit(1)
