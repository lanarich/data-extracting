from pathlib import Path

import yaml
from loguru import logger


def load_yaml_file(file_path_str: str) -> dict:
    """
    Загружает YAML-файл из указанного пути.

    Args:
        file_path_str: Строка с путем к YAML-файлу.

    Returns:
        Словарь с данными из YAML-файла.

    Raises:
        FileNotFoundError: Если файл не найден.
        ValueError: Если произошла ошибка парсинга YAML или путь пустой.
        Exception: При других неожиданных ошибках.
    """
    if not file_path_str:
        logger.error("Путь к YAML файлу не может быть пустым.")
        raise ValueError("Путь к YAML файлу не указан.")

    file_path = Path(file_path_str)
    if not file_path.is_file():
        logger.error(f"YAML файл не найден по указанному пути: {file_path}")
        raise FileNotFoundError(f"Указанный YAML файл не существует: {file_path}")

    try:
        with file_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        logger.info(f"YAML файл успешно загружен: {file_path}")
        return data
    except yaml.YAMLError as e:
        logger.error(f"Ошибка парсинга YAML файла {file_path}: {e}")
        raise ValueError(f"Не удалось распарсить YAML файл {file_path}: {e}")
    except Exception as e:
        logger.error(
            f"Неожиданная ошибка при загрузке YAML файла {file_path}: {e}",
            exc_info=True,
        )
        raise


def load_texts_from_config(cfg) -> dict:
    texts_file_path_str = None
    try:
        if hasattr(cfg, "server") and hasattr(cfg.server, "TEXTS"):
            texts_file_path_str = cfg.server.TEXTS
            if not texts_file_path_str:
                logger.error(
                    "Путь к файлу текстов (cfg.server.TEXTS) пуст в конфигурации."
                )
                raise ValueError(
                    "Путь к файлу текстов (cfg.server.TEXTS) указан, но пуст."
                )

            logger.info(f"Попытка загрузки текстов из файла: {texts_file_path_str}")
            texts = load_yaml_file(texts_file_path_str)
            return texts
        else:
            logger.error(
                "Путь к файлу текстов (cfg.server.TEXTS) не найден в конфигурации."
            )
            raise AttributeError(
                "В конфигурации отсутствует 'server.TEXTS' для сообщений бота."
            )

    except FileNotFoundError:
        logger.error(
            f"Файл текстов, указанный в конфигурации ('{texts_file_path_str}'), не найден."
        )
        raise
    except ValueError as e:
        logger.error(
            f"Не удалось загрузить или распарсить файл текстов ('{texts_file_path_str}'): {e}"
        )
        raise
    except AttributeError as e:
        raise
    except Exception as e:
        logger.error(f"Неожиданная ошибка в load_texts_from_config: {e}", exc_info=True)
        raise
