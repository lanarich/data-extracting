import asyncio
import os
import tempfile
from pathlib import Path
from typing import Any, Optional, Tuple

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    EasyOcrOptions,
    PdfPipelineOptions,
    TableFormerMode,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from loguru import logger


class DocumentParsingError(Exception):
    """Исключение для ошибок при парсинге документов."""

    pass


pdf_pipeline_options = PdfPipelineOptions(images_scale=2.0)
pdf_pipeline_options.do_table_structure = True
pdf_pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
pdf_pipeline_options.table_structure_options.do_cell_matching = True

pdf_pipeline_options.generate_page_images = False
pdf_pipeline_options.generate_picture_images = False
pdf_pipeline_options.generate_table_images = False
pdf_pipeline_options.do_picture_classification = False
pdf_pipeline_options.do_picture_description = False

pdf_pipeline_options.do_ocr = True
pdf_pipeline_options.ocr_options = EasyOcrOptions(
    lang=["ru", "en"], force_full_page_ocr=True, confidence_threshold=0.5
)

docling_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_pipeline_options)
    }
)
logger.info(
    "Docling DocumentConverter инициализирован с кастомными настройками для PDF."
)


async def parse_document_to_markdown(file_path: str) -> Optional[str]:
    """
    Парсит содержимое документа из файла и экспортирует его в Markdown строку.

    Args:
        file_path: Путь к файлу документа.

    Returns:
        Строка с содержимым документа в формате Markdown или None, если парсинг не удался.

    Raises:
        FileNotFoundError: Если файл не найден.
        DocumentParsingError: Если произошла ошибка во время парсинга.
    """
    path = Path(file_path)
    if not path.is_file():
        logger.error(f"Файл для парсинга не найден: {file_path}")
        raise FileNotFoundError(f"Файл не найден: {file_path}")

    try:
        logger.info(
            f"Начало парсинга документа для Markdown: {file_path} с помощью DocumentConverter."
        )

        loop = asyncio.get_event_loop()
        conversion_result: Optional[Any] = await loop.run_in_executor(
            None, docling_converter.convert, str(path)
        )

        if (
            not conversion_result
            or not hasattr(conversion_result, "document")
            or conversion_result.document is None
        ):
            logger.warning(
                f"Парсинг документа {file_path} не вернул валидный объект 'result.document'."
            )
            return None

        docling_doc_object: Any = conversion_result.document

        if hasattr(docling_doc_object, "export_to_markdown"):
            markdown_text = docling_doc_object.export_to_markdown()
            if isinstance(markdown_text, str) and markdown_text.strip():
                logger.info(
                    f"Документ {file_path} успешно сконвертирован в Markdown. Длина текста: {len(markdown_text)}."
                )
                return markdown_text.strip()
            else:
                logger.warning(
                    f"Экспорт в Markdown для документа {file_path} вернул пустой или не строковый результат: {type(markdown_text)}."
                )
                return None
        else:
            logger.error(
                f"Объект документа, полученный из Docling для {file_path}, не имеет метода export_to_markdown()."
            )
            raise DocumentParsingError(
                f"Не удалось экспортировать документ {file_path} в Markdown: отсутствует метод export_to_markdown."
            )

    except Exception as e:
        logger.error(
            f"Ошибка при парсинге или экспорте документа {file_path} в Markdown: {e}",
            exc_info=True,
        )
        raise DocumentParsingError(
            f"Не удалось распарсить и экспортировать документ {file_path} в Markdown: {e}"
        )


async def save_uploaded_file_temp(
    file_content: bytes, original_file_name: str
) -> Tuple[str, str]:
    try:
        suffix = Path(original_file_name).suffix
        temp_dir = Path(tempfile.gettempdir()) / "bot_uploads"
        temp_dir.mkdir(parents=True, exist_ok=True)

        fd, temp_file_path = tempfile.mkstemp(
            suffix=suffix, prefix="upload_", dir=str(temp_dir)
        )

        with os.fdopen(fd, "wb") as tmp_file:
            tmp_file.write(file_content)

        temp_file_name = Path(temp_file_path).name
        logger.info(
            f"Файл '{original_file_name}' временно сохранен как '{temp_file_name}' по пути: {temp_file_path}"
        )
        return temp_file_path, temp_file_name
    except Exception as e:
        logger.error(
            f"Ошибка при сохранении временного файла для '{original_file_name}': {e}",
            exc_info=True,
        )
        raise


def cleanup_temp_file(file_path: str):
    try:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Временный файл {file_path} удален.")
    except Exception as e:
        logger.error(
            f"Ошибка при удалении временного файла {file_path}: {e}", exc_info=True
        )
