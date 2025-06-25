from typing import Any, List, Optional, Type

from langchain_core.tools import BaseTool
from lightrag import LightRAG, QueryParam
from loguru import logger
from pydantic import BaseModel, Field


class LightRAGRetrieverInput(BaseModel):
    query: str = Field(
        description="Поисковый запрос для извлечения релевантных документов"
    )
    top_k: Optional[int] = Field(
        default=5, description="Количество извлекаемых документов"
    )


class LightRAGRetrieverTool(BaseTool):
    name: str = "lightrag_document_retriever"
    description: str = (
        "Используется для поиска и извлечения релевантных фрагментов текста "
        "из базы знаний по заданному запросу. "
        "Вход: строка запроса. Выход: список извлеченных текстовых фрагментов."
    )
    args_schema: Type[BaseModel] = LightRAGRetrieverInput
    rag_instance: LightRAG

    class Config:
        arbitrary_types_allowed = True

    async def _arun(
        self, query: str, top_k: int = 3, mode="global", **kwargs: Any
    ) -> List[str]:
        try:
            params = QueryParam(
                mode=mode,
                top_k=top_k,
                max_token_for_text_unit=1000,  # Максимум токенов для текстового фрагмента
                max_token_for_global_context=1000,  # Максимум токенов для глобального контекста
                max_token_for_local_context=1000,
            )
            context = await self.rag_instance.aquery(query, param=params)
            return context
        except Exception as e:
            logger.error(f"Ошибка в LightRAGRetrieverTool.run: {e}")
            return [f"Ошибка при aсинхронном извлечении: {e}"]

    def _run(self, *args, **kwargs):
        raise NotImplementedError(
            "Synchronous run is not supported, use _arun instead."
        )
