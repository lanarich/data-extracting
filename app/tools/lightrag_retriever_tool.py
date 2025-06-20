from typing import Any, List, Optional, Type

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from lightrag.lightrag.components.retriever import (  # Убедитесь, что путь импорта LightRAG правильный
    LightRAG,
)
from lightrag.lightrag.core.types import (  # Убедитесь, что пути импорта QueryParam и Document правильные
    Document,
    QueryParam,
)
from loguru import logger


class LightRAGRetrieverInput(BaseModel):
    query: str = Field(
        description="Поисковый запрос для извлечения релевантных документов"
    )
    top_k: Optional[int] = Field(
        default=5, description="Количество извлекаемых документов"
    )
    # chat_history: Optional[List[Dict[str, str]]] = Field(default=None, description="История чата для контекстуального поиска") # Пока уберем, т.к. LightRAG.aquery с mode='docs_only' может не использовать историю


class LightRAGRetrieverTool(BaseTool):
    """
    Инструмент для извлечения документов с использованием LightRAG.
    Возвращает только извлеченные фрагменты текста (контекст).
    """

    name: str = "lightrag_document_retriever"
    description: str = (
        "Используется для поиска и извлечения релевантных фрагментов текста "
        "из базы знаний по заданному запросу. "
        "Вход: строка запроса. Выход: список извлеченных текстовых фрагментов."
    )
    args_schema: Type[BaseModel] = LightRAGRetrieverInput
    rag_instance: LightRAG

    def _run(self, query: str, top_k: int = 5, **kwargs: Any) -> List[str]:
        """Синхронный вызов (не используется в асинхронном графе, но должен быть реализован)."""
        logger.warning(
            "LightRAGRetrieverTool._run вызван синхронно, предпочтительно использовать _arun."
        )
        # Это упрощенная реализация, в идеале нужно использовать asyncio.run() или аналогичный подход,
        # но BaseTool требует _run. Для нашего асинхронного графа он не будет основным.
        try:
            param = QueryParam(
                query=query,  # query передается напрямую в aquery
                mode="docs_only",  # Извлекаем только документы
                top_k=top_k,
                # conversation_history=chat_history or [], # Пока убрано
            )
            # Поскольку rag_instance.aquery асинхронный, для синхронного вызова нужен event loop.
            # Это плохая практика вызывать async из sync таким образом в библиотечном коде.
            # Вместо этого, мы можем просто вернуть ошибку или пустой список.
            # raise NotImplementedError("Синхронный вызов _run не поддерживается для этого асинхронного инструмента.")
            # Или, если очень нужно, но не рекомендуется:
            # import asyncio
            # return asyncio.run(self._arun(query=query, top_k=top_k, **kwargs))
            logger.error(
                "Синхронный вызов _run не реализован должным образом для асинхронного LightRAG."
            )
            return ["Синхронный вызов не поддерживается"]
        except Exception as e:
            logger.error(f"Ошибка в LightRAGRetrieverTool._run: {e}")
            return [f"Ошибка при синхронном извлечении: {e}"]

    async def _arun(self, query: str, top_k: int = 5, **kwargs: Any) -> List[str]:
        """
        Асинхронно извлекает документы с использованием LightRAG.
        Возвращает список текстовых содержимых документов.
        """
        logger.info(
            f"LightRAGRetrieverTool._arun вызван с запросом: '{query}', top_k: {top_k}"
        )
        try:
            # Параметры для LightRAG.aquery
            # chat_history = kwargs.get("chat_history") # Если будем передавать историю
            query_param = QueryParam(
                mode="docs_only",  # Указываем LightRAG вернуть только документы
                top_k=top_k,
                # conversation_history=chat_history if chat_history else [], # Пока убрано
            )

            # Вызов LightRAG.aquery
            # rag_instance передается при инициализации инструмента
            response_object = await self.rag_instance.aquery(
                query_text=query,
                param=query_param
                # system_prompt здесь не нужен, т.к. мы не генерируем ответ LLM
            )

            retrieved_doc_texts: List[str] = []
            if (
                response_object
                and hasattr(response_object, "context_data")
                and response_object.context_data
            ):
                if (
                    hasattr(response_object.context_data, "docs_with_score")
                    and response_object.context_data.docs_with_score
                ):
                    for doc_with_score in response_object.context_data.docs_with_score:
                        if (
                            isinstance(doc_with_score.document, Document)
                            and doc_with_score.document.text
                        ):
                            retrieved_doc_texts.append(doc_with_score.document.text)
                        elif isinstance(
                            doc_with_score.document, str
                        ):  # На случай если документ это просто строка
                            retrieved_doc_texts.append(doc_with_score.document)
                    logger.info(
                        f"Извлечено {len(retrieved_doc_texts)} документов через LightRAG."
                    )
                else:
                    logger.warning(
                        "LightRAG response_object.context_data не содержит 'docs_with_score'."
                    )
            else:
                logger.warning(
                    "Ответ от LightRAG.aquery пуст или не содержит 'context_data'."
                )

            return retrieved_doc_texts

        except Exception as e:
            logger.error(
                f"Ошибка в LightRAGRetrieverTool._arun для запроса '{query}': {e}",
                exc_info=True,
            )
            return [f"Ошибка при извлечении документов: {e}"]

    # Если вам нужно передавать экземпляр rag_instance динамически или он сложен для pydantic:
    class Config:
        arbitrary_types_allowed = True
