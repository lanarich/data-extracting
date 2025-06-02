from typing import Any, Callable, Dict, List, Optional

import numpy as np
from lightrag import LightRAG, QueryParam
from lightrag.base import DocStatus
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc
from loguru import logger


class RAGServiceError(Exception):
    """Базовый класс для ошибок RAG сервиса."""

    pass


async def llm_standard_mode_func(
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: Optional[List[Dict[str, str]]] = None,
    llm_base_config: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> str:
    """LLM функция для стандартного режима."""
    config = llm_base_config or {}
    model_name = config.get("model_name", "Qwen/Qwen3-8B")
    base_url = config.get("base_url", "http://localhost:30000/v1")
    api_key = config.get("api_key", "NO")
    max_tokens = config.get("max_input_tokens", 2000)

    logger.debug(f"Standard LLM call: model='{model_name}', prompt_len={len(prompt)}")

    logger.debug(
        f"--- Standard LLM Call Start ---\n"
        f"Model: {model_name}\n"
        f"Base URL: {base_url}\n"
        f"Prompt (first 200 chars): {prompt[:200]}...\n"
        f"Prompt Length: {len(prompt)}\n"
        f"System Prompt: {system_prompt}\n"
        f"History Messages: {history_messages}\n"
        f"Remaining **kwargs: {kwargs}\n"
        f"--- Standard LLM Call End ---"
    )

    return await openai_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages if history_messages else [],
        api_key=api_key,
        base_url=base_url,
        temperature=0.7,
        top_p=0.8,
        max_tokens=max_tokens,
        extra_body={
            "top_k": 20,
            "min_p": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        **kwargs,
    )


async def llm_thinking_mode_func(
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: Optional[List[Dict[str, str]]] = None,
    llm_base_config: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> str:
    """LLM функция для режима 'Размышление'."""
    config = llm_base_config or {}
    model_name = config.get("model_name", "Qwen/Qwen3-8B")
    base_url = config.get("base_url", "http://localhost:30000/v1")
    api_key = config.get("api_key", "NO")
    max_tokens = config.get("max_input_tokens", 2000)

    logger.debug(f"Thinking LLM call: model='{model_name}', prompt_len={len(prompt)}")

    logger.debug(
        f"--- Standard LLM Call Start ---\n"
        f"Model: {model_name}\n"
        f"Base URL: {base_url}\n"
        f"Prompt (first 200 chars): {prompt[:200]}...\n"
        f"Prompt Length: {len(prompt)}\n"
        f"System Prompt: {system_prompt}\n"
        f"History Messages: {history_messages}\n"
        f"Remaining **kwargs: {kwargs}\n"
        f"--- Standard LLM Call End ---"
    )

    return await openai_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages if history_messages else [],
        api_key=api_key,
        base_url=base_url,
        temperature=0.6,
        top_p=0.95,
        max_tokens=max_tokens,
        extra_body={
            "top_k": 20,
            "min_p": 0,
            "chat_template_kwargs": {"enable_thinking": True},
        },
        **kwargs,
    )


async def configured_embedding_func(
    texts: List[str], embedding_config: Dict[str, Any]
) -> np.ndarray:
    model_name = embedding_config.get("model_name", "sergeyzh/BERTA")
    base_url = embedding_config.get("base_url", "http://localhost:7997")
    api_key = embedding_config.get("api_key", "NO")
    logger.debug(
        f"RAG Embedding call: model='{model_name}', base_url='{base_url}', num_texts={len(texts)}"
    )
    return await openai_embed(
        texts, model=model_name, api_key=api_key, base_url=base_url
    )


async def initialize_lightrag_instance(app_config) -> LightRAG:
    try:
        if not hasattr(app_config, "llm") or not hasattr(app_config, "embedding"):
            raise RAGServiceError(
                "Конфигурация для LLM (cfg.llm) или эмбеддингов (cfg.embedding) отсутствует."
            )

        llm_base_cfg_dict = {
            "model_name": app_config.llm.get("model_name", "Qwen/Qwen3-8B"),
            "base_url": app_config.llm.get("api_base", "http://localhost:30000/v1"),
            "api_key": app_config.llm.get("api_key", "NO"),
            "default_max_tokens": app_config.llm.get("max_input_tokens", 2000),
        }

        embedding_cfg_dict = {
            "model_name": app_config.embedding.get("model_name", "sergeyzh/BERTA"),
            "base_url": app_config.embedding.get("api_base", "http://localhost:7997"),
            "api_key": app_config.embedding.get("api_key", "NO"),
            "dim": app_config.embedding.get("dim", 768),
            "max_tokens": app_config.embedding.get("max_input_tokens", 8192),
        }

        rag_settings = app_config.get("rag_settings", {})
        db_settings = app_config.get("qdrant", {})

        default_llm_func_wrapper = lambda prompt, system_prompt=None, history_messages=None, **kwargs: llm_standard_mode_func(
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            llm_base_config=llm_base_cfg_dict,
            **kwargs,
        )

        rag = LightRAG(
            working_dir=rag_settings.get("working_dir", "./lightrag_data"),
            llm_model_func=default_llm_func_wrapper,
            embedding_func=EmbeddingFunc(
                embedding_dim=embedding_cfg_dict["dim"],
                max_token_size=embedding_cfg_dict["max_tokens"],
                func=lambda texts: configured_embedding_func(texts, embedding_cfg_dict),
            ),
            chunk_token_size=rag_settings.get("chunk_token_size", 512),
            chunk_overlap_token_size=rag_settings.get("chunk_overlap_token_size", 75),
            tiktoken_model_name=rag_settings.get("tiktoken_model_name", "gpt-4o-mini"),
            llm_model_max_async=rag_settings.get("llm_model_max_async", 10),
            max_parallel_insert=rag_settings.get("max_parallel_insert", 4),
            enable_llm_cache=rag_settings.get("enable_llm_cache", True),
            vector_storage="QdrantVectorDBStorage",
            vector_db_storage_cls_kwargs={
                "cosine_better_than_threshold": db_settings.get(
                    "cosine_better_than_threshold", 0.2
                )
            },
            addon_params=rag_settings.get(
                "addon_params",
                {
                    "language": "Русский",
                    "insert_batch_size": 50,
                },
            ),
        )

        await rag.initialize_storages()
        await initialize_pipeline_status()
        logger.info(
            "Экземпляр LightRAG успешно инициализирован со стандартным режимом LLM по умолчанию."
        )
        return rag
    except Exception as e:
        logger.critical(f"Ошибка при инициализации LightRAG: {e}", exc_info=True)
        raise RAGServiceError(f"Не удалось инициализировать LightRAG: {e}")


async def get_rag_answer(
    rag_instance: LightRAG,
    query_text: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
    user_llm_mode: str = "standard",
    llm_base_config: Optional[Dict[str, Any]] = None,
    top_k: int = 5,
    mode: str = "hybrid",
) -> str:
    if not rag_instance:
        logger.error("Экземпляр LightRAG не инициализирован.")
        raise RAGServiceError("LightRAG не инициализирован.")
    if not llm_base_config:
        logger.error(
            "Базовая конфигурация LLM (llm_base_config) не передана в get_rag_answer."
        )
        raise RAGServiceError("Отсутствует базовая конфигурация LLM для запроса.")

    try:
        selected_model_func: Optional[Callable] = None
        if user_llm_mode == "thinking":
            # Обертка для передачи llm_base_config
            selected_model_func = lambda prompt, system_prompt=None, history_messages=None, **kwargs: llm_thinking_mode_func(
                prompt,
                system_prompt,
                history_messages,
                llm_base_config=llm_base_config,
                **kwargs,
            )
            logger.debug(
                "Для запроса будет использован llm_thinking_mode_func через QueryParam.model_func."
            )

        else:
            logger.debug(
                "Для запроса будет использована стандартная LLM функция LightRAG."
            )

        params = QueryParam(
            mode=mode,
            top_k=top_k,
            conversation_history=chat_history if chat_history else [],
            model_func=selected_model_func,
        )

        logger.debug(f"Запрос к RAG: query='{query_text}', params={params}")
        response_object = await rag_instance.aquery(query_text, param=params)

        answer = "Не удалось извлечь ответ из объекта RAG."
        if hasattr(response_object, "answer") and response_object.answer is not None:
            answer = response_object.answer
        elif isinstance(response_object, str):
            answer = response_object
        elif (
            hasattr(response_object, "final_response")
            and response_object.final_response is not None
        ):
            answer = response_object.final_response
        else:
            logger.warning(
                f"Неожиданный или пустой формат ответа от LightRAG: {response_object}. Тип: {type(response_object)}"
            )
            answer = "К сожалению, не удалось сформировать ответ."
        logger.info(f"Ответ от RAG получен. Длина ответа: {len(answer)} символов.")
        return answer
    except Exception as e:
        logger.error(
            f"Ошибка при получении ответа от RAG для запроса '{query_text}': {e}",
            exc_info=True,
        )
        raise RAGServiceError(f"Ошибка при обработке RAG запроса: {e}")


async def add_document_contents_to_rag(
    rag_instance: LightRAG,
    documents_content: List[str],
    document_ids: List[str],
    file_path: List[str],
    batch_size: Optional[int] = None,
) -> bool:
    if not rag_instance:
        logger.error("Экземпляр LightRAG не инициализирован для добавления документов.")
        raise RAGServiceError("LightRAG не инициализирован.")
    if len(documents_content) != len(document_ids):
        logger.error("Количество содержимого документов не совпадает с количеством ID.")
        raise ValueError("Количество контента и ID должно совпадать.")
    if not documents_content:
        logger.info("Нет документов для добавления в RAG.")
        return True

    original_batch_size = None
    try:
        if (
            batch_size
            and hasattr(rag_instance, "addon_params")
            and isinstance(rag_instance.addon_params, dict)
        ):
            original_batch_size = rag_instance.addon_params.get("insert_batch_size")
            rag_instance.addon_params["insert_batch_size"] = batch_size
            logger.info(
                f"Временный размер пакета для вставки установлен на: {batch_size}"
            )

        logger.info(
            f"Начало добавления {len(documents_content)} документов в LightRAG. IDs: {document_ids[:3]}..., file_paths также будут этими IDs."
        )
        await rag_instance.ainsert(
            documents_content, ids=document_ids, file_paths=file_path
        )
        logger.info("Команда ainsert вызвана. Ожидание завершения индексации...")

        ids_to_monitor = set(document_ids)

        if hasattr(rag_instance, "aget_docs_by_ids"):
            try:
                processed_quickly_infos = await rag_instance.aget_docs_by_ids(
                    list(ids_to_monitor)
                )

                for doc_info in processed_quickly_infos:
                    if (
                        doc_info
                        and hasattr(doc_info, "id")
                        and doc_info.id in ids_to_monitor
                        and hasattr(doc_info, "status")
                        and doc_info.status == DocStatus.PROCESSED
                    ):
                        ids_to_monitor.remove(doc_info.id)
                if not ids_to_monitor:
                    logger.info(
                        "Все добавленные документы, похоже, были обработаны очень быстро."
                    )
            except Exception as e_quick_check:
                logger.warning(
                    f"Ошибка при быстрой проверке статусов через aget_docs_by_ids: {e_quick_check}"
                )
        else:
            logger.warning(
                "Метод aget_docs_by_ids отсутствует в LightRAG, быстрая проверка статусов невозможна."
            )

        if (
            original_batch_size is not None
            and hasattr(rag_instance, "addon_params")
            and isinstance(rag_instance.addon_params, dict)
        ):
            rag_instance.addon_params["insert_batch_size"] = original_batch_size
            logger.info(
                f"Размер пакета для вставки восстановлен на: {original_batch_size}"
            )

        if hasattr(rag_instance, "doc_status") and hasattr(
            rag_instance.doc_status, "get_status_counts"
        ):
            status_counts = await rag_instance.doc_status.get_status_counts()
            logger.info(
                f"Общий статус документов в RAG: Всего: {sum(status_counts.values())}, Обработано: {status_counts.get(DocStatus.PROCESSED, 0)}, В обработке: {status_counts.get(DocStatus.PROCESSING, 0)}, Ошибки: {status_counts.get(DocStatus.FAILED, 0)}"
            )
        else:
            logger.warning(
                "Не удалось получить общую статистику статусов документов из rag_instance.doc_status."
            )

        logger.info("Процесс добавления и индексации документов в LightRAG завершен.")
        return True
    except Exception as e:
        logger.error(
            f"Ошибка при добавлении/индексации документов в LightRAG: {e}",
            exc_info=True,
        )
        if (
            original_batch_size is not None
            and hasattr(rag_instance, "addon_params")
            and isinstance(rag_instance.addon_params, dict)
        ):
            rag_instance.addon_params["insert_batch_size"] = original_batch_size
        return False


async def delete_document_from_rag(rag_instance: LightRAG, document_id: str) -> bool:
    if not rag_instance:
        logger.error("Экземпляр LightRAG не инициализирован для удаления документа.")
        raise RAGServiceError("LightRAG не инициализирован.")
    try:
        if hasattr(rag_instance, "adelete_by_doc_id"):
            await rag_instance.adelete_by_doc_id(doc_id=document_id)
            await rag_instance.doc_status.delete([document_id])
            doc_status = await rag_instance.aget_docs_by_ids([document_id])
            if document_id in doc_status:
                logger.warning(
                    f"Документ {document_id} все еще существует после удаления"
                )
            logger.info(
                f"Команда на удаление документа с ID '{document_id}' успешно выполнена в LightRAG."
            )
            return True
        else:
            logger.warning(
                f"Метод 'adelete_by_doc_id' для удаления документа из LightRAG не найден. Документ '{document_id}' не удален из RAG-хранилища."
            )
            return False
    except Exception as e:
        logger.error(
            f"Ошибка при удалении документа '{document_id}' из LightRAG: {e}",
            exc_info=True,
        )
        return False


async def get_multiple_document_statuses_from_rag(
    rag_instance: LightRAG, document_ids: List[str]
) -> Dict[str, Optional[str]]:
    if not rag_instance:
        logger.error("Экземпляр LightRAG не инициализирован для получения статусов.")
        return {doc_id: "RAG_OFFLINE" for doc_id in document_ids}

    statuses_map: Dict[str, Optional[str]] = {
        doc_id: "NOT_IN_RAG" for doc_id in document_ids
    }

    if not document_ids:
        return statuses_map

    method_name = "aget_docs_by_ids"

    if not hasattr(rag_instance, method_name):
        logger.error(
            f"Критическая ошибка: Метод {method_name} отсутствует в экземпляре LightRAG. "
            f"Функция не может продолжить работу."
        )
        for doc_id in document_ids:
            statuses_map[doc_id] = "RAG_UNAVAILABLE_METHOD"
        return statuses_map

    logger.debug(
        f"Попытка получить статусы RAG для {len(document_ids)} документов через {method_name}..."
    )
    try:
        docs_info_from_rag: Optional[Dict[str, Any]] = await getattr(
            rag_instance, method_name
        )(document_ids)

        if not isinstance(docs_info_from_rag, dict):
            logger.error(
                f"{method_name} не вернул словарь, как ожидалось. "
                f"Получено: {type(docs_info_from_rag)}. Содержимое: {str(docs_info_from_rag)[:200]}"
            )
            for doc_id in document_ids:
                statuses_map[doc_id] = "RAG_RESPONSE_ERROR"
        else:
            for req_doc_id in document_ids:
                doc_data = docs_info_from_rag.get(req_doc_id)

                if doc_data and isinstance(doc_data, dict) and "status" in doc_data:
                    status_val = doc_data["status"]
                    if isinstance(status_val, DocStatus):
                        statuses_map[req_doc_id] = status_val.value
                    elif isinstance(status_val, str):
                        try:
                            statuses_map[req_doc_id] = DocStatus(status_val).value
                        except ValueError:
                            statuses_map[req_doc_id] = status_val
                            logger.debug(
                                f"Статус '{status_val}' для документа {req_doc_id} "
                                f"не является стандартным членом DocStatus enum, используется как есть."
                            )
                    else:
                        statuses_map[req_doc_id] = str(status_val)
                        logger.warning(
                            f"Статус для документа {req_doc_id} имеет неожиданный тип: {type(status_val)}."
                        )
                elif doc_data:
                    statuses_map[req_doc_id] = "STATUS_UNKNOWN"
                    logger.warning(
                        f"Информация о документе {req_doc_id} получена из RAG, "
                        f"но ключ 'status' отсутствует или его значение некорректно: {str(doc_data)[:200]}"
                    )

    except Exception as e:
        logger.error(
            f"Ошибка при вызове {method_name} для LightRAG: {e}", exc_info=True
        )
        for doc_id in document_ids:
            statuses_map[doc_id] = "RAG_STATUS_ERROR"

    logger.debug(
        f"Итоговые статусы из RAG для {len(document_ids)} документов: {statuses_map}"
    )
    return statuses_map
