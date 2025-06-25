from typing import Annotated, Dict, List, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class TutorGraphState(TypedDict):
    """
    Состояние для графа ИИ Тьютора.
    Использует TypedDict для строгой типизации и совместимости с LangGraph.
    """

    # Основная информация о студенте
    student_id: int
    student_profile: Optional[Dict]

    # История сообщений с автоматическим добавлением
    chat_history: Annotated[List[BaseMessage], add_messages]

    # Пользовательский ввод
    input_query: Optional[str]

    # Информация о текущей теме
    current_topic_id: Optional[str]
    current_topic_name: Optional[str]
    current_learning_objectives: Optional[str]

    # Извлеченный контекст
    retrieved_context: Optional[List[str]]
    retrieval_query: Optional[str]  # Для отладки
    retrieval_source: Optional[str]  # Источник запроса

    # Оценка знаний
    assessment_question: Optional[str]
    student_answer: Optional[str]
    assessment_feedback: Optional[str]
    assessment_results: Optional[Dict]

    # Рекомендации
    recommendations: Optional[List[str]]

    # Обработка ошибок
    error_message: Optional[str]
    user_message: Optional[str]  # Пользовательское сообщение об ошибке
    error_handled: Optional[bool]

    # Дополнительные поля для управления потоком
    next_node_to_call: Optional[str]
    retrieval_top_k: Optional[int]
