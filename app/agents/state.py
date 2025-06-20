from typing import Dict, List, Optional

from langchain_core.messages import BaseMessage

# Если у вас есть модели Pydantic для Student, CurriculumTopic, StudentProgress,
# их можно импортировать и использовать здесь для более строгой типизации.
# from app.models.models import Student, CurriculumTopic # Пример


class TutorGraphState(Dict):
    """
    Состояние для графа ИИ Тьютора.
    Использует Dict для гибкости, но можно заменить на TypedDict или Pydantic модель.
    """

    student_id: int
    student_profile: Optional[Dict]  # Или Optional[Student]
    input_query: Optional[str]
    chat_history: List[
        BaseMessage
    ]  # Или List[Dict[str, str]] если используете простой формат

    current_topic_id: Optional[str]
    current_topic_name: Optional[str]
    current_learning_objectives: Optional[str]
    # current_topic_details: Optional[CurriculumTopic] # Пример

    retrieved_context: Optional[List[str]]
    assessment_question: Optional[str]
    student_answer: Optional[str]
    assessment_feedback: Optional[str]  # Оценка ответа студента

    recommendations: Optional[List[str]]  # Список рекомендаций

    error_message: Optional[str]  # Для отлова и передачи ошибок между узлами

    # Можно добавить другие поля по мере необходимости
    # Например, для хранения промежуточных результатов или флагов
    next_node_to_call: Optional[str]  # Для условных переходов, если понадобится

    # Langfuse trace object, если нужно передавать его явно
    # langfuse_trace: Optional[Any] # Тип зависит от объекта трейса Langfuse
