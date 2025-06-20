from typing import Dict, Optional

from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI  # Для взаимодействия с LLM
from loguru import logger

from app.agents.state import TutorGraphState

# Системные промпты для LLM
GENERATE_QUESTION_SYSTEM_PROMPT = """
Ты - ИИ-Ассистент в роли преподавателя. Твоя задача - задать студенту один четкий и конкретный вопрос на основе предоставленного учебного материала (контекста) и целей обучения.
Вопрос должен проверять понимание студентом ключевых аспектов темы.
Не задавай слишком общих или слишком узких вопросов. Вопрос должен быть сформулирован на русском языке.
Контекст:
{context}

Цели обучения по теме "{topic_name}":
{learning_objectives}

Сформулируй ОДИН вопрос для студента. Не добавляй никаких пояснений или приветствий, только сам вопрос.
"""

EVALUATE_ANSWER_SYSTEM_PROMPT = """
Ты - ИИ-Ассистент в роли преподавателя. Твоя задача - оценить ответ студента на заданный ранее вопрос.
Используй предоставленный учебный материал (контекст) как основу для правильного ответа.
Дай краткую и конструктивную обратную связь. Укажи, что было правильно, а что нет, и почему.
Если ответ неполный, уточни, чего не хватает.
Отвечай на русском языке.

Контекст (правильная информация по теме):
{context}

Заданный вопрос:
{question}

Ответ студента:
{student_answer}

Твоя оценка и обратная связь:
"""


async def run_assessment_agent(state: TutorGraphState, llm: ChatOpenAI) -> Dict:
    """
    Узел LangGraph: Агент Оценки Знаний.
    Генерирует вопросы на основе контекста или оценивает ответы студента.
    """
    logger.info("Агент Оценки Знаний запущен.")

    current_topic_name = state.get("current_topic_name")
    learning_objectives = state.get("current_learning_objectives")
    retrieved_context_list = state.get("retrieved_context")
    student_answer = state.get("student_answer")  # Ответ студента на предыдущий вопрос
    # assessment_question - это вопрос, который был задан студенту ранее, если мы сейчас оцениваем ответ
    previously_asked_question = state.get("assessment_question")

    # Объединяем контекст в одну строку для промпта
    context_str = (
        "\n\n".join(retrieved_context_list)
        if retrieved_context_list
        else "Контекст не найден."
    )

    new_question_to_ask: Optional[str] = None
    feedback_on_answer: Optional[str] = None
    error_message: Optional[str] = None

    try:
        if student_answer and previously_asked_question:
            # Фаза: Оценка ответа студента
            logger.info(
                f"Фаза оценки. Вопрос: '{previously_asked_question}'. Ответ студента: '{student_answer}'"
            )
            prompt_template_evaluate = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(content=EVALUATE_ANSWER_SYSTEM_PROMPT),
                ]
            )
            evaluation_chain = prompt_template_evaluate | llm

            response = await evaluation_chain.ainvoke(
                {
                    "context": context_str,
                    "question": previously_asked_question,
                    "student_answer": student_answer,
                    "topic_name": current_topic_name
                    or "Текущая тема",  # Добавлено для полноты, если нужно в промпте
                    "learning_objectives": learning_objectives
                    or "Цели не указаны",  # Добавлено
                }
            )
            feedback_on_answer = response.content
            logger.info(f"Обратная связь LLM: {feedback_on_answer}")
            # После оценки ответа, сбрасываем student_answer и assessment_question, чтобы не оценивать повторно
            # и, возможно, готовимся задать новый вопрос или перейти к рекомендациям.
            # Пока просто сбросим, логика переходов будет в графе.

        elif (
            retrieved_context_list
        ):  # Если есть контекст, но нет ответа студента для оценки - генерируем новый вопрос
            # Фаза: Генерация нового вопроса
            logger.info(f"Фаза генерации вопроса по теме: {current_topic_name}")
            prompt_template_generate = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(content=GENERATE_QUESTION_SYSTEM_PROMPT),
                ]
            )
            generation_chain = prompt_template_generate | llm

            response = await generation_chain.ainvoke(
                {
                    "context": context_str,
                    "topic_name": current_topic_name or "Неизвестная тема",
                    "learning_objectives": learning_objectives
                    or "Цели обучения не определены.",
                }
            )
            new_question_to_ask = response.content
            logger.info(f"Сгенерирован новый вопрос LLM: {new_question_to_ask}")
        else:
            error_message = "Недостаточно данных для Агента Оценки (нет контекста для генерации вопроса и нет ответа для оценки)."
            logger.warning(error_message)

    except Exception as e:
        logger.error(f"Ошибка в Агенте Оценки Знаний: {e}", exc_info=True)
        error_message = f"Внутренняя ошибка в Агенте Оценки: {e}"

    update_data = {
        "assessment_question": new_question_to_ask
        if new_question_to_ask
        else None,  # Сохраняем новый заданный вопрос
        "assessment_feedback": feedback_on_answer if feedback_on_answer else None,
        "student_answer": None,  # Очищаем ответ студента после обработки
        # Если мы оценили ответ, то previously_asked_question тоже можно очистить или решить в графе, что с ним делать
        "error_message": state.get("error_message") or error_message,
    }
    # Если был сгенерирован новый вопрос, он сохранится в assessment_question.
    # Если был дан фидбек, он сохранится в assessment_feedback.
    # Логика графа решит, что делать дальше (отправить вопрос/фидбек пользователю).

    logger.debug(
        f"Агент Оценки Знаний обновил состояние: вопрос='{new_question_to_ask}', фидбек='{feedback_on_answer}'"
    )
    return update_data
