import asyncio
from typing import Dict, List, Optional

from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from loguru import logger

from app.agents.state import TutorGraphState

MAX_CONTEXT_LENGTH = 4000
MAX_RETRIES = 3
RETRY_DELAY = 1.0

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


def validate_state_inputs(state: TutorGraphState) -> Dict[str, str]:
    """Валидация входных данных из состояния"""
    errors = []

    current_topic_name = state.get("current_topic_name")
    if not current_topic_name or not isinstance(current_topic_name, str):
        errors.append("Отсутствует или некорректное название темы")

    learning_objectives = state.get("current_learning_objectives")
    if not learning_objectives or not isinstance(learning_objectives, str):
        errors.append("Отсутствуют или некорректные цели обучения")

    retrieved_context = state.get("retrieved_context")
    if retrieved_context and not isinstance(retrieved_context, list):
        errors.append("Контекст должен быть списком строк")

    return {
        "errors": errors,
        "current_topic_name": current_topic_name or "Неизвестная тема",
        "learning_objectives": learning_objectives or "Цели не определены",
        "retrieved_context": retrieved_context or [],
    }


def prepare_context(context_list: List[str]) -> str:
    """Подготовка контекста с ограничением длины"""
    if not context_list:
        return "Контекст не найден."

    # Объединяем контекст и проверяем длину
    full_context = "\n\n".join(context_list)

    if len(full_context) <= MAX_CONTEXT_LENGTH:
        return full_context

    # Если контекст слишком длинный, обрезаем по предложениям
    truncated_context = ""
    for item in context_list:
        if len(truncated_context + item) <= MAX_CONTEXT_LENGTH:
            truncated_context += item + "\n\n"
        else:
            break

    if not truncated_context:
        # Если даже первый элемент слишком длинный, обрезаем его
        truncated_context = context_list[0][:MAX_CONTEXT_LENGTH] + "..."

    logger.warning(
        f"Контекст обрезан с {len(full_context)} до {len(truncated_context)} символов"
    )
    return truncated_context


async def call_llm_with_retry(
    chain, inputs: Dict, max_retries: int = MAX_RETRIES
) -> Optional[str]:
    """Вызов LLM с retry логикой"""
    for attempt in range(max_retries):
        try:
            response = await chain.ainvoke(inputs)
            return response.content
        except Exception as e:
            logger.warning(f"Попытка {attempt + 1} не удалась: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(
                    RETRY_DELAY * (attempt + 1)
                )  # Экспоненциальная задержка
            else:
                logger.error(f"Все {max_retries} попыток вызова LLM не удались")
                raise e


async def run_assessment_agent(state: TutorGraphState, llm: ChatOpenAI) -> Dict:
    """
    Узел LangGraph: Агент Оценки Знаний.
    Генерирует вопросы на основе контекста или оценивает ответы студента.
    """
    logger.info("Агент Оценки Знаний запущен.")

    # Валидация входных данных
    validation_result = validate_state_inputs(state)
    if validation_result["errors"]:
        error_message = f"Ошибки валидации: {'; '.join(validation_result['errors'])}"
        logger.error(error_message)
        return {"error_message": error_message}

    current_topic_name = validation_result["current_topic_name"]
    learning_objectives = validation_result["learning_objectives"]
    retrieved_context_list = validation_result["retrieved_context"]

    student_answer = state.get("student_answer")
    previously_asked_question = state.get("assessment_question")

    # Подготовка контекста с ограничением длины
    context_str = prepare_context(retrieved_context_list)

    new_question_to_ask: Optional[str] = None
    feedback_on_answer: Optional[str] = None
    error_message: Optional[str] = None

    try:
        if student_answer and previously_asked_question:
            # Фаза: Оценка ответа студента
            logger.info(f"Фаза оценки. Вопрос: '{previously_asked_question[:100]}...'")

            if not isinstance(student_answer, str) or len(student_answer.strip()) == 0:
                error_message = "Ответ студента пуст или некорректен"
                logger.warning(error_message)
            else:
                prompt_template_evaluate = ChatPromptTemplate.from_messages(
                    [
                        SystemMessage(content=EVALUATE_ANSWER_SYSTEM_PROMPT),
                    ]
                )
                evaluation_chain = prompt_template_evaluate | llm

                inputs = {
                    "context": context_str,
                    "question": previously_asked_question,
                    "student_answer": student_answer,
                }

                feedback_on_answer = await call_llm_with_retry(evaluation_chain, inputs)
                logger.info("Обратная связь успешно сгенерирована")

        elif retrieved_context_list:
            # Фаза: Генерация нового вопроса
            logger.info(f"Фаза генерации вопроса по теме: {current_topic_name}")

            prompt_template_generate = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(content=GENERATE_QUESTION_SYSTEM_PROMPT),
                ]
            )
            generation_chain = prompt_template_generate | llm

            inputs = {
                "context": context_str,
                "topic_name": current_topic_name,
                "learning_objectives": learning_objectives,
            }

            new_question_to_ask = await call_llm_with_retry(generation_chain, inputs)
            logger.info("Новый вопрос успешно сгенерирован")
        else:
            error_message = "Недостаточно данных для Агента Оценки (нет контекста для генерации вопроса и нет ответа для оценки)."
            logger.warning(error_message)

    except Exception as e:
        logger.error(f"Ошибка в Агенте Оценки Знаний: {e}", exc_info=True)
        error_message = f"Внутренняя ошибка в Агенте Оценки: {str(e)}"

    # Формирование результата с очисткой обработанных данных
    update_data = {
        "assessment_question": new_question_to_ask,
        "assessment_feedback": feedback_on_answer,
        "student_answer": None,  # Очищаем после обработки
        "error_message": state.get("error_message") or error_message,
    }

    logger.debug(
        f"Агент Оценки Знаний завершен. Вопрос: {bool(new_question_to_ask)}, Фидбек: {bool(feedback_on_answer)}"
    )
    return update_data
