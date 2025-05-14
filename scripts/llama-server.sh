#!/bin/bash

# ============================================================
# Скрипт для запуска llama-server с оптимизированными настройками
# и корректным завершением
# ============================================================

echo "Настройка параметров запуска llama-server..."

# --- Переменные конфигурации из вашего скрипта ---
SERVER_BIN="/home/user/llama.cpp/build/bin/./llama-server"
MODEL_PATH="../../../.cache/huggingface/hub/models--unsloth--Qwen3-30B-A3B-GGUF/snapshots/7db7f59503edb0c41e94e7c38f9d7e6717ded423/Qwen3-30B-A3B-Q4_K_M.gguf"
MODEL_ALIAS="qwen3-30-A3"
HOST_IP="0.0.0.0"
PORT_NUM="30000"
CONTEXT_SIZE=10000
GPU_LAYERS=999
PARALLEL_REQS=8
LOGICAL_BATCH_SIZE=2048
PHYSICAL_BATCH_SIZE=512
GENERATION_THREADS=$(nproc --all 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
BATCH_THREADS=$(nproc --all 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
HTTP_THREADS=-1 # Автоматический выбор
KV_K_TYPE="f16"
KV_V_TYPE="f16"
DEFAULT_TEMP=0.7 # Ваши параметры (соответствуют Non-Thinking Mode)
DEFAULT_TOP_P=0.8
DEFAULT_TOP_K=20
DEFAULT_MIN_P=0.0
ENABLE_JINJA=true
REASONING_FORMAT="deepseek"
# MOE_OFFLOAD_ARGS="-ot .ffn_.*_exps.=CPU" # Оставляем закомментированным, как у вас
MOE_OFFLOAD_ARGS=""
ENABLE_FLASH_ATTN=true
ENABLE_MLOCK=true
ENABLE_METRICS=true
# ---------------------------------------------------

# Переменная для хранения PID сервера
SERVER_PID=""

# --- Функция очистки при завершении ---
cleanup() {
    echo # Новая строка для читаемости
    echo ">>> Получен сигнал завершения, попытка штатной остановки llama-server..."
    # Проверяем, есть ли у нас PID процесса и жив ли он
    if [ -n "$SERVER_PID" ] && ps -p "$SERVER_PID" > /dev/null; then
        echo ">>> Отправка сигнала SIGTERM процессу llama-server (PID: $SERVER_PID)..."
        # Отправляем сигнал TERM для штатного завершения
        kill -TERM "$SERVER_PID"

        # Ждем некоторое время (например, 10 секунд), чтобы процесс завершился сам
        echo ">>> Ожидание до 10 секунд для штатного завершения..."
        for i in {1..10}; do
            if ! ps -p "$SERVER_PID" > /dev/null ; then
                echo ">>> Процесс llama-server (PID: $SERVER_PID) успешно завершен."
                SERVER_PID="" # Сбрасываем PID
                break # Выходим из цикла ожидания
            fi
            echo -n "." # Индикатор ожидания
            sleep 1
        done

        # Если процесс все еще жив после ожидания, принудительно убиваем его
        if [ -n "$SERVER_PID" ] && ps -p "$SERVER_PID" > /dev/null; then
            echo # Новая строка
            echo ">>> Процесс llama-server (PID: $SERVER_PID) не завершился штатно. Отправка SIGKILL..."
            kill -KILL "$SERVER_PID"
            sleep 1 # Даем системе время
             if ! ps -p "$SERVER_PID" > /dev/null ; then
                 echo ">>> Процесс (PID: $SERVER_PID) принудительно остановлен."
                 SERVER_PID=""
             else
                 echo ">>> ВНИМАНИЕ: Не удалось остановить процесс (PID: $SERVER_PID) даже с SIGKILL."
             fi
        fi
    else
        echo ">>> Процесс llama-server не найден или уже остановлен."
    fi
    echo ">>> Очистка завершена."
}

# --- Установка ловушки ---
# Вызывать функцию cleanup при получении сигналов SIGINT (Ctrl+C), SIGTERM или при выходе из скрипта (EXIT)
trap cleanup SIGINT SIGTERM EXIT

# ============================================================
# Сборка и запуск команды
# ============================================================
COMMAND_ARGS=()

# Основные параметры
COMMAND_ARGS+=("-m" "$MODEL_PATH")
COMMAND_ARGS+=("--alias" "$MODEL_ALIAS")
COMMAND_ARGS+=("-c" "$CONTEXT_SIZE")
COMMAND_ARGS+=("-ngl" "$GPU_LAYERS")
COMMAND_ARGS+=("-np" "$PARALLEL_REQS")
COMMAND_ARGS+=("-b" "$LOGICAL_BATCH_SIZE")
COMMAND_ARGS+=("-ub" "$PHYSICAL_BATCH_SIZE")
COMMAND_ARGS+=("-t" "$GENERATION_THREADS")
COMMAND_ARGS+=("-tb" "$BATCH_THREADS")
COMMAND_ARGS+=("--threads-http" "$HTTP_THREADS")
COMMAND_ARGS+=("-ctk" "$KV_K_TYPE")
COMMAND_ARGS+=("-ctv" "$KV_V_TYPE")
COMMAND_ARGS+=("--host" "$HOST_IP")
COMMAND_ARGS+=("--port" "$PORT_NUM")
COMMAND_ARGS+=("--temp" "$DEFAULT_TEMP")
COMMAND_ARGS+=("--top-p" "$DEFAULT_TOP_P")
COMMAND_ARGS+=("--top-k" "$DEFAULT_TOP_K")
COMMAND_ARGS+=("--min-p" "$DEFAULT_MIN_P")

# Флаги
if [ "$ENABLE_FLASH_ATTN" = true ]; then COMMAND_ARGS+=("-fa"); fi
if [ "$ENABLE_MLOCK" = true ]; then COMMAND_ARGS+=("--mlock"); fi
if [ "$ENABLE_METRICS" = true ]; then COMMAND_ARGS+=("--metrics"); fi
if [ "$ENABLE_JINJA" = true ]; then
    COMMAND_ARGS+=("--jinja")
    COMMAND_ARGS+=("--reasoning-format" "$REASONING_FORMAT")
fi

# MoE Offloading (если заданы аргументы)
if [ -n "$MOE_OFFLOAD_ARGS" ]; then
    echo "INFO: Включение выгрузки MoE слоев..."
    COMMAND_ARGS+=($MOE_OFFLOAD_ARGS) # Добавляем как отдельные аргументы
fi

# Добавление любых дополнительных аргументов, переданных самому скрипту
COMMAND_ARGS+=("$@")

# --- Запуск ---
echo "-------------------------------------------"
echo "Запуск llama-server в фоновом режиме..."
echo "Исполняемый файл: $SERVER_BIN"
echo "Параметры:"
printf " %q" "${COMMAND_ARGS[@]}" # Печатает аргументы безопасно
echo # Новая строка
echo "-------------------------------------------"

# Запуск сервера в ФОНОВОМ РЕЖИМЕ (& в конце)
"$SERVER_BIN" "${COMMAND_ARGS[@]}" &

# Получение PID фонового процесса
SERVER_PID=$!
echo ">>> llama-server запущен с PID: $SERVER_PID"
echo ">>> Нажмите Ctrl+C для остановки сервера."

# Ожидание завершения фонового процесса llama-server
# Команда wait вернет код завершения процесса или >128, если он убит сигналом
wait "$SERVER_PID"
EXIT_CODE=$?
echo # Новая строка
echo ">>> Команда 'wait $SERVER_PID' завершилась."

# Важно: Ловушка EXIT вызовет cleanup здесь автоматически.
# Мы просто выходим с кодом завершения серверного процесса.
exit $EXIT_CODE