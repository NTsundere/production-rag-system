Production RAG System

Production-ready сервис Retrieval-Augmented Generation с гибридным поиском, cross-encoder reranking и агентным пайплайном на LangGraph.
Возможности

    Гибридный поиск: BM25 (sparse) + плотные эмбеддинги с Reciprocal Rank Fusion

    Cross-encoder reranking: MS MARCO MiniLM-L6-v2 для точности

    Агентный пайплайн: LangGraph с состоянием rewrite → retrieve → rerank → generate

    Векторное хранилище: Qdrant с косинусной близостью

    Локальные эмбеддинги: HuggingFace all-MiniLM-L6-v2 (384 измерения, без обращений к API)

    LLM: Google Gemini 3 Flash (через REST transport)

    API: FastAPI со структурированным JSON-логированием

    Контейнеризация: Docker + docker-compose

    CI/CD: GitHub Actions (тесты + сборка Docker-образа)

    Наблюдаемость: JSON-логи, замер latency на каждый запрос

Архитектура

Клиент отправляет POST-запрос на /query. FastAPI принимает его и передаёт в LangGraph-агента, который последовательно выполняет четыре шага: переписывание запроса, извлечение документов, реранжирование и генерацию ответа. Извлечение работает через гибридный поиск — BM25 и плотные эмбеддинги объединяются через Reciprocal Rank Fusion, результаты складываются в Qdrant.
Быстрый старт
Требования

    Python 3.10+ (проект протестирован на 3.11)

    Docker + Docker Compose

    API-ключ Google Gemini (получить можно на aistudio.google.com/apikey)

1. Клонирование и установка

Склонируй репозиторий и перейди в папку проекта. Создай виртуальное окружение и активируй его. Установи зависимости из requirements.txt.
2. Конфигурация

Скопируй .env.example в .env и укажи в нём свой GOOGLE_API_KEY и модель GEMINI_MODEL (например, gemini-3-flash-preview).
3. Запуск Qdrant

Выполни docker compose up -d qdrant. Контейнер запустится на портах 6333 и 6334.
4. Индексация документов

Положи свои PDF, TXT или MD-файлы в папку data/documents. Затем выполни python -m scripts.ingest --file data/documents/твой_файл.md. Скрипт разобьёт документ на чанки, посчитает эмбеддинги локально и загрузит их в Qdrant.
5. Запуск API

Выполни uvicorn app.main:app --host 0.0.0.0 --port 8000. Сервер запустится на порту 8000.
6. Запрос

Отправь POST-запрос на http://localhost:8000/query с JSON-телом, содержащим поле question и, опционально, top_k. В ответ придёт JSON с полями answer, sources (список источников с контентом, именем файла и скором), latency_ms и model.
Структура проекта

Папка app содержит основной код: main.py — эндпоинты FastAPI, config.py — настройки через Pydantic, llm.py — фабрика Gemini, schemas.py — Pydantic-схемы, logger.py — JSON-логирование. В подпапке ingestion находятся загрузчики документов, чанкеры и индексаторы. В retrieval — гибридный поиск и реранкер. В graph — состояние, узлы и сборка LangGraph-графа. В evaluation — метрики и golden set для оценки.

Папка scripts содержит CLI-скрипты ingest.py и run_eval.py. Папка tests — pytest-тесты. В корне лежат docker-compose.yml, Dockerfile, Makefile, requirements.txt и README.md.
Тестирование

Запусти pytest tests/ --cov=app --cov-report=term-missing. Тесты покрывают retrieval-метрики и работу API.
Оценка качества

Система поддерживает два режима оценки.

Первый — retrieval-метрики без LLM. Выполни python -m scripts.run_eval, и получишь Recall@5, Precision@5 и MRR по golden set. Это работает мгновенно и не требует обращений к Gemini.

Второй — RAGAS-метрики (faithfulness, answer_relevancy, context_precision, context_recall). Требует квоты Gemini, поэтому на бесплатном тарифе может упереться в лимит 20 запросов в день.
Технологический стек

API — FastAPI и Uvicorn. Оркестрация агента — LangGraph. Векторное хранилище — Qdrant. Эмбеддинги — sentence-transformers all-MiniLM-L6-v2. Реранкер — cross-encoder ms-marco-MiniLM-L-6-v2. Sparse retrieval — BM25 через rank-bm25. LLM — Google Gemini 3 Flash. Логирование — структурированный JSON. Контейнеризация — Docker и docker-compose. CI/CD — GitHub Actions.
Известные ограничения

Латентность LLM составляет 90–130 секунд на запрос при использовании бесплатного Gemini через VPN. Лимит бесплатного тарифа Gemini — 20 запросов в день, для продакшена нужен платный тариф или локальная модель через Ollama.