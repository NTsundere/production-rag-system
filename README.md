Production RAG System

Production-ready сервис Retrieval-Augmented Generation с гибридным поиском, cross-encoder reranking и агентным пайплайном на LangGraph. Работает полностью локально через Ollama, без внешних API и без VPN.
Возможности

    Гибридный поиск: BM25 (sparse) + плотные эмбеддинги с Reciprocal Rank Fusion

    Cross-encoder reranking: MS MARCO MiniLM-L6-v2 для точности

    Агентный пайплайн: LangGraph с состоянием rewrite -> retrieve -> rerank -> generate

    Векторное хранилище: Qdrant с косинусной близостью

    Локальные эмбеддинги: HuggingFace all-MiniLM-L6-v2 (384 измерения, без обращений к API)

    Локальная LLM: Ollama qwen2.5:7b (без API-ключей, без лимитов, без VPN)

    FastAPI со структурированным JSON-логированием

    Streamlit UI для интерактивного тестирования

    Docker и docker-compose

    CI/CD через GitHub Actions

    Метрики: Recall@k, Precision@k, MRR

Архитектура

Клиент отправляет POST-запрос на /query. FastAPI принимает его и передаёт в LangGraph-агента, который последовательно выполняет четыре шага: переписывание запроса, извлечение документов, реранжирование и генерацию ответа.

Извлечение работает через гибридный поиск. BM25 и плотные эмбеддинги объединяются через Reciprocal Rank Fusion, результаты складываются в Qdrant.

Быстрый старт
Требования

    Python 3.9 или выше

    Docker и Docker Compose

    Ollama (ollama.com/download)

    Модель qwen2.5:7b (4.7 GB)

1. Клонирование и установка
git clone https://github.com/NTsundere/production-rag-system.git
cd production-rag-system
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

2. Установка Ollama и модели
ollama pull qwen2.5:7b

3. Конфигурация
Copy-Item .env.example .env

По умолчанию в .env стоит:
OLLAMA_MODEL=qwen2.5:7b
OLLAMA_BASE_URL=http://localhost:11434/v1

4. Запуск Qdrant
docker compose up -d qdrant
Контейнер запустится на портах 6333 и 6334.

5. Индексация документов
Положи свои PDF, TXT или MD-файлы в папку data/documents. Затем выполни:
python -m scripts.ingest --file data/documents/твой_файл.md
Скрипт разобьёт документ на чанки, посчитает эмбеддинги локально и загрузит их в Qdrant.

6. Запуск API
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

7. Запуск Streamlit UI
В отдельном окне:
python -m streamlit run app/ui.py
Открой в браузере http://localhost:8501
8. Запрос

Отправь POST-запрос на http://localhost:8000/query с JSON-телом, содержащим поле question и, опционально, top_k. В ответ придёт JSON с полями answer, sources, latency_ms и model.
Streamlit UI

Четыре вкладки:

    Запрос — форма вопроса, метрики latency, финальный ответ, источники с раскрытием, скачивание отчёта

    Загрузка — загрузка PDF/TXT/MD, автоматическая индексация в Qdrant

    История — все запросы текущей сессии

    Архитектура — граф пайплайна, компоненты, endpoints API

    О проекте — обоснование архитектурных решений: почему hybrid search, почему Qdrant, почему cross-encoder, почему RRF, почему LangGraph, почему локальные embeddings, почему Ollama, почему Streamlit

В сайдбаре: статус API, коллекция Qdrant, слайдер top_k.
API
Health check
curl.exe http://localhost:8000/health
Ответ: {"status": "healthy", "collection": "rag_documents"}

Query
curl.exe -X POST http://localhost:8000/query -H "Content-Type: application/json" -d "{\"question\": \"What is the refund policy?\", \"top_k\": 5}"

Ingest
curl.exe -X POST "http://localhost:8000/ingest?file_path=data/documents/your_doc.md"

Тестирование
python -m pytest tests/ --cov=app --cov-report=term-missing
Тесты покрывают retrieval-метрики и работу API.

Оценка качества
Система поддерживает два режима.
Retrieval-метрики без LLM
python -m scripts.run_eval

Возвращает Recall@5, Precision@5 и MRR по golden set. Работает мгновенно, не требует обращений к LLM.

RAGAS-метрики

faithfulness, answer_relevancy, context_precision, context_recall. Требует LLM с большим контекстным окном. На Ollama qwen2.5:7b работает медленно из-за размера промптов.