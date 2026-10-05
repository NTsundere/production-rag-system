import io
import time

import requests
import streamlit as st
import pandas as pd

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Production RAG System",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(90deg, #0ea5e9 0%, #6366f1 50%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.4rem;
    }
    .sub-header {
        color: #6b7280;
        font-size: 1.05rem;
        margin-bottom: 1.8rem;
    }
    .stButton>button {
        background: linear-gradient(90deg, #0ea5e9 0%, #6366f1 100%);
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.55rem 1.8rem;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #0284c7 0%, #4f46e5 100%);
        color: white;
    }
    .status-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .status-online { background: #d1fae5; color: #065f46; }
    .status-offline { background: #fee2e2; color: #991b1b; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">Production RAG System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    'Hybrid search, cross-encoder reranking, LangGraph agent. FastAPI + Qdrant + Ollama.'
    '</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Управление")

    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        health = r.json()
        st.markdown(
            '<span class="status-badge status-online">ONLINE</span>',
            unsafe_allow_html=True,
        )
        st.caption(f"Collection: {health.get('collection', 'n/a')}")
    except Exception:
        st.markdown(
            '<span class="status-badge status-offline">OFFLINE</span>',
            unsafe_allow_html=True,
        )
        st.error("API недоступен. Запусти:\npython -m uvicorn app.main:app --port 8000")

    st.markdown("---")
    st.subheader("Параметры поиска")
    top_k = st.slider("top_k", min_value=1, max_value=20, value=5, step=1)
    st.caption("Количество источников, попадающих в контекст после reranking.")

    st.markdown("---")
    st.subheader("О системе")
    st.markdown("""
    - Hybrid search: BM25 + dense
    - Fusion: Reciprocal Rank Fusion
    - Reranker: cross-encoder MiniLM
    - Agent: LangGraph (rewrite -> retrieve -> rerank -> generate)
    - Embeddings: HuggingFace MiniLM-L6-v2
    - Vector DB: Qdrant
    - LLM: Ollama qwen2.5:7b
    """)

if "history" not in st.session_state:
    st.session_state.history = []

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Запрос", "Загрузка", "История", "Архитектура", "О проекте"])

with tab1:
    st.subheader("Задай вопрос по документам")

    with st.form("query_form"):
        question = st.text_area(
            "Вопрос",
            placeholder="Например: What is the refund policy?",
            height=90,
        )
        submitted = st.form_submit_button("Найти ответ")

    if submitted:
        if not question.strip():
            st.warning("Введи вопрос.")
        else:
            with st.spinner("Обработка запроса..."):
                t0 = time.time()
                try:
                    response = requests.post(
                        f"{API_URL}/query",
                        json={"question": question, "top_k": top_k},
                        timeout=600,
                    )
                    response.raise_for_status()
                    result = response.json()
                    elapsed = time.time() - t0

                    st.success(f"Ответ получен за {elapsed:.1f} секунд")

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Latency (API)", f"{result.get('latency_ms', 0):.0f} ms")
                    c2.metric("Total time", f"{elapsed:.1f} s")
                    c3.metric("Источников", len(result.get("sources", [])))
                    c4.metric("Модель", result.get("model", "-"))

                    st.markdown("---")
                    st.markdown("### Ответ")
                    with st.container(border=True):
                        st.markdown(result.get("answer", "-"))

                    sources = result.get("sources", [])
                    if sources:
                        st.markdown("### Источники")
                        for i, src in enumerate(sources, 1):
                            with st.expander(
                                f"{i}. {src.get('source', 'unknown')} "
                                f"(score: {src.get('score', 0):.3f})"
                            ):
                                st.markdown(src.get("content", ""))

                    st.session_state.history.append({
                        "question": question,
                        "answer": result.get("answer", ""),
                        "sources": sources,
                        "latency_ms": result.get("latency_ms", 0),
                        "time": time.strftime("%H:%M:%S"),
                    })

                except Exception as e:
                    st.error(f"Ошибка: {e}")

with tab2:
    st.subheader("Загрузить документ в базу знаний")
    st.caption("Поддерживаются форматы PDF, TXT, MD. Документ будет разбит на чанки и проиндексирован в Qdrant.")

    uploaded = st.file_uploader(
        "Выбери файл",
        type=["pdf", "txt", "md"],
        key="ingest_upload",
    )

    if uploaded is not None:
        st.info(f"Файл: {uploaded.name} ({uploaded.size} байт)")

        if st.button("Индексировать"):
            import os
            import tempfile

            suffix = os.path.splitext(uploaded.name)[1]
            tmp_path = os.path.join(tempfile.gettempdir(), f"upload{suffix}")
            with open(tmp_path, "wb") as f:
                f.write(uploaded.read())

            with st.spinner("Индексация..."):
                try:
                    response = requests.post(
                        f"{API_URL}/ingest",
                        params={"file_path": tmp_path},
                        timeout=600,
                    )
                    response.raise_for_status()
                    result = response.json()
                    st.success(
                        f"Готово. Чанков создано: {result.get('chunks_created', 0)}. "
                        f"Коллекция: {result.get('collection', '-')}"
                    )
                except Exception as e:
                    st.error(f"Ошибка индексации: {e}")

with tab3:
    st.subheader("История запросов")

    if not st.session_state.history:
        st.info("Пока пусто. Запусти первый запрос во вкладке Запрос.")
    else:
        for i, item in enumerate(reversed(st.session_state.history), 1):
            with st.expander(
                f"{i}. {item['question'][:80]}... "
                f"({item['time']}, {item['latency_ms']:.0f} ms)"
            ):
                st.markdown("**Ответ:**")
                st.markdown(item["answer"])
                st.markdown("---")
                st.caption(f"Источников использовано: {len(item['sources'])}")
                for j, src in enumerate(item["sources"], 1):
                    st.markdown(
                        f"**{j}.** {src.get('source', 'unknown')} "
                        f"(score: {src.get('score', 0):.3f})"
                    )
                    st.text(src.get("content", "")[:400] + "...")

        if st.button("Очистить историю"):
            st.session_state.history = []
            st.rerun()

with tab4:
    st.subheader("Архитектура RAG пайплайна")

    st.markdown("### Граф обработки запроса")
    st.code(
        """
        START
          |
          v
        REWRITE  (LLM переписывает вопрос для лучшего поиска)
          |
          v
        RETRIEVE  (hybrid search: BM25 + dense + RRF)
          |
          v
        DECIDE  (есть ли документы?)
          |
       +--+--+
       |     |
       v     v
    REWRITE  RERANK  (cross-encoder переупорядочивает)
              |
              v
           GENERATE  (LLM отвечает на основе контекста)
              |
              v
             END
        """,
        language="text",
    )

    st.markdown("### Hybrid search")
    st.markdown("""
Параллельно выполняются два поиска:

- BM25 — лексический поиск по точным совпадениям слов
- Dense — семантический поиск по векторным эмбеддингам

Результаты объединяются через Reciprocal Rank Fusion:

score(d) = sum_i  weight_i / (k + rank_i(d))

где k = 60, weight_dense = 0.7, weight_bm25 = 0.3.
    """)

    st.markdown("### Компоненты")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        **Ingestion:**

        - Загрузка PDF, TXT, MD
        - Recursive chunking (800 символов, overlap 150)
        - Локальные эмбеддинги MiniLM-L6-v2
        - Upsert в Qdrant с payload
        """)
    with c2:
        st.markdown("""
        **Retrieval:**

        - Dense: Qdrant cosine similarity
        - Sparse: BM25Okapi over corpus
        - Fusion: RRF
        - Reranking: cross-encoder
        """)

    st.markdown("### Endpoints API")
    st.markdown("""
| Метод | Путь | Описание |
|-------|------|----------|
| GET | /health | Проверка статуса и коллекции |
| POST | /query | RAG-запрос с гибридным поиском |
| POST | /ingest | Индексация документа |
""")

    st.markdown("**Пример запроса:**")
    st.code(
        'curl -X POST http://localhost:8000/query '
        '-H "Content-Type: application/json" '
        '-d \'{"question": "What is the refund policy?", "top_k": 5}\'',
        language="bash",
    )

with tab5:
    st.subheader("О проекте")
    st.caption("Обоснование архитектурных решений.")

    with st.expander("Почему hybrid search, а не только dense"):
        st.markdown("""
Dense retrieval (только векторы) хорошо работает на семантически близких запросах, но плохо
находит точные совпадения: аббревиатуры, артикулы, имена собственные. Если в документе написано
"SKU-12345", а пользователь вводит "SKU-12345", dense-модель может не найти этот чанк, потому
что обучалась на естественном языке.

BM25, наоборот, точно находит лексические совпадения, но не понимает синонимы. Если пользователь
спросит "цена", а в документе "стоимость", BM25 не найдёт.

Гибридный поиск объединяет оба подхода через Reciprocal Rank Fusion. RRF не требует нормализации
скоров (они у BM25 и dense в разных шкалах), а просто учитывает ранги: если документ попал в
топ-10 у обоих поисковиков, он поднимется выше.

Аналоги: только BM25 (Elasticsearch), только dense (Pinecone, Weaviate), ColBERT (late
interaction). Выбрана комбинация BM25 + dense, потому что она даёт лучший recall на смешанных
запросах без обучения дополнительной модели.
""")

    with st.expander("Почему Qdrant, а не FAISS или pgvector"):
        st.markdown("""
FAISS — библиотека от Facebook, работает в памяти, без сетевого сервера. Быстрая, но:
нет persistence из коробки, нет REST API, нет фильтрации по метаданным без кастомного кода.

pgvector — расширение PostgreSQL. Хорошо, если уже есть Postgres. Но требует настройки
БД, индексов (IVFFlat, HNSW), отдельного администратора.

Qdrant — отдельный сервис, но:
REST и gRPC API из коробки, persistence через снапшоты, фильтрация по payload (метаданным)
как в полноценной БД, HNSW индекс без настройки, стабильный Python-клиент.

Компромисс: Qdrant требует запуска отдельного контейнера. Для проекта это оправдано:
Docker Compose запускает Qdrant одной командой, а API получает production-grade хранилище.
""")

    with st.expander("Почему cross-encoder reranking"):
        st.markdown("""
Гибридный поиск даёт топ-20 кандидатов, но порядок внутри топ-20 не идеален. Cross-encoder
(MiniLM-L6-v2) читает пару (запрос, документ) целиком и выдаёт оценку релевантности от 0 до 1.
Это точнее, чем dot-product эмбеддингов, потому что модель видит оба текста одновременно.

Почему не bi-encoder: bi-encoder кодирует запрос и документ раздельно, что быстрее (можно
предвычислить эмбеддинги документов), но менее точно. Cross-encoder в 10-100 раз медленнее, но
мы применяем его только к топ-20, а не ко всей базе.

Компромисс: +50-200 ms к latency запроса, зато +10-20% к точности ответа. Для RAG это
оправдано: качество ответа важнее скорости.

Аналоги: Cohere Rerank (платный API), BGE-reranker (сильнее, но тяжелее), monoT5.
Выбрана MiniLM-L6-v2, потому что она быстрая (22M параметров) и работает на CPU.
""")

    with st.expander("Почему Reciprocal Rank Fusion, а не взвешенная сумма"):
        st.markdown("""
Прямое сложение скоров из BM25 и dense не работает: BM25 возвращает значения от 0 до десятков,
dense — от 0 до 1. Без нормализации один сигнал подавит другой.

Варианты решения:
1. Нормализовать оба скора в [0, 1] через min-max или z-score.
2. Использовать Reciprocal Rank Fusion.

RRF выбран потому что:
не требует нормализации, устойчив к выбросам (один документ с огромным BM25-скором не
подавит остальные), параметр k=60 подтверждён эмпирически в литературе.

Формула: score(d) = sum(weight_i / (60 + rank_i(d)))

Аналоги: CombSUM, CombMNZ, borda count. RRF стандарт в современных поисковых системах.
""")

    with st.expander("Почему LangGraph, а не простой pipeline"):
        st.markdown("""
RAG можно реализовать как цепочку функций:
rewrite(query) -> retrieve(query) -> rerank(docs) -> generate(context, query)

LangGraph даёт:
состояние (TypedDict), которое передаётся между узлами, условные переходы (если документов
нет, переписать запрос), точки расширения для grading, self-reflection, multi-hop,
визуализацию графа, checkpointer для сохранения состояния.

Компромисс: LangGraph добавляет зависимость и небольшой оверхед. Для простого RAG это
избыточно. Но структура готова к расширению: добавление grading или multi-hop — это один
новый узел, а не рефакторинг всего кода.
""")

    with st.expander("Почему локальные embeddings (HuggingFace), а не OpenAI"):
        st.markdown("""
OpenAI embeddings (text-embedding-3-small):
1536 измерений, отличное качество, требует API-ключ и оплату, недоступны из Беларуси без
VPN, лимит 8191 токен на чанк.

HuggingFace MiniLM-L6-v2:
384 измерения, хорошее качество, работает локально на CPU, 90 MB скачивается один раз,
не требует интернета после скачивания, не требует API-ключа.

Компромисс: MiniLM уступает OpenAI по качеству на сложных семантических задачах
(на 5-10% по MTEB benchmark). Но для большинства RAG-задач разница не критична, а
независимость от API и региона важнее. Замена — три строки в app/ingestion/indexers.py.
""")

    with st.expander("Почему Ollama, а не Gemini, Groq, OpenAI или OpenRouter"):
        st.markdown("""
Проект изначально разрабатывался с прицелом на облачные LLM-провайдеры. В процессе работы
были протестированы Gemini API, Groq API и OpenRouter. Каждый из них упёрся в ограничения,
которые делают их непригодными для воспроизводимого проекта.

Gemini API. Бесплатный тариф даёт 20 запросов в день на модель. RAG-запрос тратит два вызова
(rewrite и generate), RAGAS eval — ещё десять. Дневной лимит исчерпывается за два запроса.
Дополнительно модели Gemini 3.x требуют передачи thought_signature при tool calling, а
текущая версия langchain-google-genai для Python 3.9 не умеет это делать. При работе из
Беларуси Gemini возвращает ошибку "User location is not supported" без VPN, и часто
нестабилен даже с VPN.

Groq API. Ключи Groq требуют VPN и актуальных моделей. Устаревшие модели отключают для
бесплатных аккаунтов. При тестировании был получен 403 Forbidden на уровне Cloudflare.

OpenAI и Anthropic. Требуют привязки банковской карты и оплаты за токены. Не подходят
для проекта, который должен запускаться у любого пользователя без затрат.

OpenRouter. Агрегатор, работающий через VPN. Бесплатные модели имеют лимиты по количеству
запросов в минуту, что несовместимо с многошаговыми агентами.

Ollama. Локальный инференс. Работает без интернета после загрузки модели, без API-ключей,
без лимитов и без региональных блокировок. Единственный минус — скорость. На CPU модель
qwen2.5:7b обрабатывает запрос от 30 секунд до 3 минут в зависимости от длины контекста.

Итог: Ollama — единственный провайдер, который гарантированно работает в любом регионе,
не требует VPN, карт и ключей, и не имеет суточных лимитов. Для production-сценариев можно
заменить одну строку в app/llm.py на ChatOpenAI или ChatGoogleGenerativeAI.
""")

    with st.expander("Почему Streamlit, а не Gradio или FastAPI + HTML"):
        st.markdown("""
Gradio: оптимален для демонстрации ML-моделей с одной формой ввода-вывода. Для
многостраничного интерфейса с загрузкой файлов, историей, метриками — неудобен.

FastAPI + HTML/CSS/JS: полный контроль, но требует верстки, шаблонизатора, JS. Для
внутреннего инструмента это избыточно.

Streamlit: 200 строк Python дают полноценный веб-интерфейс с табами, формами, метриками,
таблицами, загрузкой файлов. Не требует знания фронтенда.

Компромисс: Streamlit не подходит для production UI (нет аутентификации, кастомизации,
роутинга). Но для внутреннего инструмента, демонстрации и разработки — идеален.
""")

    with st.expander("Ограничения и известные проблемы"):
        st.markdown("""
Скорость Ollama. Локальный инференс qwen2.5:7b на CPU обрабатывает запрос от 30 секунд
до 3 минут. Для RAG это значит, что rewrite и generate — два вызова, каждый занимает
время. Ускорение: GPU (Ollama автоматически подхватит CUDA/ROCm) или уменьшение модели
до qwen2.5:3b.

Локальные эмбеддинги на CPU. Первый запрос к get_embedder() скачивает модель (90 MB) и
загружает её в память (30-60 секунд). Последующие запросы быстрые. На GPU можно ускорить
установив onnxruntime-gpu и передав model_kwargs={"device": "cuda"}.

SqliteSaver и thread_id. История диалога хранится по thread_id, но результаты retrieval
не сбрасываются между запросами. В текущей версии каждый запрос — независимая операция.

Нет streaming. Ответ возвращается целиком после генерации. Для длинных ответов это
создаёт паузу. Streaming через SSE возможен в будущих версиях.

Нет мониторинга. Метрики latency пишутся в JSON-лог, но не агрегируются. Для production
нужен Prometheus, Grafana или Langfuse.
""")

    with st.expander("Что можно улучшить"):
        st.markdown("""
- Добавить streaming ответа через Server-Sent Events.
- Добавить Langfuse для трейсинга каждого шага RAG.
- Добавить RAGAS eval с golden set и LLM-as-a-judge.
- Добавить кэширование эмбеддингов через Redis.
- Добавить multi-tenancy через namespaces в Qdrant.
- Добавить query expansion: генерация нескольких переформулировок запроса.
- Добавить HyDE: генерация гипотетического ответа для улучшения retrieval.
- Добавить Self-RAG: модель сама решает, нужен ли дополнительный retrieval.
- Поддержка нескольких LLM-провайдеров через абстракцию app/llm.py.
""")

st.markdown("---")
st.caption(
    "Production RAG System. MIT. "
    "github.com/NTsundere/production-rag-system"
)