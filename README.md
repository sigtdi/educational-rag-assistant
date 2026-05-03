# 📚 Educational RAG Assistant

Учебный ассистент на основе RAG (Retrieval-Augmented Generation) для ответов на вопросы по алгоритмам. Система разработана для работы с русскоязычными учебниками, содержащими математические формулы и изображения, и помогает студентам разобраться в учебном материале.

---

## 🗂️ Содержание

- [Особенности](#особенности)
- [Архитектура](#архитектура)
- [Структура проекта](#структура-проекта)
- [Установка и запуск](#установка-и-запуск)

---

## ✨ Особенности

- **Специализированный парсинг** — обработка русскоязычных PDF-учебников с формулами и изображениями через [Marker](https://github.com/VikParuchuri/marker) + VLM-коррекция
- **Семантическое чанкирование** — извлечение структурных метаданных (определения, теоремы, рисунки, межчанковые ссылки, иерархия разделов)
- **Гибридный поиск** — векторный + ключевой поиск в Qdrant с реранкингом и parent retrieval
- **Агент на LangGraph** — цикл LLM ↔ инструмент поиска для многошагового рассуждения
- **Полностью локальный стек** — модели запускаются через Ollama, без внешних API

---

## 🏗️ Архитектура

Система состоит из двух независимых пайплайнов: **обработки и индексации документов** и **ответа на запросы**.

### 1. Пайплайн обработки документов (`document_processing`)

```
PDF-учебник
    │
    ▼
┌─────────────────────────────────────────────┐
│ ПАРСИНГ (parser/)                           │
│                                             │
│  Marker → первичное извлечение текста       │
│      │                                      │
│      ├─ Текст с формулами/латиницей?        │
│      │       └─ VLM-коррекция по скриншоту  │
│      │          чанка (image_processor)     │
│      │                                      │
│      └─ Изображения?                        │
│              └─ VLM-генерация описания      │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│ ЧАНКИРОВАНИЕ (chunking/)                    │
│                                             │
│  • Определение типа чанка                   │
│    (определение / теорема / рисунок / текст)│
│  • Обнаружение межчанковых ссылок           │
│  • Группировка для parent retrieval         │
│  • Построение пути: Глава > Подглава        │
│  • Запись меток и ссылок в метаданные       │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│ ИНДЕКСАЦИЯ (indexing/)                      │
│                                             │
│  • Генерация UUID                           │
│  • Нормализация полей                       │
│  • Векторизация (Ollama embeddings)         │
│  • Загрузка в Qdrant                        │
└─────────────────────────────────────────────┘
```

### 2. Пайплайн ответа на запросы (`rag/`)

```
Вопрос студента
    │
    ▼
┌──────────────────────────────────────────────┐
│ АГЕНТ (LangGraph)                            │
│                                              │
│  ┌─────────┐     ┌──────────────────────┐   │
│  │   LLM   │────▶│    search_tool       │   │
│  │(Ollama) │◀────│                      │   │
│  └─────────┘     │  1. Гибридный поиск  │   │
│       │          │     (вектор + BM25)  │   │
│       │          │  2. Реранкинг        │   │
│  Финальный       │  3. Parent retrieval │   │
│   ответ          │  4. Форматирование   │   │
│                  └──────────────────────┘   │
└──────────────────────────────────────────────┘
```

---

## 📁 Структура проекта

```
educational-rag-assistant/
├── app/
│   ├── api/                        # API-интерфейс
│   └── services/
│       ├── document_processing/    # Пайплайн обработки документов
│       │   ├── parser/             # Парсинг PDF: Marker + VLM-коррекция
│       │   │   ├── image_processor.py
│       │   │   ├── marker_processor.py
│       │   │   ├── text_processor.py
│       │   │   ├── parser_pipeline.py
│       │   │   └── parser_config.py
│       │   ├── chunking/           # Семантическое чанкирование
│       │   │   ├── chunker_processor.py
│       │   │   └── chunker_config.py
│       │   ├── indexing/           # Загрузка в Qdrant
│       │   │   ├── loader.py
│       │   │   ├── storage_pipeline.py
│       │   │   ├── storage_preparer.py
│       │   │   └── storage_config.py
│       │   ├── data/               # Исходные PDF-учебники
│       │   └── processing_pipeline.py
│       └── rag/                    # RAG-агент
│           ├── chains/
│           │   ├── agent.py        # LangGraph агент
│           │   └── agent_config.py
│           ├── retrieval/
│           │   ├── retriever.py    # Гибридный поиск + реранкинг
│           │   ├── formatting.py
│           │   └── retriever_config.py
│           └── tools/
│               └── search_tool.py  # Инструмент поиска для агента
├── tests/
├── config.yaml                     # Конфигурация подключений и моделей
├── logger_setup.py
└── docker-compose.yml              # Qdrant
```

---

## 🚀 Установка и запуск

### Предварительные требования

- Python 3.10+
- [Docker](https://www.docker.com/)
- [Ollama](https://ollama.com/) с установленными моделями

### 1. Клонирование репозитория

```bash
git clone https://github.com/your-username/educational-rag-assistant.git
cd educational-rag-assistant
```

### 2. Установка зависимостей

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Запуск Qdrant

```bash
docker-compose up -d
```

Qdrant будет доступен по адресу `http://localhost:6333`.

### 4. Конфигурация

Заполните `config.yaml` параметрами подключения и выбором моделей:

```yaml
  qdrant: &qdrant
    qdrant_host: "localhost"
    qdrant_port: 6333
    qdrant_url:  ""
    collection_name:     "chunks"
```

### 5. Индексация учебников

Поместите PDF-файлы в `app/services/document_processing/data/`, затем запустите пайплайн индексации:

```bash
python -m app.services.document_processing.indexing.storage_pipeline
```

> Этот шаг выполняется один раз (или при добавлении новых документов). Он парсирует PDF, обогащает чанки метаданными и загружает всё в Qdrant.

### 6. Запуск агента

```bash
python -m app.services.rag.chains.agent
```

После запуска можно задавать вопросы по теме алгоритмов — агент найдёт релевантные фрагменты из учебников и сформирует объяснение.

---

## 🛠️ Стек технологий

| Компонент | Технология |
|---|---|
| LLM / VLM / Embeddings | [Ollama](https://ollama.com/) |
| RAG-фреймворк | [LangChain](https://www.langchain.com/) |
| Агент | [LangGraph](https://github.com/langchain-ai/langgraph) |
| Векторная БД | [Qdrant](https://qdrant.tech/) |
| Парсинг PDF | [Marker](https://github.com/VikParuchuri/marker) |
| Язык | Python 3.10+ |
| Контейнеризация | Docker |