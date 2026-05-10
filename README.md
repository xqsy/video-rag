# Video RAG

Интеллектуальный диалоговый ассистент для семантического поиска и анализа контекста в видеозаписях (RAG).

Репозиторий объединяет две параллельные курсовые работы:

- **КР-1 (инженерный прототип):** работающее приложение с веб-интерфейсом.
- **КР-2 (модель и алгоритмы):** формальная модель, алгоритмы и эксперименты.

## Требования

- **Python:** 3.11+
- **ffmpeg:** должен быть доступен в `PATH` (нужен для извлечения аудио)
- **ОС:** Windows, Linux или macOS
- **Аппаратные ресурсы:** CPU-only (int8 квантизация whisper), 16 GB RAM рекомендуется

## Установка

### 1. Клонирование репозитория

```bash
git clone <url-репозитория>
cd coursework_1
```

### 2. Создание виртуального окружения

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS:**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Установка зависимостей

Через `requirements.txt`:
```bash
pip install -r requirements.txt
```

Или через `pyproject.toml` (рекомендуется для разработки):
```bash
pip install -e ".[dev]"
```

### 4. Настройка окружения

Создайте файл `.env` в корне проекта:

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

> Без ключа LLM-этапы (ответы на вопросы, конспекты) работать не будут. Получить ключ можно на [openrouter.ai](https://openrouter.ai).

### 5. Проверка установки

```bash
python -m compileall video_rag
pytest -q
```

## Запуск

### Веб-интерфейс (основной способ)

```bash
video-rag-web
```

Или:

```bash
python -m video_rag.web
```

Откройте в браузере: [`http://127.0.0.1:8000`](http://127.0.0.1:8000)

Функционал:
- **Чат** — выбор видео, просмотр, конспект и диалог с ассистентом.
- **Библиотека** — загрузка, переиндексация, удаление и экспорт транскриптов.

### CLI: индексация видео

```bash
python -m video_rag.ingest path/to/video.mp4 --profile draft
```

Этапы:
1. Извлечение mono 16 kHz WAV через `ffmpeg`.
2. Транскрипция аудио через `faster-whisper`.
3. Кэширование JSON-транскрипта в `data/transcripts/`.
4. Построение time-window чанков для индексации.

## Структура проекта

```
coursework_1/
├── video_rag/          # Основной пакет
│   ├── ingest/         # Загрузка и обработка видео
│   ├── retrieval/      # Плотный поиск (dense retrieval)
│   ├── storage/        # ChromaDB + SQLite
│   ├── generation/     # LLM: промпты, ответы, конспекты
│   └── web/            # FastAPI + vanilla HTML/CSS/JS
├── tests/              # Unit-тесты
├── docs/               # Планы и пояснительные записки
├── data/               # Данные проекта (в .gitignore)
├── pyproject.toml      # Конфигурация проекта и зависимостей
├── requirements.txt    # Runtime-зависимости
└── README.md           # Этот файл
```

## Разработка

### Линтинг и форматирование

```bash
ruff check video_rag tests
black video_rag tests
```

### Запуск тестов

```bash
pytest -q
```

## Лицензия

Проект создан в рамках учебных курсовых работ.
