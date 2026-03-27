# SciWriter MAS

Multi-agent system for scientific article generation using CrewAI + LangGraph.

## Architecture

4 specialized agents orchestrated by a LangGraph StateGraph with conditional review loop:

```
[Topic] → Research → Write → Fact-Check → Review Gate → Edit → Publish
                       ↑                       │
                       └── revise (max 2x) ────┘
```

### Agents (CrewAI)

| Agent | Role | Tools |
|-------|------|-------|
| **Researcher** | Finds sources via web search + RAG | DuckDuckGo, Qdrant hybrid search |
| **Writer** | Generates article draft from sources | Text analysis |
| **Fact-Checker** | Verifies claims against sources | DuckDuckGo, Qdrant hybrid search |
| **Editor** | Polishes style, structure, SEO | Text analysis |

### Orchestration (LangGraph)

- **Conditional review loop**: if fact-check score < 7/10, article goes back for revision (max 2 iterations)
- **Human-in-the-loop**: editor can intervene at review gate via Streamlit UI
- **State checkpointing**: full audit trail of every agent decision

### Format Presets (YAML)

| Preset | Platform | Style | Length |
|--------|----------|-------|--------|
| `habr` | Habr | Technical, code examples | 8-15K chars |
| `dzen` | Yandex Dzen | Popular science, simple language | 5-8K chars |

## Stack

- **Agents**: CrewAI 1.10+
- **Orchestration**: LangGraph 0.3+
- **LLM**: Ollama (Qwen3 8B) — local, free
- **RAG**: Qdrant + multilingual-e5-large + BM25 (hybrid search with RRF)
- **Web Search**: DuckDuckGo (no API key needed)
- **API**: FastAPI
- **UI**: Streamlit
- **Containerization**: Docker Compose

## Quick Start

### 1. Start infrastructure

```bash
docker compose up -d qdrant
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure

```bash
cp .env.example .env
# Edit .env — set LLM_BASE_URL to your Ollama instance
```

### 4. Index knowledge base

```bash
python -m app.main index data/knowledge_base/
```

### 5. Generate an article

```bash
# CLI
python -m app.main generate "Квантовые компьютеры: прорыв 2026" --preset habr -o article.md

# API
python -m app.main serve
# POST http://localhost:8000/generate {"topic": "...", "preset": "habr"}

# Streamlit UI
streamlit run ui/app.py
```

## Project Structure

```
sciwriter-mas/
├── app/
│   ├── agents/          # CrewAI agent definitions
│   ├── tools/           # Agent tools (web search, RAG, text analysis)
│   ├── graph/           # LangGraph workflow (state, nodes, edges)
│   ├── presets/         # YAML format presets (habr, dzen)
│   ├── rag/             # RAG infrastructure (embedder, retriever, indexer)
│   ├── api/             # FastAPI routes
│   ├── config.py        # Pydantic settings
│   └── main.py          # Entry point (CLI + FastAPI)
├── ui/
│   └── app.py           # Streamlit dashboard
├── data/
│   └── knowledge_base/  # Documents for RAG indexing
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

## How It Works

1. **User provides a topic** and selects a format (Habr/Dzen)
2. **Researcher** searches the web and local knowledge base for sources
3. **Writer** generates an article draft based on the sources and preset format
4. **Fact-Checker** verifies every claim against sources, assigns accuracy score
5. **Review Gate** decides: accept (score ≥ 7) or send back for revision
6. **Editor** polishes the article: grammar, style, SEO, structure
7. **Final article** is saved with full audit trail

The review loop ensures factual accuracy — the article can be revised up to 2 times before final editing.
