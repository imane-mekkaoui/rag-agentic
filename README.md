# Agentic RAG

Assistant RAG agentique : vous déposez un ou plusieurs PDF, puis vous posez des questions. Les réponses s’appuient **uniquement** sur ces fichiers (pas de base CV externe, pas de connaissances générales).

Interface principale : **Streamlit**. Agent LangGraph / LangChain, embeddings OpenAI, recherche vectorielle Chroma.

## Fonctionnement

1. Déposez un PDF dans la barre latérale.
2. Le texte est extrait, découpé et indexé.
3. L’agent cherche dans cet index (`pdf_search` ou contenu complet pour un résumé).
4. S’il n’y a pas l’information dans le PDF, il l’indique clairement.

## Prérequis

- Python 3.13+
- Une clé OpenAI (`OPENAI_API_KEY`)

## Installation

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install -e .
```

Créez un fichier `.env` à la racine (ne le commitez pas) :

```
OPENAI_API_KEY=votre_cle
```

## Lancer l’application

```bash
streamlit run streamlit_app.py
```

Ouvrez [http://localhost:8501](http://localhost:8501), déposez un PDF, puis posez une question (résumé, points clés, détail précis).

Interface FastAPI optionnelle :

```bash
python main.py
```

## Fichiers

```
├── streamlit_app.py   # Interface Streamlit (chat + dépôt PDF)
├── agentic_rag.py     # Agent, index Chroma, outils de recherche
├── server.py          # API FastAPI
├── static/            # Front statique de l’API
├── main.py            # Lancement uvicorn
├── langgraph.json     # Config LangGraph
└── pyproject.toml     # Dépendances
```

`.env` et `.venv` sont ignorés par Git.

## Dépendances principales

LangChain, LangGraph, Chroma, OpenAI, Streamlit, FastAPI, pypdf, python-dotenv.
