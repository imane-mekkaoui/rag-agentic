# Agentic RAG - Retrieval-Augmented Generation with Multi-Agent System

A sophisticated multi-agent AI system powered by LangGraph, LangChain, and OpenAI that implements Retrieval-Augmented Generation (RAG) with agent capabilities.

## 🎯 Project Overview

This project demonstrates an intelligent agent system that combines:
- **Retrieval-Augmented Generation (RAG)**: Retrieve relevant information from a vector store before generating responses
- **Multi-Agent Architecture**: Using LangGraph for orchestrating complex agent workflows
- **Tool Integration**: Custom tools for accessing employee information and CV data
- **Vector Search**: Chroma for efficient semantic search over document chunks

## 🏗️ Project Structure

```
agentic_rag/
├── main.py                 # Entry point for the application
├── agentic_rag.py          # Core RAG and agent implementation
├── langgraph.json          # LangGraph workflow configuration
├── pyproject.toml          # Project dependencies and metadata
├── .env                    # Environment variables (OpenAI API key, etc.)
└── README.md              # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.13 or higher
- OpenAI API key
- Virtual environment (recommended)

### Installation

1. **Clone or navigate to the project directory**

2. **Create and activate virtual environment**
   ```bash
   python -m venv .venv
   # On Windows:
   .\.venv\Scripts\Activate.ps1
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -e .
   ```

4. **Configure environment variables**
   
   Create or update `.env` file with:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

### Running the Application

```bash
python main.py
```

## 📚 Key Components

### agentic_rag.py

**Vector Store & RAG Setup:**
- Initializes a Chroma vector store with CV information about Mohamed Youssfi
- Uses OpenAI embeddings for semantic search
- Creates a retriever tool for accessing CV data

**Available Tools:**
- `cv_tool`: Retrieves information from the knowledge base about Mohamed's CV
- `get_employee_info`: Fetches employee information (name, salary, seniority)

### main.py

Entry point that prints a welcome message. Extend this to initialize your agent workflow.

### langgraph.json

Configuration file for the LangGraph workflow orchestration.

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| langchain | >=1.2.15 | Core framework for building AI applications |
| langchain-core | >=1.2.26 | Core abstractions |
| langchain-community | >=0.4.1 | Community integrations |
| langchain-openai | >=1.1.12 | OpenAI integration |
| langgraph | >=1.1.6 | Multi-agent workflow orchestration |
| chromadb | >=1.5.5 | Vector database for semantic search |
| python-dotenv | >=1.2.2 | Environment variable management |
| ipython | >=9.12.0 | Interactive computing |

## 🔧 Usage Examples

### Initialize the RAG System

```python
from agentic_rag import vectorstore, retriever_tool, get_employee_info

# Query the CV information
results = retriever_tool.invoke("Tell me about Mohamed's background")
```

### Create an Agent

```python
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4")
tools = [retriever_tool, get_employee_info]

agent = create_agent(llm, tools)
```

## 🔐 Environment Configuration

The project uses `python-dotenv` to manage configuration:

```python
from dotenv import load_dotenv
load_dotenv(override=True)
```

Ensure your `.env` file contains:
- `OPENAI_API_KEY`: Your OpenAI API key

## 📖 Knowledge Base

The system is pre-loaded with information about Mohamed Youssfi, a professor at ENSET Mohammedia, including:
- Educational background and credentials
- Professional experience
- Research interests
- Personal interests (music, culture, philosophy, writing)
- Career timeline

## 🎓 Academic Context

**Institution:** ENSET Mohammedia, Université Hassan II de Casablanca

**Professor:** Mohamed Youssfi
- **PhD (Doctorate d'État):** 2015
- **Doctorate (3rd Cycle):** 1996  
- **Teaching Diploma:** 1993
- **Specialization:** Parallel and Distributed Computing Systems

## 🵻 Next Steps

- [ ] Implement the main agent workflow
- [ ] Add more tools and capabilities
- [ ] Integrate with LangGraph for complex multi-step reasoning
- [ ] Add conversation memory management
- [ ] Implement streaming responses
- [ ] Add logging and monitoring

## 🤝 Contributing

Feel free to extend this project with:
- Additional knowledge bases
- More specialized tools
- Enhanced agent capabilities
- Performance optimizations

## 📝 License

Add your license information here.

## 📞 Contact

For questions about this project or the professor's information, refer to ENSET Mohammedia.

---

**Last Updated:** April 2026
