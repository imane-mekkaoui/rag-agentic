from io import BytesIO

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_community.vectorstores import Chroma
from langchain_core.tools import create_retriever_tool
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

load_dotenv(override=True)

PDF_SYSTEM_PROMPT = """Tu es un assistant RAG. Tu réponds UNIQUEMENT à partir des PDF déposés.

Règles :
- Appelle TOUJOURS un outil avant de répondre. Ne pose pas de question de clarification à la place d'un appel d'outil.
- Pour un résumé, une vue d'ensemble, "de quoi parle ce PDF", les points clés : utilise get_uploaded_pdf_content.
- Pour une question précise : utilise pdf_search avec une requête en mots-clés.
- N'utilise pas tes connaissances générales ni d'autres documents.
- Si l'outil ne contient pas l'information, réponds : "Cette information n'apparaît pas dans le(s) PDF déposé(s)."
- Cite le nom du fichier quand c'est possible.
- Après l'outil, rédige une réponse utile et complète.
"""

embedding_model = OpenAIEmbeddings()
vectorstore = Chroma(
    collection_name="uploaded_pdfs",
    embedding_function=embedding_model,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
retriever_tool = create_retriever_tool(
    retriever=retriever,
    name="pdf_search",
    description="Search uploaded PDFs for a specific topic or keyword.",
)


@tool
def get_uploaded_pdf_content() -> str:
    """Return the full indexed text of uploaded PDFs. Use for summaries and overview questions."""
    try:
        data = vectorstore.get()
    except Exception as exc:
        return f"Aucun PDF indexé ({exc})."
    documents = data.get("documents") or []
    metadatas = data.get("metadatas") or []
    if not documents:
        return "Aucun PDF n'est indexé."
    parts = []
    for text, meta in zip(documents, metadatas or [{}] * len(documents)):
        source = (meta or {}).get("source", "PDF")
        parts.append(f"[{source}]\n{text}")
    return "\n\n".join(parts)[:20000]


llm = ChatOpenAI(model="gpt-4o", temperature=0)
agent = create_agent(
    model=llm,
    tools=[retriever_tool, get_uploaded_pdf_content],
    system_prompt=PDF_SYSTEM_PROMPT,
)


def clear_documents() -> None:
    try:
        data = vectorstore.get()
    except Exception:
        return
    ids = data.get("ids") or []
    if ids:
        vectorstore.delete(ids=ids)


def ingest_pdf(file_bytes: bytes, filename: str) -> int:
    reader = PdfReader(BytesIO(file_bytes))
    pages = []
    for index, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(f"[{filename} p.{index}]\n{text}")
    if not pages:
        raise ValueError(f"Aucun texte extractible dans {filename}.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    pieces = splitter.split_text("\n\n".join(pages))
    vectorstore.add_texts(
        texts=pieces,
        metadatas=[{"source": filename} for _ in pieces],
    )
    return len(pieces)
