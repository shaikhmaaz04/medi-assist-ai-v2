import os
import shutil
from operator import itemgetter
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# --- CONFIGURATION ---
load_dotenv()
CHROMA_DIR = "chroma_store"
COLLECTION = "fasting_research"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.3-70b-versatile"

# --- SINGLETON DATABASE CONNECTION ---
# This ensures we only ever open one connection to SQLite, preventing read/write locks.
_vectorstore = None

def get_vectorstore():
    """Returns the active database connection, creating it if it doesn't exist."""
    global _vectorstore
    if _vectorstore is None:
        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        _vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
            collection_name=COLLECTION
        )
    return _vectorstore

def clear_database():
    """Safely releases the database connection and deletes the files."""
    global _vectorstore
    _vectorstore = None  # Release the active connection so the OS allows deletion
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)

def format_docs(docs):
    """Injects Title, Citation, AND PMID into the context block."""
    formatted_context = []
    for doc in docs:
        title = doc.metadata.get('title', 'Unknown Title')
        citation = doc.metadata.get('citation', 'Unknown Source')
        pmid = doc.metadata.get('pmid', 'N/A')
        
        formatted_context.append(
            f"PAPER TITLE: {title}\n"
            f"SOURCE: {citation}\n"
            f"PMID: {pmid}\n"
            f"CONTENT: {doc.page_content}"
        )
    return "\n\n".join(formatted_context)

def ingest_new_articles(articles):
    """Dynamically chunks and embeds newly fetched PubMed articles into ChromaDB."""
    if not articles:
        return 0

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_chunks = []
    
    for art in articles:
        abstract_data = art.get('abstract', {})
        if isinstance(abstract_data, dict):
            full_abstract = "\n".join([f"{k}: {v}" for k, v in abstract_data.items()])
        else:
            full_abstract = str(abstract_data)
            
        content_for_ai = f"STUDY: {art['title']}\nFINDINGS: {full_abstract}"
        
        authors = art.get('authors', 'No Authors')
        author_names = authors.split(',')
        last_name = author_names[0].split(' ')[-1] if authors != "No Authors" else "Unknown"
        pub_date = art.get('publication_date', 'Unknown Year')
        pmid = art.get('pmid', 'N/A')
        citation = f"{last_name} et al., {pub_date} (PMID: {pmid})"

        chunks = text_splitter.split_text(content_for_ai)
        for i, chunk in enumerate(chunks):
            all_chunks.append(Document(
                page_content=chunk,
                metadata={
                    "pmid": pmid,
                    "title": art.get('title', 'Unknown Title'),
                    "citation": citation,
                    "journal": art.get('journal', 'Unknown Journal'),
                    "chunk_id": i
                }
            ))

    if not all_chunks:
        return 0

    # Use the global singleton connection to append data safely
    vs = get_vectorstore()
    vs.add_documents(all_chunks)
    
    return len(all_chunks)

def get_mediassist_chain():
    """Builds the RAG pipeline. Returns None if the database is empty."""
    if not os.path.exists(CHROMA_DIR):
        return None

    # Use the same global connection for retrieval
    vs = get_vectorstore()
    retriever = vs.as_retriever(search_kwargs={"k": 5})

    llm = ChatGroq(model_name=LLM_MODEL, temperature=0.4)

    system_prompt = (
        "You are MediAssist AI, a highly accurate clinical research assistant. "
        "Your ONLY source of truth is the CURRENT CONTEXT provided below. "
        "You are strictly forbidden from using pre-trained general knowledge.\n\n"
        "CRITICAL INSTRUCTION:\n"
        "First, determine if the CURRENT CONTEXT contains the SPECIFIC answer to the user's question. "
        "Just because the context mentions a general topic (e.g., a drug name) does NOT mean it contains the specific answer.\n\n"
        "IF THE CONTEXT LACKS THE SPECIFIC ANSWER:\n"
        "You MUST reply EXACTLY and ONLY with this phrase: 'The current database lacks this specific information.'\n"
        "Do NOT add 'However...'. Do NOT use PMIDs from your internal memory. Stop generating immediately.\n\n"
        "IF THE CONTEXT CONTAINS THE EXACT ANSWER:\n"
        "Follow these strict formatting guidelines:\n"
        "1. SYNTHESIS: Merge findings from the CURRENT CONTEXT into a cohesive narrative.\n"
        "2. CITATIONS: Every clinical claim must have a concise inline citation using ONLY the ID from the context, formatted exactly as: [PMID: XXXXXX]. "
        "CRITICAL: Do NOT invent PMIDs. If a PMID is not physically present in the CURRENT CONTEXT below, you cannot use it.\n"
        "3. DETAIL: Include specific metrics when available.\n"
        "4. CLINICAL SUMMARY: Conclude with a 2-3 sentence 'Executive Summary'.\n"
        "5. REFERENCES: Provide a numbered list of ONLY the papers cited in this specific response at the very end. The list should contain the respective paper titles as well.\n\n"
        "CURRENT CONTEXT:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    setup_and_retrieval = RunnableParallel({
        "context": itemgetter("input") | retriever | format_docs, 
        "input": itemgetter("input"), 
        "chat_history": itemgetter("chat_history"),
        "raw_docs": itemgetter("input") | retriever
    })

    rag_chain = setup_and_retrieval | {
        "answer": prompt | llm | StrOutputParser(),
        "docs": itemgetter("raw_docs")
    }
    
    return rag_chain