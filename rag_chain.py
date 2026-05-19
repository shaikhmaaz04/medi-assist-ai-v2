import os
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
# LLM_MODEL = "llama-3.1-8b-instant"

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
        # Format the abstract dictionary into a string
        abstract_data = art.get('abstract', {})
        if isinstance(abstract_data, dict):
            full_abstract = "\n".join([f"{k}: {v}" for k, v in abstract_data.items()])
        else:
            full_abstract = str(abstract_data)
            
        content_for_ai = f"STUDY: {art['title']}\nFINDINGS: {full_abstract}"
        
        # Build a clean citation string
        authors = art.get('authors', 'No Authors')
        author_names = authors.split(',')
        last_name = author_names[0].split(' ')[-1] if authors != "No Authors" else "Unknown"
        pub_date = art.get('publication_date', 'Unknown Year')
        pmid = art.get('pmid', 'N/A')
        citation = f"{last_name} et al., {pub_date} (PMID: {pmid})"

        # Split and package into LangChain Documents
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

    # Embed and store directly in the active ChromaDB collection
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION
    )
    vectorstore.add_documents(all_chunks)
    
    return len(all_chunks)

def get_mediassist_chain():
    """Builds the RAG pipeline. Returns None if the database is empty."""
    if not os.path.exists(CHROMA_DIR):
        # Gracefully handle empty states before the user ingests data
        return None

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

   
    llm = ChatGroq(model_name=LLM_MODEL, temperature=0.5, max_tokens=4500)

    system_prompt = (
        "You are MediAssist AI, a clinical research assistant. "
        "Answer ONLY using the CURRENT CONTEXT provided below. "
        "Do not use outside or general knowledge.\n\n"

        "If the CURRENT CONTEXT does not contain the answer, "
        "reply ONLY with:\n"
        "'The current database lacks this specific information.'\n"
        "Do not add anything else.\n\n"

        "If the answer exists in the CURRENT CONTEXT:\n"
        "1. Write a clear, detailed and combined explanation from the context.\n"
        "2. Add inline citations for every clinical claim using this exact format: [PMID: XXXXXX]\n"
        "3. Include important metrics when available (example: HbA1c changes, weight loss, percentages).\n"
        "4. End with a short 2-3 sentence Executive Summary.\n"
        "5. Add a numbered References section at the end containing ONLY the cited papers and their titles.\n\n"

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