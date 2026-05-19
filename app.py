import os
import streamlit as st
import re 
from dotenv import load_dotenv

from pubmed import PubMedRetriever 
from rag_chain import get_mediassist_chain, ingest_new_articles 
from router import get_route
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Load environment variables
load_dotenv()

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="MediAssist AI | Clinical Research Assistant",
    page_icon="🏥",
    layout="wide"
)

# --- CUSTOM STYLING ---
st.markdown("""
    <style>
    .stChatMessage { border-radius: 10px; margin-bottom: 10px; }
    .source-box { 
        background-color: #f0f2f6; 
        padding: 10px; 
        border-radius: 5px; 
        border-left: 5px solid #007bff;
        font-size: 0.85rem;
    }
    </style>
    """, unsafe_allow_html=True)

# --- INITIALIZATION & CACHING ---
@st.cache_resource
def load_rag_chain():
    return get_mediassist_chain()

@st.cache_resource
def load_chitchat_llm():
    return ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.7)

# --- SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []
# NEW: Track fetched articles before ingesting them
if "fetched_articles" not in st.session_state:
    st.session_state.fetched_articles = []

# --- SIDEBAR: DYNAMIC INGESTION & TOOLS ---
with st.sidebar:
    st.title("🏥 MediAssist AI")
    st.divider()

    # --- STEP 1: SEARCH PUBMED ---
    st.header("1️⃣ Search Research")
    st.caption("Search PubMed for relevant clinical studies.")
    
    search_query = st.text_input("PubMed Search Query", placeholder="e.g., Metformin and T2DM")
    num_articles = st.slider("Number of articles to fetch", min_value=1, max_value=20, value=5)
    
    if st.button("🔍 Search PubMed"):
        if search_query:
            with st.spinner(f"Searching PubMed for '{search_query}'..."):
                scraper = PubMedRetriever()
                pmids = scraper.search_pubmed_articles(search_query, max_results=num_articles)
                
                if not pmids:
                    st.warning("No articles found for that query.")
                    st.session_state.fetched_articles = []
                else:
                    st.info(f"Downloading {len(pmids)} abstracts...")
                    # Store the results in session state so they don't disappear
                    st.session_state.fetched_articles = scraper.fetch_pubmed_abstracts(pmids)
        else:
            st.warning("Please enter a search query first.")

    # --- STEP 2: REVIEW & INGEST ---
    if st.session_state.fetched_articles:
        st.divider()
        st.subheader("2️⃣ Review & Ingest")
        st.caption("Select the papers you want to add to the knowledge base:")
        
        # Track which articles the user selects
        selected_articles = []
        
        for art in st.session_state.fetched_articles:
            pmid = art['pmid']
            title = art['title']
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            
            # Display the Title as a clickable link
            st.markdown(f"**[{title}]({url})**")
            
            # Checkbox to select/deselect (Defaults to True/Checked)
            is_selected = st.checkbox(f"Select PMID: {pmid}", value=True, key=f"chk_{pmid}")
            if is_selected:
                selected_articles.append(art)
                
        # Ingest Button
        if st.button("📥 Ingest Selected"):
            if selected_articles:
                with st.spinner("Embedding into Vector Database..."):
                    chunks_added = ingest_new_articles(selected_articles)
                    
                    # CRITICAL: Clear the cache so the Retriever connects to the newly updated Database!
                    load_rag_chain.clear()
                    
                    st.success(f"✅ Success! Added {len(selected_articles)} articles ({chunks_added} chunks).")
                   
            else:
                st.warning("Please select at least one article to ingest.")

    st.divider()

    # --- CHAT TOOLS & SOURCES ---
    st.header("3️⃣ Chat & Data Tools")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.session_state.last_sources = []
            st.rerun()
            
    with col2:
        if st.button("⚠️ Clear Database"):
            import shutil
            # Delete the Chroma directory physically
            if os.path.exists("chroma_store"):
                shutil.rmtree("chroma_store")
            # Clear UI states and cache
            st.session_state.messages = []
            st.session_state.last_sources = []
            st.session_state.fetched_articles = []
            load_rag_chain.clear()
            st.success("Database wiped!")
            st.rerun()

    st.divider()
    st.write("**Recent Cited PMIDs:**")
    sources_placeholder = st.empty()

def update_sidebar_sources(pmids):
    sources_placeholder.empty()
    with sources_placeholder.container():
        if pmids:
            for pmid in pmids:
                if pmid and pmid != 'N/A':
                    st.markdown(f"🔗 [PMID: {pmid}](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)")
        else:
            st.write("No sources cited yet.")

update_sidebar_sources(st.session_state.last_sources)

# --- MAIN INTERFACE: QUERY BAR ---
st.title("Clinical Research Assistant")

if not os.environ.get("GROQ_API_KEY"):
    st.warning("⚠️ `GROQ_API_KEY` is not set. Please add it to your environment or `.env` file.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a clinical question about the ingested documents..."):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    chat_history = []
    for m in st.session_state.messages[-7:-1]: 
        if m["role"] == "user":
            chat_history.append(HumanMessage(content=m["content"]))
        else:
            chat_history.append(AIMessage(content=m["content"]))

    route = get_route(prompt)

    with st.chat_message("assistant"):
        
        # --- CHITCHAT ROUTE ---
        if route == "chitchat" or (route is None and len(prompt.split()) < 4):
            with st.spinner("MediAssist is typing..."):
                llm = load_chitchat_llm()
                
                system_instructions = (
                    "You are MediAssist AI, a professional, empathetic, and friendly clinical assistant. "
                    "For general greetings, identity questions, or casual talk, be warm and helpful. "
                    "CRITICAL: Do NOT use clinical formatting, do NOT mention 'References'."
                )
                
                context_messages = [SystemMessage(content=system_instructions)] + chat_history + [HumanMessage(content=prompt)]
                
                try:
                    response = llm.invoke(context_messages)
                    full_answer = response.content
                except Exception as e:
                    full_answer = f"Error communicating with LLM: {str(e)}"
                    
        # --- CLINICAL RAG ROUTE ---
        else:
            with st.spinner("Analyzing PubMed evidence..."):
                chain = load_rag_chain()
                
                # Graceful handling if the database is completely empty
                if chain:
                    try:
                        response = chain.invoke({"input": prompt, "chat_history": chat_history})
                        full_answer = response["answer"]
                        
                        cited_pmids = set(re.findall(r"PMID:\s*(\d+)", full_answer))
                        st.session_state.last_sources = sorted(list(filter(None, cited_pmids)))
                        update_sidebar_sources(st.session_state.last_sources)
                        
                    except Exception as e:
                        full_answer = f"I encountered an error querying the research engine: {str(e)}"
                else:
                    full_answer = "I'm sorry, my clinical knowledge base is currently empty. Please use the **Search & Ingest** tools in the sidebar to add PubMed articles before asking clinical questions."

        st.markdown(full_answer)
        st.session_state.messages.append({"role": "assistant", "content": full_answer})

st.divider()
st.caption("Data powered by NCBI PubMed API.")