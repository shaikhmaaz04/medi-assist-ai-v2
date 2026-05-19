import os
from semantic_router import Route
from semantic_router.encoders import HuggingFaceEncoder
from semantic_router.routers import SemanticRouter

# Suppress tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- 1. CHITCHAT ROUTE ---
# We ONLY define the chitchat route. Everything else will default to clinical!
chitchat = Route(
    name="chitchat",
    score_threshold=0.35,  # Balanced threshold to confidently catch conversational phrases
    utterances=[
        "hi", "hello", "hey", "hi there", "hello assistant",
        "how are you?", "who are you?", "what is your name?",
        "what can you do?", "help me", "tell me a joke",
        "thank you", "thanks", "bye", "see ya", "good morning",
        "who created you?", "are you a doctor?", "how's it going?",
        "how are you man", "what's up", "hey man", "ok", "okay",
        "good evening", "good afternoon", "nice to meet you", "pleasure to meet you",
        "what's your purpose?", "what's your function?", "do you have feelings?",
        "can you chat?", "let's talk", "talk to me", "are you alive?", "do you understand me?", "what's the weather?", "what's the time?",
        "what are your capabilities", "how can you help me", 
        "what do you do", "explain your features", "are you an ai",

    ],
)

try:
    _encoder = HuggingFaceEncoder(name="sentence-transformers/all-MiniLM-L6-v2")
    # Notice we removed the 'clinical' route from the list
    _router = SemanticRouter(encoder=_encoder, routes=[chitchat], auto_sync="local")
except Exception as e:
    print(f"Failed to initialize router globally: {e}")
    _router = None

def get_route(text):
    """Executes the pre-loaded router safely with a Default Fallback."""
    if not _router:
        return "clinical"  # Failsafe default
        
    try:
        # Semantic Router mapping
        result = _router(text)
        
        # 1. If the router confidently matches Chitchat, route to Chitchat.
        if result and result.name == "chitchat":
            print(f"🧠 Route identified (Semantic Router): chitchat")
            return "chitchat"
            
        # 2. THE CATCH-ALL: If it's NOT chitchat, it MUST be a clinical question!
        else:
            print(f"🧠 Route identified (Fallback): clinical")
            return "clinical"
        
    except Exception as e:
        print(f"Routing Error: {e}")
        # Always default to the RAG pipeline if something goes wrong
        return "clinical"