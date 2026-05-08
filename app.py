import streamlit as st
from dotenv import load_dotenv
import os

# Load environment variables
try:
    load_dotenv(override=True)
except:
    pass

if hasattr(st, "secrets") and "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="YouTube RAG Chatbot",
    page_icon="▶️",
    layout="wide"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #FF0000 0%, #cc0000 100%);
        padding: 2rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }

    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
    }

    .main-header p {
        font-size: 1rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }

    .video-info-box {
        background: #f8f9fa;
        border: 1px solid #e0e0e0;
        border-left: 4px solid #FF0000;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }

    .chat-message-user {
        background: #e8f4fd;
        border-radius: 12px 12px 2px 12px;
        padding: 0.8rem 1.2rem;
        margin: 0.5rem 0;
        margin-left: 20%;
        border: 1px solid #c2e0f4;
    }

    .chat-message-bot {
        background: #f0f0f0;
        border-radius: 12px 12px 12px 2px;
        padding: 0.8rem 1.2rem;
        margin: 0.5rem 0;
        margin-right: 20%;
        border: 1px solid #e0e0e0;
    }

    .status-success {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }

    .stButton > button {
        background: #FF0000;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        width: 100%;
    }

    .stButton > button:hover {
        background: #cc0000;
    }
</style>
""", unsafe_allow_html=True)

# ─── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>▶️ YouTube RAG Chatbot</h1>
    <p>Paste a YouTube video URL and chat with its transcript using AI</p>
</div>
""", unsafe_allow_html=True)

# ─── Session State Init ──────────────────────────────────────────────────────────
if "main_chain" not in st.session_state:
    st.session_state.main_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "video_loaded" not in st.session_state:
    st.session_state.video_loaded = False
if "video_id" not in st.session_state:
    st.session_state.video_id = ""

# ─── Helper Functions ────────────────────────────────────────────────────────────
def extract_video_id(url_or_id: str) -> str:
    """Extract video ID from full URL or return as-is if already an ID."""
    url_or_id = url_or_id.strip()
    if "youtube.com/watch?v=" in url_or_id:
        return url_or_id.split("v=")[1].split("&")[0]
    elif "youtu.be/" in url_or_id:
        return url_or_id.split("youtu.be/")[1].split("?")[0]
    else:
        return url_or_id  # Assume it's already a video ID


def build_rag_chain(video_id: str):
    """Build the full RAG chain from a YouTube video ID."""
    # Step 1: Fetch transcript
    ytt_api = YouTubeTranscriptApi()
    transcript_list = ytt_api.fetch(video_id)
    transcript = " ".join(chunk.text for chunk in transcript_list)

    # Step 2: Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])

    # Step 3: Embed and store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    # Step 4: LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-lite",
        temperature=0.2,
        google_api_key=os.environ.get("GOOGLE_API_KEY")
    )

    # Step 5: Prompt
    prompt = PromptTemplate(
        template="""
          You are a helpful assistant that answers questions about a YouTube video transcript.
          Answer ONLY from the provided transcript context.
          If the context is insufficient, just say you don't know based on the transcript.

          {context}
          Question: {question}
        """,
        input_variables=["context", "question"]
    )

    # Step 6: Chain
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    parallel_chain = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    })

    main_chain = parallel_chain | prompt | llm | StrOutputParser()
    return main_chain, len(chunks), len(transcript)


# ─── Sidebar: Video Loader ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎬 Load a YouTube Video")
    st.markdown("---")

    video_input = st.text_input(
        "YouTube URL or Video ID",
        placeholder="https://youtube.com/watch?v=... or Gfr50f6ZBvo"
    )

    load_btn = st.button("🚀 Load & Process Video")

    if load_btn and video_input:
        video_id = extract_video_id(video_input)
        with st.spinner("⏳ Fetching transcript and building index..."):
            try:
                chain, num_chunks, transcript_len = build_rag_chain(video_id)
                st.session_state.main_chain = chain
                st.session_state.video_loaded = True
                st.session_state.video_id = video_id
                st.session_state.chat_history = []  # Reset chat on new video
                st.success(f"✅ Video loaded!\n\n📄 {transcript_len:,} chars → {num_chunks} chunks")
            except TranscriptsDisabled:
                st.error("❌ No captions available for this video.")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    if st.session_state.video_loaded:
        st.markdown("---")
        st.markdown(f"**Loaded Video ID:**\n`{st.session_state.video_id}`")
        st.markdown(f"[Open on YouTube ↗](https://youtube.com/watch?v={st.session_state.video_id})")
        st.markdown("---")

        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

    st.markdown("---")
    st.markdown("### ℹ️ How to use")
    st.markdown("""
1. Paste a YouTube URL
2. Click **Load & Process Video**
3. Ask any question about the video!
    """)

# ─── Main: Chat Interface ────────────────────────────────────────────────────────
if not st.session_state.video_loaded:
    st.info("👈 Paste a YouTube URL in the sidebar and click **Load & Process Video** to get started.")

    # Show example questions
    st.markdown("### 💡 Example questions you can ask:")
    cols = st.columns(2)
    examples = [
        "Can you summarize the video?",
        "What are the main topics discussed?",
        "What did they say about AI?",
        "Who is being interviewed?",
    ]
    for i, ex in enumerate(examples):
        with cols[i % 2]:
            st.markdown(f"- *{ex}*")

else:
    st.markdown("### 💬 Chat with the Video")

    # Display chat history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="chat-message-user">
                <strong>You:</strong> {msg["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message-bot">
                <strong>🤖 Assistant:</strong> {msg["content"]}
            </div>
            """, unsafe_allow_html=True)

    # Chat input
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        with col1:
            user_question = st.text_input(
                "Ask a question about the video",
                placeholder="e.g. What is the main topic of this video?",
                label_visibility="collapsed"
            )
        with col2:
            submit = st.form_submit_button("Send ➤")

    if submit and user_question:
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": user_question})

        with st.spinner("🤔 Thinking..."):
            try:
                answer = st.session_state.main_chain.invoke(user_question)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
            except Exception as e:
                err = f"Error getting answer: {str(e)}"
                st.session_state.chat_history.append({"role": "assistant", "content": err})

        st.rerun()

    # Quick question buttons
    if not st.session_state.chat_history:
        st.markdown("#### 💡 Quick Questions")
        qcols = st.columns(2)
        quick_qs = [
            "Summarize this video",
            "What are the key points?",
            "Who is being interviewed?",
            "What topics are covered?"
        ]
        for i, q in enumerate(quick_qs):
            with qcols[i % 2]:
                if st.button(q, key=f"quick_{i}"):
                    st.session_state.chat_history.append({"role": "user", "content": q})
                    with st.spinner("🤔 Thinking..."):
                        try:
                            answer = st.session_state.main_chain.invoke(q)
                            st.session_state.chat_history.append({"role": "assistant", "content": answer})
                        except Exception as e:
                            st.session_state.chat_history.append({"role": "assistant", "content": str(e)})
                    st.rerun()