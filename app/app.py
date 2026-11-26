
import streamlit as st
import sys
from pathlib import Path

# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = Path(__file__).resolve().parent / "assets"

sys.path.insert(0, str(PROJECT_ROOT))

from src.rag_pipeline import RAGPipeline


# ==========================================
# 2. LOAD RAG PIPELINE
# ==========================================

@st.cache_resource
def load_rag():
    return RAGPipeline()


# ==========================================
# 3. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="UniGuide AI",
    page_icon="🎓",
    layout="wide"
)


# ==========================================
# 4. SESSION STATE
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ==========================================
# 5. CUSTOM CSS
# ==========================================

st.markdown("""
<style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        color: #244B80;
    }

    .subtitle {
        font-size: 18px;
        color: #777777;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 12px;
        padding: 12px;
    }

</style>
""", unsafe_allow_html=True)


# ==========================================
# 6. SIDEBAR
# ==========================================

with st.sidebar:

    st.image(
        str(ASSETS_DIR / "user-robot.png"),
        width=90
    )

    st.title("UniGuide AI")

    st.caption("Your scholarship assistant")

    st.divider()

    st.subheader("Explore")

    if st.button(
        "New conversation",
        use_container_width=True
    ):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.caption("Powered by RAG and Qwen 2.5")


# ==========================================
# 7. MAIN PAGE
# ==========================================

st.markdown(
    '<p class="main-title">Welcome to UniGuide AI</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">'
    'Find answers to your scholarship questions.'
    '</p>',
    unsafe_allow_html=True
)

st.divider()


# ==========================================
# 8. SCHOLARSHIP CATEGORIES
# ==========================================

st.subheader("What would you like to know?")

col1, col2 = st.columns(2)


# Left column
with col1:

    with st.container(border=True):

        st.image(
            str(ASSETS_DIR / "graduation-cap.png"),
            width=85
        )

        st.subheader("Scholarship opportunities")

        st.write(
            "Discover available scholarships "
            "and study programmes."
        )

    with st.container(border=True):

        st.image(
            str(ASSETS_DIR / "book-brain.png"),
            width=85
        )

        st.subheader("Eligibility requirements")

        st.write(
            "Understand admission and eligibility criteria."
        )


# Right column
with col2:

    with st.container(border=True):

        st.image(
            str(ASSETS_DIR / "earth-americas.png"),
            width=85
        )

        st.subheader("Study destinations")

        st.write(
            "Explore international study opportunities."
        )

    with st.container(border=True):

        st.image(
            str(ASSETS_DIR / "memo-pad.png"),
            width=85
        )

        st.subheader("Deadlines and applications")

        st.write(
            "Learn about application procedures and deadlines."
        )


st.divider()


# ==========================================
# 9. CHAT HISTORY
# ==========================================

st.subheader("Chat with UniGuide AI")

# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ==========================================
# 10. CHAT INPUT
# ==========================================

question = st.chat_input(
    "Ask UniGuide AI anything about scholarships..."
)


# ==========================================
# 11. GENERATE ANSWER
# ==========================================

if question:

    # Display user question
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)


    # Generate assistant answer
    with st.chat_message("assistant"):

        with st.spinner("Searching scholarship information..."):

            try:

                rag = load_rag()

                answer = rag.generate_answer(
                    question,
                    k=3
                )

            except Exception as e:

                answer = (
                    f"An error occurred while generating "
                    f"the answer: {e}"
                )

        st.markdown(answer)


    # Save assistant answer
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })
