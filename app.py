import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os

# Load environment variables
load_dotenv()

# Functions for PDF and text processing
def get_pdf_text(pdf):
    """Extract text from a PDF file."""
    text = ""
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    """Split text into chunks."""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    """Generate embeddings and store them in FAISS vector database."""
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    """Create a conversational retrieval chain."""
    llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), temperature=0.2)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# Streamlit App
st.set_page_config(page_title="PDF Chatbot", layout="wide")

st.title("📄 PDF Chatbot with LangChain & OpenAI")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    with st.spinner("Processing PDF..."):
        # Extract text from PDF
        raw_text = get_pdf_text(uploaded_file)
        st.write("### Extracted Text Snippet:")
        st.write(raw_text[:500])  # Display a snippet of the text

        # Chunk the text
        text_chunks = get_text_chunks(raw_text)
        st.success(f"✅ Split the PDF into {len(text_chunks)} chunks.")

        # Create FAISS vector store
        vectorstore = get_vectorstore(text_chunks)
        conversation_chain = get_conversation_chain(vectorstore)

        # Chat Interface
        st.write("### Start Chatting with Your PDF!")
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        # User Input
        user_question = st.text_input("Ask a question about the PDF:")
        if user_question:
            response = conversation_chain({"question": user_question})
            st.session_state["chat_history"].append(("user", user_question))
            st.session_state["chat_history"].append(("bot", response["answer"]))

        # Display Chat History
        for role, message in st.session_state["chat_history"]:
            if role == "user":
                st.markdown(f"**You:** {message}")
            else:
                st.markdown(f"**Assistant:** {message}")