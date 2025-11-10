import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA 
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# Configuration
DB_FAISS_PATH = "vectorstore/db_faiss"

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's medical question.
If you don't know the answer based on the context, just say that you don't know. Don't try to make up an answer.
Don't provide anything outside of the given context.

Context: {context}
Question: {question}

Provide a clear, concise medical answer based only on the context above.
Start the answer directly without any introduction.
"""

# Simple and clean medical CSS
st.markdown("""
<style>
    .stApp {
        background-image: linear-gradient(rgba(255,255,255,0.9), rgba(255,255,255,0.9)), 
                          url('https://images.unsplash.com/photo-1559757148-5c350d0d3c56?ixlib=rb-4.0.3&auto=format&fit=crop&w=2000&q=80');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    .main-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 30px;
        margin: 20px auto;
        max-width: 900px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        border: 1px solid #e0f2f1;
    }
    .medical-header {
        text-align: center;
        color: #1e88e5;
        margin-bottom: 10px;
    }
    .medical-subtitle {
        text-align: center;
        color: #546e7a;
        margin-bottom: 30px;
        font-size: 1.1em;
    }
    .stChatInput {
        border-radius: 20px !important;
        border: 2px solid #1e88e5 !important;
        padding: 12px !important;
        background: white !important;
    }
    .user-message {
        background: #e3f2fd;
        padding: 12px 18px;
        border-radius: 15px 15px 5px 15px;
        margin: 8px 0;
        border-left: 4px solid #1e88e5;
    }
    .assistant-message {
        background: #e8f5e8;
        padding: 12px 18px;
        border-radius: 15px 15px 15px 5px;
        margin: 8px 0;
        border-left: 4px solid #4caf50;
    }
    .medical-tag {
        display: inline-block;
        background: #1e88e5;
        color: white;
        padding: 6px 12px;
        border-radius: 15px;
        font-size: 0.8em;
        margin: 2px;
    }
    .disclaimer {
        background: #fff8e1;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #ffa000;
        margin-top: 20px;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_vectorstore():
    """Load the FAISS vector store with embeddings"""
    try:
        embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        return db
    except Exception as e:
        st.error(f"Error loading vector store: {str(e)}")
        return None

def setup_qa_chain():
    """Set up the QA chain with OpenAI"""
    try:
        # Get vector store
        vectorstore = get_vectorstore()
        if vectorstore is None:
            return None
        
        # Create prompt template
        prompt = PromptTemplate(
            template=CUSTOM_PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )
        
        # Initialize OpenAI LLM
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.1,
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            max_tokens=500
        )
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={'k': 4}
            ),
            return_source_documents=True,
            chain_type_kwargs={'prompt': prompt}
        )
        
        return qa_chain
        
    except Exception as e:
        st.error(f"Error setting up QA chain: {str(e)}")
        return None

def format_response(result, source_documents):
    """Format the response in a user-friendly way"""
    response = result
    
    if source_documents:
        unique_sources = set()
        for doc in source_documents:
            if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                unique_sources.add(doc.metadata['source'])
        
        if unique_sources:
            sources_text = "\n\n**📚 Sources:** " + ", ".join([f"`{src}`" for src in unique_sources])
            response += sources_text
    
    return response

def main():
    # Page configuration
    st.set_page_config(
        page_title="Medical AI Assistant",
        page_icon="🏥",
        layout="centered"
    )
    
    # Main container
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="medical-header"><h1>🏥 Medical AI Assistant</h1></div>', unsafe_allow_html=True)
    st.markdown('<div class="medical-subtitle">Ask medical questions • Get AI-powered answers • Trusted information</div>', unsafe_allow_html=True)
    
    # Simple tags
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="medical-tag">🤖 AI Powered</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="medical-tag">🔒 Private</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="medical-tag">⚡ Instant</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Initialize session state for chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'qa_chain' not in st.session_state:
        with st.spinner("Loading medical knowledge..."):
            st.session_state.qa_chain = setup_qa_chain()
    
    # Display chat messages
    for message in st.session_state.messages:
        if message['role'] == 'user':
            st.markdown(f'<div class="user-message"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="assistant-message"><strong>Assistant:</strong> {message["content"]}</div>', unsafe_allow_html=True)
    
    # Chat input
    prompt = st.chat_input("💬 Ask a medical question...")
    
    if prompt:
        # Add user message to chat history
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        
        # Display user message
        st.markdown(f'<div class="user-message"><strong>You:</strong> {prompt}</div>', unsafe_allow_html=True)
        
        # Check if QA chain is available
        if st.session_state.qa_chain is None:
            st.markdown('<div class="assistant-message"><strong>Assistant:</strong> System not available. Please check configuration.</div>', unsafe_allow_html=True)
            return
        
        # Generate response
        with st.spinner("Analyzing your question..."):
            try:
                # Get response from QA chain
                response = st.session_state.qa_chain.invoke({'query': prompt})
                
                # Extract results
                result = response.get("result", "Sorry, I couldn't process your question.")
                source_documents = response.get("source_documents", [])
                
                # Format the response
                formatted_response = format_response(result, source_documents)
                
                # Display assistant response
                st.markdown(f'<div class="assistant-message"><strong>Assistant:</strong> {formatted_response}</div>', unsafe_allow_html=True)
                
                # Add assistant response to chat history
                st.session_state.messages.append({
                    'role': 'assistant', 
                    'content': formatted_response
                })
                
            except Exception as e:
                error_msg = f"Sorry, I encountered an error. Please try again."
                st.markdown(f'<div class="assistant-message"><strong>Assistant:</strong> {error_msg}</div>', unsafe_allow_html=True)
                st.session_state.messages.append({
                    'role': 'assistant', 
                    'content': error_msg
                })
    
    # Simple disclaimer
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()