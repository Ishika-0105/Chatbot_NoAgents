import streamlit as st
import os
from config import Config
from document_processor import DocumentProcessor
from vector_store import VectorStore
from chat_memory import ChatMemory
from llm_manager import LLMManager
from evaluation_metrics import TriadEvaluationMetrics, TriadMetricsDisplay
import time
from typing import Dict, Any
import pandas as pd
import json
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import io

# Page configuration
st.set_page_config(
    page_title="📊 Advanced Financial RAG Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for professional appearance
st.markdown(f"""
<style>
    .stApp {{
        background: linear-gradient(135deg, {Config.COLORS['background']} 0%, #F8F9FA 100%);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }}
    
    .main-header {{
        background: linear-gradient(135deg, {Config.COLORS['gradient_start']}, {Config.COLORS['gradient_end']});
        color: white;
        text-align: center;
        padding: 2rem 0;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(178, 34, 34, 0.3);
    }}
    
    .main-header h1 {{
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }}
    
    .main-header p {{
        margin: 0.5rem 0 0 0;
        font-size: 1.2rem;
        opacity: 0.9;
    }}
    
    .chat-message {{
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: transform 0.2s ease;
    }}
    
    .chat-message:hover {{
        transform: translateY(-2px);
    }}
    
    .user-message {{
        background: linear-gradient(135deg, #FFF 0%, #F8F9FA 100%);
        border-left: 4px solid {Config.COLORS['primary']};
        margin-left: 2rem;
    }}
    
    .assistant-message {{
        background: linear-gradient(135deg, #FFF 0%, #FAFBFC 100%);
        border-left: 4px solid {Config.COLORS['accent']};
        margin-right: 2rem;
    }}
    
    .source-info {{
        font-size: 0.85rem;
        color: {Config.COLORS['text_secondary']};
        font-style: italic;
        margin-top: 1rem;
        padding: 0.5rem;
        background-color: {Config.COLORS['background']};
        border-radius: 8px;
        border: 1px solid {Config.COLORS['border']};
    }}
    
    .stats-container {{
        background: linear-gradient(135deg, {Config.COLORS['gradient_start']}, {Config.COLORS['gradient_end']});
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 6px 20px rgba(178, 34, 34, 0.3);
    }}
    
    .stats-container h4 {{
        margin: 0 0 1rem 0;
        font-size: 1.2rem;
        font-weight: 600;
    }}
    
    .stats-container p {{
        margin: 0.5rem 0;
        font-size: 0.95rem;
    }}
    
    .metric-card {{
        background: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid {Config.COLORS['primary']};
    }}
    
    .stButton > button {{
        background: linear-gradient(135deg, {Config.COLORS['primary']}, {Config.COLORS['secondary']});
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(178, 34, 34, 0.3);
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(178, 34, 34, 0.4);
    }}
    
    .sidebar .stSelectbox > div > div {{
        background-color: white;
        border: 2px solid {Config.COLORS['border']};
        border-radius: 8px;
    }}
    
    .welcome-card {{
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-top: 4px solid {Config.COLORS['primary']};
    }}
    
    .feature-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }}
    
    .feature-card {{
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        border-left: 4px solid {Config.COLORS['accent']};
        transition: transform 0.2s ease;
    }}
    
    .feature-card:hover {{
        transform: translateY(-3px);
    }}
    
    .processing-indicator {{
        display: flex;
        align-items: center;
        padding: 1rem;
        background: linear-gradient(135deg, #E3F2FD, #BBDEFB);
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #2196F3;
    }}
    
    .evaluation-panel {{
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 6px 25px rgba(0,0,0,0.1);
        border-top: 4px solid {Config.COLORS['success']};
    }}
    
    .search-mode-info {{
        background: rgba(255,255,255,0.1);
        padding: 0.5rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        font-size: 0.85rem;
    }}
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize all session state variables"""
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = VectorStore()
    if 'chat_memory' not in st.session_state:
        st.session_state.chat_memory = ChatMemory()
    if 'llm_manager' not in st.session_state:
        st.session_state.llm_manager = LLMManager()
    if 'document_processor' not in st.session_state:
        st.session_state.document_processor = DocumentProcessor()
    if 'search_mode' not in st.session_state:
        st.session_state.search_mode = 'hybrid' 
    if 'evaluation_metrics' not in st.session_state:
        st.session_state.evaluation_metrics = TriadEvaluationMetrics()
    if 'documents_loaded' not in st.session_state:
        st.session_state.documents_loaded = False
    if 'show_evaluation' not in st.session_state:
        st.session_state.show_evaluation = Config.EVALUATION_ENABLED
    if 'last_evaluation' not in st.session_state:
        st.session_state.last_evaluation = None
    if 'last_retrieved_docs' not in st.session_state:
        st.session_state.last_retrieved_docs = []
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'processing_question' not in st.session_state:
        st.session_state.processing_question = False
    # --- Initialize Session State for Threads ---
    # This ensures that these variables persist across reruns
    if 'conversation_threads' not in st.session_state:
    # Key: thread_id, Value: ChatMemory instance
        st.session_state.conversation_threads = {} 

    if 'current_thread_id' not in st.session_state:
        st.session_state.current_thread_id = None # No thread selected initially

    if 'thread_metadata' not in st.session_state:
    # Stores display names and creation times for ordering
    # Key: thread_id, Value: {'name': 'Conversation X', 'created_at': datetime_obj}
        st.session_state.thread_metadata = {}

    # Ensure at least one default thread exists when the app starts
    if not st.session_state.conversation_threads:
        default_thread_id = "thread_default_" + datetime.now().strftime("%Y%m%d%H%M%S")
        st.session_state.conversation_threads[default_thread_id] = ChatMemory()
        st.session_state.thread_metadata[default_thread_id] = {
            'name': 'New Conversation 1',
            'created_at': datetime.now()
        }
        st.session_state.current_thread_id = default_thread_id

def create_new_thread():
    new_thread_id = f"thread_{len(st.session_state.conversation_threads) + 1}_{datetime.now().strftime('%H%M%S')}"
    new_thread_name = f"New Conversation {len(st.session_state.conversation_threads) + 1}"
    
    st.session_state.conversation_threads[new_thread_id] = ChatMemory()
    st.session_state.thread_metadata[new_thread_id] = {
        'name': new_thread_name,
        'created_at': datetime.now()
    }
    st.session_state.current_thread_id = new_thread_id
    st.rerun() # Rerun to update the UI with the new thread

def switch_thread(thread_id_to_switch_to):
    st.session_state.current_thread_id = thread_id_to_switch_to
    st.rerun() # Rerun to update the UI with the selected thread's history

def display_model_info():
    """Display current model information"""
    model_info = st.session_state.llm_manager.get_model_info()
    st.markdown(f"""
    <div class="stats-container">
        <h4>🎯 Current Model</h4>
        <p><strong>Provider:</strong> {model_info['provider'].title()}</p>
        <p><strong>Model:</strong> {model_info['model']}</p>
        <p><strong>Status:</strong> {'🟢 Ready' if model_info['status'] else '🔴 Not Available'}</p>
    </div>
    """, unsafe_allow_html=True)

    # Enhanced stats display with HNSW information
    stats = st.session_state.vector_store.get_stats()
    
    st.markdown(f"""
    <div class="stats-container">
        <h4>📈 Knowledge Base Status</h4>
        <p><strong>Documents:</strong> {stats['total_documents']}</p>
        <p><strong>Embeddings:</strong> {stats['index_size']}</p>
        <p><strong>Embedding Shape:</strong> {stats['embeddings_shape']}</p>
        <p><strong>Index Type:</strong> {stats.get('index_type', 'HNSW')}</p>
        <p><strong>HNSW M:</strong> {stats.get('hnsw_m', 'N/A')}</p>
        <p><strong>HNSW efSearch:</strong> {stats.get('hnsw_ef_search', 'N/A')}</p>
    </div>
    """, unsafe_allow_html=True)

def display_control_buttons():
    """Display control buttons"""
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🧹 Clear Documents"):
            st.session_state.vector_store.clear()
            st.session_state.documents_loaded = False
            st.rerun()
    
    with col2:
        if st.button("💬 Clear Chat"):
            st.session_state.chat_memory.clear_history()
            st.session_state.conversation_history = []
            st.session_state.last_evaluation = None
            st.rerun()

def process_documents(uploaded_files, website_url):
    """Process uploaded documents and website content with enhanced feedback"""
    all_documents = []
    
    # Create processing indicator
    progress_container = st.container()
    
    with progress_container:
        st.markdown('<div class="processing-indicator">📚 Processing financial documents with HNSW indexing...</div>', unsafe_allow_html=True)
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_steps = len(uploaded_files or []) + (1 if website_url else 0)
        current_step = 0
        
        # Process uploaded files
        if uploaded_files:
            for i, file in enumerate(uploaded_files):
                status_text.text(f"Processing {file.name}...")
                
                file_extension = os.path.splitext(file.name)[1].lower()
                
                if file_extension == '.pdf':
                    docs = st.session_state.document_processor.process_pdf(file)
                elif file_extension == '.csv':
                    docs = st.session_state.document_processor.process_csv(file) 
                elif file_extension in ['.xlsx', '.xls']:
                    docs = st.session_state.document_processor.process_xlsx(file)
                elif file_extension == '.json':
                    docs = st.session_state.document_processor.process_json(file)
                else:
                    st.warning(f"⚠️ Unsupported file type: {file.name}")
                    continue
                
                if docs:
                    all_documents.extend(docs)
                    st.success(f"✅ Processed: {file.name} ({len(docs)} chunks)")
                
                current_step += 1
                progress_bar.progress(current_step / total_steps)
        
        # Process website URL
        if website_url:
            status_text.text("Processing website content...")
            docs = st.session_state.document_processor.process_website(website_url)
            if docs:
                all_documents.extend(docs)
                st.success(f"✅ Processed website: {len(docs)} chunks")
            
            current_step += 1
            progress_bar.progress(current_step / total_steps)
        
        # Add documents to vector store
        if all_documents:
            status_text.text("Building knowledge base...")
            success = st.session_state.vector_store.add_documents(all_documents)
            if success:
                st.session_state.documents_loaded = True
                
                # Generate and display document summary
                sample_content = " ".join([doc.page_content[:300] for doc in all_documents[:5]])
                summary = st.session_state.llm_manager.generate_response(
                    query="Provide a comprehensive summary of these financial documents including company name, reporting period, key financial highlights, and main business segments.",
                    context=sample_content
                )
                
                progress_bar.progress(1.0)
                status_text.text("✅ Processing complete!")
                
                st.markdown("### 📋 Document Summary")
                st.markdown(f'<div class="welcome-card">{summary}</div>', unsafe_allow_html=True)
                
                time.sleep(1)
                st.rerun()
        else:
            st.error("❌ No documents were successfully processed.")

def display_welcome_message():
    """Display welcome message when no documents are loaded"""
    st.markdown("""
    ### 👋 Welcome to Financial RAG Assistant!
    
    I'm here to help you analyze financial documents from telecom companies. Here's how to get started:
    
    **📤 Upload Your Documents:**
    - 📄 PDF files (Annual/Quarterly reports)
    - 📊 CSV files (Financial data)
    - 📈 Excel files (Spreadsheets)
    - 🌐 Website URLs (Financial pages)
    
    **💬 What You Can Ask:**
    - "*What were the key financial highlights this quarter?*"
    - "*Show me the revenue breakdown by segment*"
    - "*What are the main risks mentioned in the report?*"
    - "*Compare this year's performance with last year*"
    - "*What is the company's debt-to-equity ratio?*"
    
    **🎯 Features:**
    - 🧠 **Smart Analysis:** AI-powered insights from your documents
    - 🔍 **Semantic Search:** Find relevant information instantly  
    - 📚 **Memory:** Maintains conversation context
    - 🎯 **Source Attribution:** Know where answers come from
    
    **Getting Started:**
    1. Upload your financial documents using the sidebar
    2. Click "Process Documents" to build your knowledge base
    3. Start asking questions about your financial data!
    
    *Ready to dive into your financial analysis? Upload some documents to begin! 🚀*
    """)

def display_chat_interface():
    """Display the main chat interface with improved conversation history handling"""
    # Get chat history from memory
    chat_history = st.session_state.chat_memory.get_chat_history()
    
    # Display conversation history
    for i, message in enumerate(chat_history):
        if message['role'] == 'user':
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>👤 You:</strong> {message['content']}
            </div>
            """, unsafe_allow_html=True)
        else:
            # Assistant message
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <strong>🤖 Assistant:</strong><br>{message['content']}
                {f'<div class="source-info">📚 Sources: {", ".join(message.get("sources", []))}</div>' if message.get('sources') else ''}
            </div>
            """, unsafe_allow_html=True)
            
            # Show evaluation metrics only for the most recent assistant response
            if (st.session_state.show_evaluation and st.session_state.last_evaluation and 
                i == len(chat_history) - 1 and message['role'] == 'assistant'):
                st.markdown('<div class="evaluation-panel">', unsafe_allow_html=True)
                TriadMetricsDisplay.display_metrics(
                    st.session_state.last_evaluation,
                    "Response Quality Evaluation"
                )
                st.markdown('</div>', unsafe_allow_html=True)

    # Chat input
    if not st.session_state.processing_question:
        user_input = st.chat_input("Ask me anything about your financial documents...")
        
        if user_input:
            process_user_question(user_input)

def process_user_question(user_input: str):
    """Process user question and generate response"""
    st.session_state.processing_question = True
    
    try:
        st.session_state.chat_memory.add_message('user', user_input)
        
        with st.spinner("🔍 Searching relevant information with HNSW..."):
            if st.session_state.search_mode == 'hybrid':
                relevant_docs = st.session_state.vector_store.get_relevant_documents(
                    user_input, 
                    k=getattr(st.session_state, 'k_docs', 5),
                    use_hybrid=True
                )
            else:
                relevant_docs = st.session_state.vector_store.get_relevant_documents(
                    user_input, 
                    k=getattr(st.session_state, 'k_docs', 5),
                    use_hybrid=False
                )
        
        if relevant_docs:
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            conversation_context = st.session_state.chat_memory.get_conversation_context()
            
            with st.spinner("🤔 Generating response..."):
                response = st.session_state.llm_manager.generate_response(
                    query=user_input,
                    context=context,
                    conversation_history=conversation_context
                )
            
            sources = list(set([doc.metadata.get('source', 'Unknown') for doc in relevant_docs]))
            
            # ⭐ FIXED EVALUATION CALL
            if st.session_state.show_evaluation:
                with st.spinner("📊 Evaluating response quality..."):
                    try:
                        # Extract just the text content from documents
                        retrieved_doc_contents = [doc.page_content for doc in relevant_docs]
                        
                        # ⭐ KEY FIX: Pass parameters correctly
                        evaluation = st.session_state.evaluation_metrics.evaluate_response(
                            answer=response,          # Changed from 'response' to 'answer'
                            query=user_input,
                            context=context,
                            retrieved_docs=retrieved_doc_contents,
                            reference=context  # ⭐ Use context as reference for better relevance
                        )
                        
                        st.session_state.last_evaluation = evaluation
                        
                    except Exception as e:
                        st.error(f"⚠️ Evaluation failed: {str(e)}")
                        st.session_state.last_evaluation = None
            
            st.session_state.chat_memory.add_message('assistant', response, sources)
            
        else:
            response = "I couldn't find relevant information in your documents to answer your question."
            st.session_state.chat_memory.add_message('assistant', response)
            st.session_state.last_evaluation = None
    
    except Exception as e:
        st.error(f"❌ Error processing question: {str(e)}")
        response = "I apologize, but I encountered an error while processing your question."
        st.session_state.chat_memory.add_message('assistant', response)
        st.session_state.last_evaluation = None
    
    finally:
        st.session_state.processing_question = False
        st.rerun()

def display_suggested_questions():
    """Display suggested questions for users"""
    if st.session_state.documents_loaded:
        st.markdown("### 💡 Try asking:")
        
        questions = [
            "What are the key financial metrics for this period?",
            "Show me the revenue trends and growth rates",
            "What are the major risks and challenges mentioned?",
            "How is the company's cash flow position?",
            "What are the main business segments and their performance?",
            "What capital expenditures were made this quarter?"
        ]
        
        cols = st.columns(2)
        for i, question in enumerate(questions):
            with cols[i % 2]:
                if st.button(question, key=f"suggested_{i}"):
                    st.session_state.suggested_question = question
                    st.rerun()
        
        # Handle suggested question
        if hasattr(st.session_state, 'suggested_question'):
            user_input = st.session_state.suggested_question
            delattr(st.session_state, 'suggested_question')
            
            # Process the suggested question
            
            st.session_state.chat_memory.add_message('user', user_input)
            
            relevant_docs = st.session_state.vector_store.get_relevant_documents(user_input, k=5)
            
            if relevant_docs:
                context = "\n\n".join([doc.page_content for doc in relevant_docs])
                conversation_context = st.session_state.chat_memory.get_conversation_context()
                
                response = st.session_state.llm_manager.generate_response(
                    query=user_input,
                    context=context,
                    conversation_history=conversation_context
                )
                
                sources = list(set([doc.metadata.get('source', 'Unknown') for doc in relevant_docs]))
                
                # Evaluate response if enabled
                if st.session_state.show_evaluation:
                    evaluation = st.session_state.evaluation_metrics.evaluate_response(
                        answer=response,
                        query=user_input,
                        context=context
                    )
                    st.session_state.last_evaluation = evaluation
                
                st.session_state.chat_memory.add_message('assistant', response, sources)
            else:
                response = "I couldn't find relevant information to answer this question."
                st.session_state.chat_memory.add_message('assistant', response)
                st.session_state.last_evaluation = None
            
            st.rerun()
def download_chat_history_for_thread(chat_memory_instance: ChatMemory):
    chat_history = chat_memory_instance.get_chat_history()
    if not chat_history:
        st.warning("No chat history to download for this thread.")
        return

    download_content = ""
    for message in chat_history:
        role = message.get('role', 'Unknown').capitalize()
        content = message.get('content', 'No Content')
        timestamp = message.get('timestamp', 'N/A')
        sources = message.get('sources', [])
        
        download_content += f"[{timestamp}] {role}:\n{content}\n"
        if sources:
            download_content += f"Sources: {', '.join(sources)}\n"
        download_content += "-" * 50 + "\n\n" # Separator for readability
    
    st.download_button(
        label="Download Current Chat History",
        data=download_content,
        file_name=f"chat_history_{st.session_state.thread_metadata[st.session_state.current_thread_id]['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain",
        key=f"download_history_button_{st.session_state.current_thread_id}" # Unique key
    )

def main():
    """Main application function"""
    initialize_session_state()
    
    # Header
    st.markdown(f'''
    <div class="main-header">
        <h1>📊 Advanced Financial RAG Assistant</h1>
        <p>AI-powered financial document analysis with advanced evaluation metrics</p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Sidebar for document upload and management
    with st.sidebar:
        st.markdown("### 📁 Document Management")
        
        # Enhanced file upload section
        uploaded_files = st.file_uploader(
            "Upload Financial Documents",
            type=['pdf', 'csv', 'xlsx', 'xls', 'json'],
            accept_multiple_files=True,
            help="Upload PDF annual reports, CSV/Excel financial data, or JSON structured data"
        )
        
        # Website URL input
        website_url = st.text_input(
            "🌐 Financial Website URL:",
            placeholder="https://investor.company.com/annual-report",
            help="Enter URL of financial reports or investor relations pages"
        )

        # Search Configuration Section
        st.markdown("### 🔍 Search Configuration")
        
        # Search mode selection
        search_mode = st.selectbox(
            "Search Mode:",
            options=['hybrid', 'semantic'],
            index=['hybrid', 'semantic'].index(st.session_state.search_mode) if st.session_state.search_mode in ['hybrid', 'semantic'] else 0,
            help="Choose how to search through documents"
        )
        
        if search_mode != st.session_state.search_mode:
            st.session_state.search_mode = search_mode

        # Search mode info
        search_info = {
            'hybrid': "🔄 Combines semantic similarity with keyword matching using HNSW for best results",
            'semantic': "🧠 Pure semantic search using HNSW embeddings for fast and accurate retrieval"
        }
        
        st.markdown(f"""
        <div class="search-mode-info">
            {search_info[search_mode]}
        </div>
        """, unsafe_allow_html=True)
        
        # Search parameters
        with st.expander("⚙️ Search Parameters"):
            k_docs = st.slider("Documents to retrieve:", 1, 10, 5, help="Number of relevant documents to retrieve")
            if search_mode == 'hybrid':
                alpha = st.slider("Semantic vs Keyword weight:", 0.0, 1.0, Config.HYBRID_SEARCH_ALPHA, 0.1, 
                              help="0 = pure keyword, 1 = pure semantic")
                st.session_state.hybrid_alpha = alpha
            st.session_state.k_docs = k_docs
        
        # HNSW Parameters Section
        with st.expander("🔧 HNSW Parameters"):
            ef_search = st.slider("HNSW efSearch:", 10, 500, st.session_state.vector_store.hnsw_ef_search if hasattr(st.session_state.vector_store, 'hnsw_ef_search') else 100, 
                             help="Higher values = better accuracy, slower search")
            if st.button("Update HNSW Parameters"):
                st.session_state.vector_store.update_hnsw_parameters(ef_search=ef_search)
                st.success("✅ HNSW parameters updated!")
        
        # Evaluation settings
        st.markdown("### 📊 Evaluation Settings")
        enable_evaluation = st.checkbox(
            "Enable Response Evaluation",
            value=st.session_state.show_evaluation,
            help="Show detailed evaluation metrics for each response"
        )
        st.session_state.show_evaluation = enable_evaluation
        
        # Process documents button
        if st.button("🔄 Process Documents", type="primary"):
            if uploaded_files or website_url:
                process_documents(uploaded_files, website_url)
            else:
                st.warning("⚠️ Please upload files or enter a website URL first.")
        
        # Model selection section
        st.markdown("### 🤖 AI Model Configuration")
        
        # Get available providers
        available_providers = st.session_state.llm_manager.get_available_providers()
        
        # Provider selection
        enabled_providers = [provider for provider, available in available_providers.items() if available]
        
        if enabled_providers:
            selected_provider = st.selectbox(
                "LLM Provider:",
                options=enabled_providers,
                index=enabled_providers.index(st.session_state.llm_manager.current_provider) if st.session_state.llm_manager.current_provider in enabled_providers else 0,
                help="Select your preferred LLM provider"
            )
            
            # Model selection based on provider
            if selected_provider in Config.LLM_MODELS:
                available_models = list(Config.LLM_MODELS[selected_provider].keys())
                selected_model = st.selectbox(
                    "Model:",
                    options=available_models,
                    index=available_models.index(st.session_state.llm_manager.current_model) if st.session_state.llm_manager.current_model in available_models else 0,
                    help=f"Select a model from {selected_provider}"
                )
                
                # Update model if changed
                if (selected_provider != st.session_state.llm_manager.current_provider or 
                    selected_model != st.session_state.llm_manager.current_model):
                    st.session_state.llm_manager.set_model(selected_provider, selected_model)
                    st.success(f"✅ Switched to {selected_provider}: {selected_model}")
        else:
            st.error("❌ No LLM providers configured. Please set up API keys in your .env file.")
        
        st.subheader("Conversation Threads")

        # Sort threads by creation date, most recent first
        sorted_thread_ids = sorted(
            st.session_state.thread_metadata.keys(),
            key=lambda tid: st.session_state.thread_metadata[tid]['created_at'],
            reverse=True
        )

        # Display the last 3 conversation threads prominently
        st.markdown("##### Recent Threads:")
        recent_threads_to_display = sorted_thread_ids[:3] # Get the 3 most recent IDs

        for thread_id in recent_threads_to_display:
            display_name = st.session_state.thread_metadata[thread_id]['name']
            is_current = (thread_id == st.session_state.current_thread_id)

            # Use a button to switch threads, make current thread visually distinct
            button_label = f"💬 {display_name}"
            if is_current:
                st.button(button_label, key=f"thread_btn_{thread_id}", disabled=True, use_container_width=True)
            else:
                st.button(button_label, key=f"thread_btn_{thread_id}", on_click=switch_thread, args=(thread_id,), use_container_width=True)

        st.markdown("---") # Separator

        # Dropdown for all threads (useful if many threads exist)
        all_thread_options = {
            st.session_state.thread_metadata[tid]['name']: tid 
            for tid in sorted_thread_ids
        }

        # Get current display name for the selected thread
        current_thread_name = st.session_state.thread_metadata[st.session_state.current_thread_id]['name']

        selected_thread_name = st.selectbox(
            "Or select another conversation:",
            options=list(all_thread_options.keys()),
            index=list(all_thread_options.keys()).index(current_thread_name) if current_thread_name in all_thread_options else 0,
            key="thread_select_box",
            on_change=lambda: switch_thread(all_thread_options[st.session_state.selected_thread_box]) # Update current_thread_id
        )
        # Ensure the session_state.current_thread_id is aligned if selectbox changes it
        st.session_state.current_thread_id = all_thread_options[selected_thread_name]


        st.button("➕ New Conversation", on_click=create_new_thread, use_container_width=True)
        st.markdown("---")

        st.subheader("Chat Options")
        # Download history button for the currently active thread
        current_chat_memory = st.session_state.conversation_threads[st.session_state.current_thread_id]
        download_chat_history_for_thread(current_chat_memory)

        # You might also want a clear current thread history button
        if st.button("Clear Current Thread History", help="Clears all messages in the active conversation thread."):
            current_chat_memory.clear_history()
            st.success("Current thread history cleared!")
            st.rerun()
        
        # Display current model info
        display_model_info()
        
        # Control buttons
        display_control_buttons()

    # Main content area
    if not st.session_state.documents_loaded:
        display_welcome_message()
    else:
        display_chat_interface()
    
    # Add suggested questions at the bottom
    display_suggested_questions()

if __name__ == "__main__":
    main()
