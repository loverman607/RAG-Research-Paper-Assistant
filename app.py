import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from huggingface_hub import login
from langchain.document_loaders import PyPDFLoader, ArxivLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from tempfile import NamedTemporaryFile
import torch
import os

CHROMA_DB_DIR = "chroma_db"

# --- Authenticate with Hugging Face Hub ---
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)

# --- Load LLM and Embeddings ---
@st.cache_resource
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf",
        quantization_config=quant_config,
        device_map="auto"
    )

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return HuggingFacePipeline(pipeline=pipe)

@st.cache_resource
def load_embeddings():
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# --- Function to load or create persistent ChromaDB vectorstore ---
def get_or_update_vectorstore(docs, source_name="unknown"):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs_split = text_splitter.split_documents(docs)
    if not docs_split:
        st.warning("No content was found in the document. Please check the file or arXiv ID.")
        return
    for doc in docs_split:
        doc.metadata["source"] = source_name
    embeddings = load_embeddings()
    db = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
    db.add_documents(docs_split)
    db.persist()
    return db

# --- Load existing vectorstore for global query ---
def get_vectorstore():
    embeddings = load_embeddings()
    return Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)

# --- Streamlit UI ---
st.title("🧠 RAG Chatbot for Research Papers")
st.write("Upload a PDF or provide an arXiv ID to add to the searchable database.")

uploaded_file = st.file_uploader("Upload a research PDF", type="pdf")
arxiv_id = st.text_input("Or enter an arXiv ID (e.g., 2301.12345):")
query = st.text_input("Ask a question across documents:")

if uploaded_file or arxiv_id:
    with st.spinner("Loading document..."):
        if uploaded_file:
            with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            loader = PyPDFLoader(tmp_path)
            source_name = uploaded_file.name
        else:
            loader = ArxivLoader(query=f"id:{arxiv_id}", load_max_docs=1, pdf=True)
            source_name = arxiv_id

        docs = loader.load()
        get_or_update_vectorstore(docs, source_name=source_name)

        if uploaded_file:
            os.remove(tmp_path)  # Clean up temporary file

# Dropdown to filter by source document
vectorstore = get_vectorstore()
all_sources = list({doc.metadata["source"] for doc in vectorstore.similarity_search("", k=100) if "source" in doc.metadata})
selected_source = st.selectbox("Select a document to query", ["All documents"] + all_sources)

if query:
    if selected_source != "All documents":
        retriever = vectorstore.as_retriever(search_kwargs={"filter": {"source": selected_source}})
    else:
        retriever = vectorstore.as_retriever()

    qa_chain = RetrievalQA.from_chain_type(
        llm=load_llm(),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    with st.spinner("Generating answer..."):
        result = qa_chain({"query": query})
        st.markdown("### Question:")
        st.write(query)
        st.markdown("### Helpful Answer:")
        
        raw_answer = result["result"]
        
        # Split the response at "Question:" and take the last part
        parts = raw_answer.split("Question:")
        if len(parts) > 1:
            answer = parts[-1].strip()
            # Further clean if there's additional structure
            answer = answer.split("Helpful Answer:")[-1].strip()
        else:
            answer = raw_answer.strip()
        
        st.write(answer)

# --- License Notice ---
st.markdown("---")
st.caption("This app uses the LLaMA 2 model from Meta. Usage complies with Meta's non-commercial license. Learn more at [Meta's LLaMA license page](https://ai.meta.com/resources/models-and-libraries/llama-downloads/).")
