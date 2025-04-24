import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from huggingface_hub import login
from langchain.document_loaders import PyPDFLoader, ArxivLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from tempfile import NamedTemporaryFile
import torch
import os

CHROMA_DB_DIR = "chroma_db"

# --- Authenticate with Hugging Face Hub ---
# DeepSeek model is public — no login needed

# --- Load LLM and Embeddings ---
@st.cache_resource
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-llm-7b-chat")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    model = AutoModelForCausalLM.from_pretrained(
        "deepseek-ai/deepseek-llm-7b-chat",
        quantization_config=quant_config,
        device_map="auto"
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True,
        top_p=0.95
    )
    from langchain.llms import HuggingFacePipeline
    return HuggingFacePipeline(pipeline=pipe)

@st.cache_resource
def load_embeddings():
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# --- Function to load or create persistent ChromaDB vectorstore ---
def get_or_update_vectorstore(docs, source_name="unknown"):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs_split = text_splitter.split_documents(docs)
    st.info(f"🧩 Split into {len(docs_split)} chunks from {source_name}")
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
        st.success(f"✅ Loaded {len(docs)} document(s) from: {source_name}")
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

    prompt_template = PromptTemplate.from_template(
        "IMPORTANT: NEVER SAY 'based on the context' or 'according to the documents' - just answer naturally as if you knew the information.\n\n"
        "Context:{context}\n\n"
        "Question: {question}\n\n"
        "Helpful Answer:"
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=load_llm(),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt_template}
    )

    with st.spinner("Generating answer..."):
        result = qa_chain({"query": query})
        st.markdown("### Question:")
        st.write(query)
        st.markdown("### Helpful Answer:")
        
        raw_answer = result["result"]
        parts = raw_answer.split("Question:")
        if len(parts) > 1:
            answer = parts[-1].strip()
            answer = answer.split("Helpful Answer:")[-1].strip()
        else:
            answer = raw_answer.strip()
        st.write(answer)



# --- License Notice ---
st.markdown("---")
st.caption("This app uses the DeepSeek LLM (7B Chat) model released under the MIT license. Learn more at [DeepSeek on Hugging Face](https://huggingface.co/deepseek-ai/deepseek-llm-7b-chat)..")
# --- Credits ---
st.markdown("---")
st.caption("**Credits:** This app integrates components from [Meta LLaMA 2](https://ai.meta.com/resources/models-and-libraries/llama-downloads/), [LangChain](https://github.com/langchain-ai/langchain), [Hugging Face Transformers](https://huggingface.co/docs/transformers), [ChromaDB](https://www.trychroma.com/), and [Streamlit](https://streamlit.io/).")
