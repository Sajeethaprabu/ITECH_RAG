import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


DOCS_FOLDER = "book.pdf"
COLLECTION_NAME = "my_rag_docs"

# ✅ Use lighter model
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen2.5:3b-instruct"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 180
TOP_K = 6
SCORE_THRESHOLD = 0.6

# ✅ Docker-safe Ollama URL
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")

st.set_page_config(page_title="Cryptography RAG", layout="wide")
st.title("Cryptography & Network Security RAG application")
st.markdown("Answering strictly from William Stallings (4th Edition)")


@st.cache_resource
def initialize_system():

    # ✅ Use Qdrant server mode (NOT local path)
    client = QdrantClient(host="qdrant", port=6333)

    embeddings = OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url=OLLAMA_BASE_URL
    )

    # Get embedding dimension dynamically
    test_vector = embeddings.embed_query("test")
    vector_dim = len(test_vector)

    try:
        client.get_collection(COLLECTION_NAME)
    except:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=vector_dim,
                distance=Distance.COSINE,
            ),
        )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )

    # Load PDF only if empty
    if client.get_collection(COLLECTION_NAME).points_count == 0:
        loader = PyPDFLoader(DOCS_FOLDER)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

        splits = splitter.split_documents(docs)
        vector_store.add_documents(splits)

    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "score_threshold": SCORE_THRESHOLD,
            "k": TOP_K,
        },
    )

    llm = ChatOllama(
        model=LLM_MODEL,
        temperature=0.2,
        num_ctx=4096,  # reduce context to save memory
        base_url=OLLAMA_BASE_URL
    )

    prompt_template = """
You are a senior academic expert in Cryptography and Network Security.

Answer ONLY using the provided context.

If insufficient context, respond exactly with:
"The provided document sections do not contain enough information to fully answer this."

Context:
{context}

Question:
{question}

Comprehensive Answer:
"""

    prompt = PromptTemplate.from_template(prompt_template)

    chain = (
        {
            "context": retriever | (lambda docs: "\n\n".join(
                f"Page: {d.metadata.get('page','?')}\n{d.page_content}"
                for d in docs
            )),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


chain = initialize_system()

query = st.text_input("Ask a question:")

if st.button("Generate Answer") and query:
    with st.spinner("Thinking..."):
        answer = chain.invoke(query)
        st.write(answer)