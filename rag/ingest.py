from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


PDF_PATH = "data/gcp-learning-guide.pdf"
VECTOR_STORE_PATH = "data/faiss_index"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def build_vector_store():

    print("Loading PDF...")

    loader = PyPDFLoader(PDF_PATH)

    documents = loader.load()

    print(f"Loaded {len(documents)} pages")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    Path(VECTOR_STORE_PATH).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    vector_store.save_local(VECTOR_STORE_PATH)

    print(f"Vector store saved to {VECTOR_STORE_PATH}")


if __name__ == "__main__":
    build_vector_store()