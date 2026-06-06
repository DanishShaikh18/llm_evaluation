from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()

VECTOR_STORE_PATH = "data/faiss_index"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class SimpleRAG:

    def __init__(self):

        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )

        self.vector_store = FAISS.load_local(
            VECTOR_STORE_PATH,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0
        )

    def generate_answer(self, question: str) -> str:

        docs = self.vector_store.similarity_search(
            question,
            k=3
        )

        context = "\n\n".join(
            doc.page_content
            for doc in docs
        )

        prompt = f"""
You are a GCP learning assistant.

Use ONLY the provided context.

If the answer is not found in the context,
say:

"I could not find the answer in the provided document."

Context:
{context}

Question:
{question}

Answer:
"""

        response = self.llm.invoke(prompt)

        return response.content.strip()