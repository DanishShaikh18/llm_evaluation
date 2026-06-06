from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class SimpleRAG:
    def __init__(self, knowledge_base_path: str):
        self.knowledge_base_path = knowledge_base_path

        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0
        )

        self.vector_store = self._build_vectorstore()

    def _build_vectorstore(self):
        text = Path(self.knowledge_base_path).read_text(
            encoding="utf-8"
        )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )

        chunks = splitter.split_text(text)

        return FAISS.from_texts(
            texts=chunks,
            embedding=self.embeddings
        )

    def generate_answer(self, question: str) -> str:

        docs = self.vector_store.similarity_search(
            question,
            k=3
        )

        context = "\n\n".join(
            doc.page_content for doc in docs
        )

        prompt = PromptTemplate.from_template(
            """
You are a question-answering assistant.

Answer ONLY using the provided context.

If the answer is not available in the context,
say:

"I could not find the answer in the provided knowledge base."

Context:
{context}

Question:
{question}

Answer:
"""
        )

        final_prompt = prompt.format(
            context=context,
            question=question
        )

        response = self.llm.invoke(final_prompt)

        return response.content.strip()