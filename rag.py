# rag.py

import os
from dotenv import load_dotenv
from groq import Groq

from vector_db import VectorDB
from constants import LLM_MODEL_NAME, RAG_SYSTEM_PROMPT_PATH, N_CHUNKS_RETRIEVED


class RAG:
    def __init__(self, chunks=None):
        load_dotenv()

        self.groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.vector_db = VectorDB(chunks=chunks)

        with open(RAG_SYSTEM_PROMPT_PATH, encoding="utf-8") as f:
            self.system_prompt_template = f.read()

    def _build_system_prompt(self, question):
        """Recupere les chunks pertinents et les injecte dans le prompt a trous."""
        # vector_db.retrieve() renvoie directement une liste de textes
        # (cf. implementation de la Brique 1 : resultats["documents"][0])
        documents = self.vector_db.retrieve(question, n=N_CHUNKS_RETRIEVED)
        chunks_text = "\n".join(f"- {doc}" for doc in documents)
        return self.system_prompt_template.replace("{{Chunks}}", chunks_text)

    def answer_question(self, question):
        system_prompt = self._build_system_prompt(question)

        response = self.groq_client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ]
        )

        return response.choices[0].message.content