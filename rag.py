# rag.py

import os
from dotenv import load_dotenv
from groq import Groq

from vector_db import VectorDB
from moderateur import Moderateur
from constants import LLM_MODEL_NAME, RAG_SYSTEM_PROMPT_PATH, N_CHUNKS_RETRIEVED


class RAG:
    def __init__(self, chunks=None):
        load_dotenv()

        self.groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.vector_db = VectorDB(chunks=chunks)
        self.moderateur = Moderateur(client=self.groq_client)

        with open(RAG_SYSTEM_PROMPT_PATH, encoding="utf-8") as f:
            self.system_prompt_template = f.read()

    def _build_system_prompt(self, question):
        """Recupere les chunks pertinents et les injecte dans le prompt a trous."""
        documents = self.vector_db.retrieve(question, n=N_CHUNKS_RETRIEVED)
        chunks_text = "\n".join(f"- {doc}" for doc in documents)
        return self.system_prompt_template.replace("{{Chunks}}", chunks_text)

    def answer_question(self, question):
        # ETAPE 1 : moderation, AVANT tout le reste.
        # Si c'est une tentative d'injection, on s'arrete net :
        # ni retrieval, ni appel au LLM principal.
        moderation = self.moderateur.moderate(question)

        if moderation.get("is_prompt_injection"):
            return (
                "Je ne peux pas traiter cette demande : elle a ete identifiee "
                "comme une tentative de manipulation du systeme."
            )

        # ETAPE 2 : le pipeline RAG normal, uniquement si la question est saine
        system_prompt = self._build_system_prompt(question)

        response = self.groq_client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ]
        )

        return response.choices[0].message.content
