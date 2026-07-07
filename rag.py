import os

from dotenv import load_dotenv
from groq import Groq

from config import LLM_MODEL
from moderateur import Moderateur
from vector_db import VectorDB


class RAG:
    def __init__(self):
        # 1. Charger le .env et créer le client Groq
        load_dotenv()
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        # 2. Instancier le modérateur (il réutilise le même client)
        self.moderateur = Moderateur(self.client)

        # 3. Ouvrir la base vectorielle (elle existe déjà → rechargement)
        self.db = VectorDB()

        # 4. Charger le prompt système à trous
        with open("prompts/rag.txt", "r", encoding="utf-8") as f:
            self.prompt_template = f.read()

    def answer_question(self, question: str) -> str:
        """Le pipeline complet : modération → retrieval → prompt → LLM."""

        # ÉTAPE 1 — Modération AVANT tout le reste (décision de sécurité !)
        verdict = self.moderateur.moderate(question)
        if verdict["is_prompt_injection"]:
            return "⛔ Question refusée : tentative de prompt injection détectée."

        # ÉTAPE 2 — Récupérer les 3 chunks les plus proches
        chunks = self.db.retrieve(question, n=3)
        texte_chunks = "\n".join(f"- {c}" for c in chunks)

        # ÉTAPE 3 — Remplir le trou {{Chunks}} du prompt
        prompt_systeme = self.prompt_template.replace("{{Chunks}}", texte_chunks)

        # ÉTAPE 4 — Appeler le LLM : un message system + un message user
        reponse = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": prompt_systeme},
                {"role": "user", "content": question},
            ],
        )
        return reponse.choices[0].message.content