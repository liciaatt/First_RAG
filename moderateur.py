import json

from groq import Groq

# Le modèle de modération : famille "safeguard" de Groq
MODERATION_MODEL = "openai/gpt-oss-safeguard-20b"


class Moderateur:
    def __init__(self, client: Groq):
        # On réutilise le client Groq créé par le RAG (pas besoin d'en refaire un)
        self.client = client

        # Le prompt système vit dans son fichier texte
        with open("prompts/moderateur.txt", "r", encoding="utf-8") as f:
            self.prompt_systeme = f.read()

    def moderate(self, question: str) -> dict:
        """Retourne {"is_prompt_injection": True/False}"""
        reponse = self.client.chat.completions.create(
            model=MODERATION_MODEL,
            messages=[
                {"role": "system", "content": self.prompt_systeme},
                {"role": "user", "content": question},
            ],
            response_format={"type": "json_object"},  # force une sortie JSON
        )

        # La réponse est une CHAÎNE de caractères JSON → on la transforme en dictionnaire
        texte_json = reponse.choices[0].message.content
        return json.loads(texte_json)