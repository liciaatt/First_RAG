# vector_db.py

import chromadb
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL, CHROMA_PATH, COLLECTION_NAME


class VectorDB:
    def __init__(self, chunks=None):
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)

        noms_collections = [c.name for c in self.client.list_collections()]

        if COLLECTION_NAME in noms_collections:
            print("📂 Base existante trouvée → rechargement")
            self._recharger()
        elif chunks is not None:
            print("🆕 Pas de base → création")
            self._creer(chunks)
        else:
            raise ValueError(
                "Aucune base sur le disque et aucun chunk fourni : "
                "impossible de démarrer. Fournissez des chunks pour créer la base."
            )

    def _creer(self, chunks):
        """Crée la collection et indexe tous les chunks."""
        self.model = SentenceTransformer(EMBEDDING_MODEL)

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"embedding_model": EMBEDDING_MODEL},
        )

        embeddings = self.encoder(chunks)

        self.collection.add(
            ids=[f"chunk_{i}" for i in range(len(chunks))],
            documents=chunks,
            embeddings=embeddings,
            metadatas=[{"source": "corpus_test"} for _ in chunks],
        )
        print(f"✅ {len(chunks)} chunks indexés")

    def _recharger(self):
        """Recharge une base existante SANS réindexer."""
        self.collection = self.client.get_collection(name=COLLECTION_NAME)

        nom_modele = self.collection.metadata["embedding_model"]
        self.model = SentenceTransformer(nom_modele)
        print(f"✅ Base rechargée ({self.collection.count()} chunks, modèle : {nom_modele})")

    def encoder(self, textes):
        """Encode une liste de textes en vecteurs normalisés."""
        return self.model.encode(
            textes,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

    def retrieve(self, question, n=3):
        """Retourne les n chunks les plus proches de la question."""
        vecteur_question = self.encoder([question])

        resultats = self.collection.query(
            query_embeddings=vecteur_question,
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )

        if not isinstance(resultats, dict):
            print("⚠️ DIAGNOSTIC — type inattendu renvoye par collection.query() :")
            print("TYPE:", type(resultats))
            print("CONTENU:", resultats)
            raise TypeError(
                f"collection.query() a renvoye un {type(resultats)} au lieu d'un dict. "
                "Voir le diagnostic ci-dessus."
            )

        return resultats["documents"][0]
