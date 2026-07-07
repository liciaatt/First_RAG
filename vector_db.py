import chromadb
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL, CHROMA_PATH, COLLECTION_NAME


class VectorDB:
    def __init__(self, chunks=None):
        # 1. On ouvre un client ChromaDB PERSISTANT :
        #    les données sont sauvegardées sur le disque (dossier chroma_db)
        #    et survivent à l'arrêt du programme
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)

        # 2. L'aiguillage du constructeur (le point clé du TP !) :
        #    la base existe déjà ? on la RECHARGE
        #    sinon, on a des chunks ? on la CRÉE
        #    sinon → erreur explicite
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
        # On charge le modèle d'embedding
        self.model = SentenceTransformer(EMBEDDING_MODEL)

        # On crée la collection en mémorisant le nom du modèle
        # dans ses MÉTADONNÉES (le "détail malin" du TP !)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"embedding_model": EMBEDDING_MODEL},
        )

        # On encode tous les chunks en vecteurs
        embeddings = self.encoder(chunks)

        # On insère tout : un id unique, le texte, son vecteur, une source
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

        # On relit le nom du modèle DANS LES MÉTADONNÉES de la collection,
        # et on charge CE modèle-là (pas celui de config.py !)
        nom_modele = self.collection.metadata["embedding_model"]
        self.model = SentenceTransformer(nom_modele)
        print(f"✅ Base rechargée ({self.collection.count()} chunks, modèle : {nom_modele})")

    def encoder(self, textes):
        """Encode une liste de textes en vecteurs normalisés."""
        return self.model.encode(
            textes,
            batch_size=32,
            normalize_embeddings=True,   # indispensable pour la similarité cosinus
            show_progress_bar=False,
        ).tolist()

    def retrieve(self, question, n=3):
        """Retourne les n chunks les plus proches de la question."""
        vecteur_question = self.encoder([question])
        resultats = self.collection.query(
            query_embeddings=vecteur_question,
            n_results=n,
        )
        return resultats["documents"][0]