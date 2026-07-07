# config.py
# Fichier passerelle : vector_db.py (Brique 1, ecrit par ma collegue) utilise
# des noms de variables differents de ceux de constants.py (fichier de config
# commun du projet). En attendant d'harmoniser ca avec elle, ce fichier
# fait simplement la traduction entre les deux.

from constants import (
    EMBEDDING_MODEL_NAME as EMBEDDING_MODEL,
    PERSIST_DIRECTORY as CHROMA_PATH,
    COLLECTION_NAME,
)
