from vector_db import VectorDB
from corpus import CHUNKS

# Premier lancement : crée la base / lancements suivants : la recharge
db = VectorDB(chunks=CHUNKS)

# Les 5 questions de test du TP
questions = [
    "Quelle est la couleur du chat de Bob ?",
    "Combien mesure la tour de Villeneuve ?",
    "Que mange la sorcière Margarita ?",
    "Quel est le métier du dragon Firmin ?",
    "Combien coûtent les croissants de Gaspard ?",
]

for q in questions:
    print(f"\n❓ {q}")
    for i, chunk in enumerate(db.retrieve(q, n=3), start=1):
        print(f"   {i}. {chunk}")