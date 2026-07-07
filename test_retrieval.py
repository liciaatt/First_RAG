from rag import RAG
import csv

def load_chunks_from_csv(path):
    chunks = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            chunks.append(row["text"])
    return chunks

chunks = load_chunks_from_csv("data/05_corpus_rag.csv")
rag = RAG(chunks=chunks)

questions = [
    "Quelle est la couleur du chat de Bob ?",
    "Comment s'appelle la tortue de Carla ?",
    "Que mange le lapin de Fatou ?",
    "Quelle est la capitale du Japon ?",
    "Le chat de Bob est-il vert ?"
]

for q in questions:
    print(f"\n--- {q} ---")
    docs = rag.vector_db.retrieve(q, n=3)
    for d in docs:
        print(" -", d)
