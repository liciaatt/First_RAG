# constants.py

EMBEDDING_MODEL_NAME = "distiluse-base-multilingual-cased-v2"
LLM_MODEL_NAME = "openai/gpt-oss-120b"

PERSIST_DIRECTORY = "data/chroma_db"
COLLECTION_NAME = "corpus_rag"
N_CHUNKS_RETRIEVED = 3

RAG_SYSTEM_PROMPT_PATH = "prompts/rag_system_prompt.txt"