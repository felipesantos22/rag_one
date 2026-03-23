import chromadb

db = chromadb.PersistentClient(path="./chroma_db")

def get_collection():
    return db.get_or_create_collection(name="docs")