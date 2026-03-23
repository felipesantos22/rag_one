from pypdf import PdfReader
from openai import OpenAI
from cdb import get_collection

client = OpenAI()

reader = PdfReader("docs/manual.pdf")

text = ""

for page in reader.pages:
    text += page.extract_text() or ""

chunks = [text[i:i + 500] for i in range(0, len(text), 500)]

collection = get_collection()

for i, chunk in enumerate(chunks):
    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=chunk
    )

    collection.add(
        documents=[chunk],
        embeddings=[emb.data[0].embedding],
        ids=[str(i)]
    )
