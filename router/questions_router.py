import os
from fastapi import APIRouter
from openai import OpenAI
from cdb import get_collection
from schema.questions_schema import QuestionRequest

router = APIRouter(prefix="/questions", tags=["Questions"])


def get_client():
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@router.post("")
def ask(data: QuestionRequest):
    client = get_client()

    question = data.question
    collection = get_collection()

    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    )

    results = collection.query(
        query_embeddings=[emb.data[0].embedding],
        n_results=3
    )

    context = "\n".join(results["documents"][0])

    print("CONTEXT:", context)

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=f"""
        Use o contexto abaixo para responder.

        Context:
        {context}

        Questions:
        {question}
        """
    )

    return {"answer": response.output_text}
