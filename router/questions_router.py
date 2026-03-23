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
    try:
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

        docs = results.get("documents", [])

        if not docs or not docs[0]:
            context = 'Sem contexto relevante.'
        else:
            context = "\n".join(docs[0])

        response = client.responses.create(
            model="gpt-4.1-mini",
            input=f"""
            Use o contexto abaixo para responder.

            Context:
            {context}

            Question:
            {question}
            """
        )

        return {"answer": response.output_text}

    except Exception as e:
        print("🔥 ERROR:", e)
        return {"error": str(e)}
