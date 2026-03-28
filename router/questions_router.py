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
            Você é um assistente que responde perguntas com base nas informações fornecidas.

            Regras:
            - NÃO mencione a palavra "contexto"
            - NÃO diga frases como "com base no contexto"
            - Responda de forma natural como um especialista
            - Se não souber, diga: "Não encontrei essa informação"
            - Fale em primeira pessoa, como se fosse um agente do site

            Informações:
            {context}

            Pergunta:
            {question}
            """
        )

        return {"answer": response.output_text}

    except Exception as e:
        print("🔥 ERROR:", e)
        return {"error": str(e)}
