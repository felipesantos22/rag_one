import uvicorn
from fastapi import FastAPI
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware

load_dotenv()

from router.questions_router import router as questions_router

app = FastAPI(title="Rag the traveler")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(questions_router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
