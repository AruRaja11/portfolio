from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from rag import awake_rag

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def serve_portfolio(request: Request):
    return templates.TemplateResponse("portfolio.html", {"request": request})

class Query(BaseModel):
    question: str

@app.post("/query/")
async def query_rag(query: Query):
    try:
        print(f"Received query: {query.question}")
        answer = awake_rag(query.question)
        print(f"Sending answer: {answer}")
        return {'answer': answer}
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        return {'error': str(e)}

@app.get("/api")
def read_root():
    return {"message": "RAG API is running!"}

@app.get('/favicon.ico', include_in_schema=False)
async def get_favicon():
    return ""
