from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from rag import awake_rag

# Initialize FastAPI app
app = FastAPI()

# Mount the 'static' directory to serve files like CSS, JS, and images.
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize Jinja2Templates to serve HTML files from the 'templates' directory.
templates = Jinja2Templates(directory="templates")

# Add CORS middleware to allow cross-origin requests from your frontend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for simplicity in this example
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# This route serves your portfolio.html file when a user visits the root URL.
@app.get("/")
async def serve_portfolio(request: Request):
    return templates.TemplateResponse("portfolio.html", {"request": request})

# This is your existing API endpoint for querying the RAG model.
class Query(BaseModel):
    question: str

@app.post("/query/")
async def query_rag(query: Query):
    try:
        print(f"Received query: {query.question}")
        # The key change: await the async function from rag.py
        answer = await awake_rag(query.question)
        print(f"Sending answer: {answer}")
        return {'answer': answer}
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        return {'error': str(e)}

# An additional endpoint to indicate the API is running, accessible at '/api'.
@app.get("/api")
def read_root():
    return {"message": "RAG API is running!"}

# A small endpoint to handle favicon requests gracefully.
@app.get('/favicon.ico', include_in_schema=False)
async def get_favicon():
    return ""
