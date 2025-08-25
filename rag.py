import os
from dotenv import load_dotenv

# Ensure these imports are at the top
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Set the Google API key from the environment
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

# --- RAG Initialization (Run only once) ---
# Define persistence directory
PERSIST_DIRECTORY = "./chroma_db"

# Check if the vector store already exists
if os.path.exists(PERSIST_DIRECTORY) and os.listdir(PERSIST_DIRECTORY):
    print("Loading existing vector store...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
else:
    print("Vector store not found, creating a new one...")
    # Load document
    loader = TextLoader("data/info.txt")
    document = loader.load()

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    text = text_splitter.split_documents(document)

    # Create embeddings and vector store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma.from_documents(text, embeddings, persist_directory=PERSIST_DIRECTORY)

# --- RAG Chain Setup ---
# Create a prompt template
template = """
You are a helpful assistant to assist the user to share your thoughts about the knowledge and experience of J K Jayanth.
Keep your answers in a clean and professional tone.
Do not answer irrelevant answers. eg: if user says hi just greet the user, if he wants any details about J K Jayanth only then answer about him.
Do not use sentences like - based on the given text or based on the provided text just act like a agent
Do not add any special characters in the output.

Context: {context}

Question: {question}

Helpful Answer:
"""
chat_prompt = PromptTemplate(template=template, input_variables=["context", "question"])

# Create a model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
qa = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=vectorstore.as_retriever(), chain_type_kwargs={"prompt": chat_prompt})

# --- Main RAG function ---
# Change to an async function to work better with FastAPI
async def awake_rag(query):
    result = await qa.ainvoke(query)  # Use ainvoke for asynchronous execution
    return result['result']
