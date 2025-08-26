import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

# --- RAG INITIALIZATION ---
# This part is a one-time setup
print("Starting RAG initialization...")

try:
    # 1. Load documents from the text file
    loader = TextLoader("data/info.txt")
    document = loader.load()

    # 2. Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    text = text_splitter.split_documents(document)
    print(f"Split {len(text)} chunks from the document.")

    # 3. Create embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # 4. Create vector store from the chunks
    # Note: We are creating it in-memory, which is safe for Render's ephemeral filesystem.
    vectorstore = Chroma.from_documents(text, embeddings)
    print("Vector store created successfully.")

    # 5. Define the prompt template
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

    # 6. Initialize the LLM and the RetrievalQA chain
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type='stuff', 
        retriever=vectorstore.as_retriever(), 
        chain_type_kwargs={"prompt": chat_prompt}
    )
    print("RAG chain initialized.")

except Exception as e:
    print(f"CRITICAL RAG INITIALIZATION ERROR: {e}")
    qa_chain = None # Set to None to handle errors gracefully

# --- RAG FUNCTION ---
def awake_rag(query):
    """
    This function processes the query using the pre-initialized RAG chain.
    """
    if qa_chain is None:
        return "Sorry, the chatbot is not available. There was an internal error."

    try:
        result = qa_chain.invoke(query)
        return result['result']
    except Exception as e:
        print(f"Error during RAG invocation: {e}")
        return "Sorry, something went wrong while processing your request."
