from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import  PromptTemplate

from dotenv import load_dotenv
import asyncio
import os



load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

loader = TextLoader("data/info.txt")
document = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
text = text_splitter.split_documents(document)

#embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# vectorstores
vectorstore = Chroma.from_documents(text, embeddings,persist_directory="./chroma_db")

# creating a prompt

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

chat_prompt =   PromptTemplate(template=template, input_variables=["context", "question"])
# create a model 
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
qa = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=vectorstore.as_retriever(), chain_type_kwargs={"prompt":chat_prompt})


def awake_rag(query):
    result = qa.invoke(query)
    return result['result']
