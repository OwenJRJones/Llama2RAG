from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import CTransformers
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from PyPDF2 import PdfReader
from datetime import datetime
from io import BytesIO
import chainlit as cl


# Model and embedding models
MODEL_PATH = "./models/llama-2-7b-chat.Q2_K.gguf"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Initialize embedding model
EMBEDDINGS = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={'device': 'cuda'}
)

# Model parameters
INDEX_PATH = "./data/vectorstore"
TEMPERATURE = 0.1
N_GPU_LAYERS = 10
N_BATCH = 256

CONFIG = {
    'max_new_tokens': 512, 
    'context_length': 4096,
    'gpu_layers': N_GPU_LAYERS,
    'batch_size': N_BATCH,
    'temperature': TEMPERATURE
}

# Prompt template
PROMPT_TEMPLATE = """You are a helpful AI assistant. You are kind and respectful to the user. Your job is to answer the question sent by the user in concise and step by step manner. 
If you don't know the answer to a question, please don't share false information.
 
Context: {context}
Question: {question}
Response for Questions asked.
answer:"""

# Load Llama2 model
@cl.cache # Decorator to cache the model before app starts
def load_model(model_path):
    llm = CTransformers(
        model=model_path,
        model_type="llama",
        config=CONFIG
    )
    return llm

# Load the local model into LLM
llm = load_model(MODEL_PATH)

# Handle uplaoded PDF
@cl.on_chat_start
async def factory():
    files = None

    # Wait for user to upload a file
    while files == None:
        files = await cl.AskFileMessage(
            content="""Hello there!
            I'm J.A.R.V.I.S., your personal AI mechanic. I'm here to answer any questions you may have.
            To get started:
                     
1. Upload a pdf file                     
2. Ask any questions about the file""",
            accept={"application/pdf": [".pdf"]},
            max_size_mb=10
        ).send()
    
    # Alert the user the system is ready
    await cl.Message(
        content=f"""Document - `"{files[0].name}"` uploaded, initializing vectorstore..."""
    ).send()

    # Read PDF and convert to text
    file = files[0]

    # Read bytes
    with open(file.path, 'rb') as f:
        content = f.read()

    # Convert contents of PDF to BytesIO stream
    text_stream = BytesIO(content)

    # Create PDFReader object from stream to extract text
    pdf = PdfReader(text_stream)
    pdf_text = ""

    # Extract text from each page of the PDF
    for page in pdf.pages:
        pdf_text += page.extract_text()
    
    # Create embeddings for the uploaded PDF and store in vectorstore
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=10
    )

    # Create "documents" by splitting the PDF text into chunks
    documents = text_splitter.create_documents([pdf_text])

    # Create a FAISS index from the embeddings
    faiss_index = FAISS.from_documents(documents, EMBEDDINGS)

    # Save FAISS index locally
    faiss_index_path = INDEX_PATH + '-temp-index'
    faiss_index.save_local(faiss_index_path)

    # Load FAISS vectorstore with embeddings created and saved earlier
    db = FAISS.load_local(
        faiss_index_path,
        EMBEDDINGS,
        allow_dangerous_deserialization=True
    )

    # Create a prompt using the template
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=['context', 'question']
    )

    # Create a retrieval QA chain
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k': 1}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )

    msg = cl.Message(content="Initializing, please wait...")
    await msg.send()

    msg.content = "System ready. Fire away..."
    await msg.update()
    cl.user_session.set("chain", chain)

    @cl.on_message
    async def main(message):
        start_time = datetime.now()

        chain = cl.user_session.get("chain")

        response = await chain.ainvoke(
            message.content,
            callbacks=[cl.AsyncLangchainCallbackHandler()]
        )

        end_time = datetime.now()
        time_taken = end_time - start_time
        
        print("Total time taken was:", time_taken)

        await cl.Message(content=response['result']).send()
        