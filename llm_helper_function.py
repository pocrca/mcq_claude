from langchain.text_splitter import TokenTextSplitter
from langchain.docstore.document import Document
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader
from prompts import PROMPT_QUESTIONS, REFINE_PROMPT_QUESTIONS
from PyPDF2 import PdfReader
from langchain.vectorstores import FAISS
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceBgeEmbeddings


# Function to extract text from PDF for question generation
def extract_text_from_pdf_for_q_gen(uploaded_file):
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    docs = split_text_q_gen(text)
    return docs

# Function to extract text from PDF for question answering
def extract_text_from_pdf_for_q_answer(uploaded_file):
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    docs = split_text_q_answer(text)
    return docs

# Text splitter when the input text is just text, not documents.
# Used for question generation
def split_text_q_gen(data):
    text_splitter = TokenTextSplitter(chunk_size=50000, chunk_overlap=500)
    texts = text_splitter.split_text(data)
    docs = [Document(page_content=t) for t in texts]
    return docs

# Text splitter when the input text is just text, not documents.
# Used for question answering, vector database
def split_text_q_answer(data):
    text_splitter = TokenTextSplitter(chunk_size=5000, chunk_overlap=200)
    texts = text_splitter.split_text(data)
    docs = [Document(page_content=t) for t in texts]
    return docs

# Function to create a single Tweet
def create_questions(docs, llm):
    question_chain = load_summarize_chain(llm, chain_type="refine", verbose=True, question_prompt=PROMPT_QUESTIONS, refine_prompt=REFINE_PROMPT_QUESTIONS)
    questions = question_chain.run(docs)
    return questions


def create_vectordatabase(docs):
    model_name = "BAAI/bge-small-en-v1.5"
    encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        encode_kwargs=encode_kwargs
    )

    db = FAISS.from_documents(docs, embeddings)
    return db


def convert_to_markdown(text):
    lines = text.strip().split('\n')
    markdown = ""
    for num, line in enumerate(lines):
        line = line.strip().lstrip('1234567890.- ')
        if line:
            # markdown += f"{num+1}. {line}\n"
            markdown += f"{line}\n"
    return markdown