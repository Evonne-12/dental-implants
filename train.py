from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import Docx2txtLoader
from langchain.llms import CTransformers
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import GPT4All
from huggingface_hub import hf_hub_download


def lma2():


   llm = CTransformers(
                    model="llama-2-7b-chat.ggmlv3.q4_0.bin",
                    model_type='llama', # Model type Llama
                    config={'max_new_tokens': 4028,
                            'context_length':1024,
                            'temperature': 0.8})
   return llm

