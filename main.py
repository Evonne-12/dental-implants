import streamlit as st
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
from train import lma2
import time
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')

st.set_page_config(page_title="Dental implants QA", page_icon="🤖", layout="wide")   

def display_existing_messages():
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

synonym_dict = {
    "periodontal": "peri-implant"
    
}


def replace_synonyms(text):
    for source, target in synonym_dict.items():
        text = text.replace(source, target)
    return text
def add_user_message_to_session(prompt):
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state["messages"].append({"role": "user", "content": prompt})
display_existing_messages()
embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
db = FAISS.load_local("vector", embeddings)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
llm=lma2()
#llm1=g4all()
if "initial_message_displayed" not in st.session_state:
    time.sleep(2)
    with st.chat_message("assistant"):
        st.markdown("Hello, I am your AI helper. What can I do for you?")
    st.session_state["initial_message_displayed"] = True
    st.session_state["messages"].append(
            {"role": "assistant", "content": "Hello, I am your AI helper. What can I do for you?"})

question = st.chat_input("Ask any question related here")
custom_responses = {
    "Hello": "Hello! I'm doing well, thank you for asking.",
    "hello": "Hello! I'm doing well, thank you for asking.",
    "How are you?": "I'm just a AI helper, so I don't have feelings, but I'm here to assist you!",
    "how are you?": "I'm just a AI helper, so I don't have feelings, but I'm here to assist you!",
    "What's your name?": "I am your AI helper. What can I do for you?",
    "what's your name?": "I am your AI helper. What can I do for you?" 
}
#question = st.text_input("Ask something from the file")    
if question:
    question1 = replace_synonyms(question)
    add_user_message_to_session(question)
    with st.spinner('Wait for it...'):
        if question1 in custom_responses:
          response_text = custom_responses[question1]
        else:
         qa = ConversationalRetrievalChain.from_llm(llm, db.as_retriever(search_kwargs={'k':1}), memory=memory)
         response = qa(question1) 
         response_text=response["answer"]
        with st.chat_message("assistant"):
            message_placeholder = st.empty() 
            if response_text:
              formatted_sentences=[]
              formatted_response = response_text.split("Context:")[0].strip()
              formatted_response = formatted_response.split("Please select one of the following options to answer the question:")[0].strip()
              formatted_response = formatted_response.split("Please answer the following question: ")[0].strip()
              sentences = sent_tokenize(formatted_response)
              if len(sentences)==1:
                 formatted_response="".join(sentences)
              else:
               for sentence in sentences:
            
                formatted_sentence = sentence.strip()
                formatted_sentences.append(formatted_sentence)
        
        
               #formatted_response = "  \n".join(formatted_sentences)
               formatted_response = " ".join(formatted_sentences)

        
              message_placeholder.markdown(f"{formatted_response}")
              #message_placeholder.markdown(response_text)
            else:
                formatted_response = llm(question1, max_tokens=4028)  
                message_placeholder.markdown(formatted_response)
            st.session_state["messages"].append(
            {"role": "assistant", "content": formatted_response})
                
