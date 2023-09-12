import os

from dotenv import load_dotenv
import streamlit as st
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import  CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, user_template, bot_template
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings

def init():
    load_dotenv()
    if os.getenv('OPENAI_API_KEY') is None or os.getenv('HUGGINGFACEHUB_API_TOKEN') is None:
        print("KEYS NOT SET!")
    else:
        print("API KEYS SET!")

urls = ['http://www.paulgraham.com/greatwork.html',
        'http://www.paulgraham.com/getideas.html']

embedding_model_name = os.environ.get('EMBEDDING_MODEL_NAME')

def load_urls(urls):
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()
    return data

def get_text_chunks(data):
    text_splitter = CharacterTextSplitter(separator='\n',
                                          chunk_size=1000,
                                          chunk_overlap=200,
                                          length_function=len)
    text_chunks = text_splitter.split_documents(data)
    return text_chunks

def get_vector_store(text_chunks):
    #embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(text_chunks, embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    # llm = HuggingFaceHub(repo_id='google/flan-t5-xxl',
    #                      model_kwargs={'temperature':0.5, 'max_length':512})
    llm = OpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm,
                                                               retriever=vectorstore.as_retriever(),
                                                               memory=memory)
    return conversation_chain


def handle_user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    init()
    st.set_page_config(page_title='Chat With Your Site', page_icon=":chatbot:")
    st.write(css, unsafe_allow_html=True)
    st.header("Chat to your website")
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    user_question = st.text_input("Please ask any question? ")
    if user_question:
        handle_user_input(user_question)



    with st.sidebar:
        st.title("LLM chatapp using LangChain")
        st.markdown('''
        This app is an LLM powered chatbot built using:
        - [Steamlit](https://streamlit.io/)
        - [OpenAI](https://www.openai.com)
        ''')

        if st.button("Start"):
            with st.spinner("Processing..."):
                data = load_urls(urls)
                text_chunks = get_text_chunks(data)
                print(len(text_chunks))
                vectorestore = get_vector_store(text_chunks)

                #create a conversation chain
                st.session_state.conversation = get_conversation_chain(vectorestore)
                st.success('Completed!')

if __name__ == '__main__':
    main()