import os 
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_ollama import ChatOllama
from qdrant_client import QdrantClient
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA

import streamlit as st 

class ChatBotManager:
    def __init__(
        self,
        model_name: str = os.path.join(os.getcwd(), "bge-small-en"),
        device: str = "cpu",
        encode_kwargs: dict = {"normalize_embeddings": True},
        llm_model: str = "llama3.2:1b",
        llm_temperature: float = 0.7,
        qdrant_url: str = "localhost:6333",
        collection_name: str = "vector_db",
    ): 
        """
        Initializes the ChatbotManager with embedding models, LLM, and vector store.

        Args:
            model_name (str): The HuggingFace model name for embeddings.
            device (str): The device to run the model on ('cpu' or 'cuda').
            encode_kwargs (dict): Additional keyword arguments for encoding.
            llm_model (str): The local LLM model name for ChatOllama.
            llm_temperature (float): Temperature setting for the LLM.
            qdrant_url (str): The URL for the Qdrant instance.
            collection_name (str): The name of the Qdrant collection.
        """  

        self.model_name = model_name
        self.device = device
        self.encode_kwargs = encode_kwargs
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature 
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name

        # Initialize embeddings

        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name = self.model_name,
            model_kwargs = {"device": self.device},
            encode_kwargs = self.encode_kwargs
        )        

        # Initialize LLM 

        self.llm = ChatOllama(
            model = self.llm_model,
            temperature=self.llm_temperature,
        )
        
       
        
        # Define the prompt template
        self.prompt_template = """Use the following pieces of information to answer the user's question.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context: {context}
        Question: {question}

        Only return the helpful answer. Answer must be detailed and well explained.
        Helpful answer:
        """
     #   st.info(self.prompt_template)
        # Initialize Qdrant Client
        self.client = QdrantClient(
            url=self.qdrant_url,
            prefer_grpc=False
        )

        # Initialize Qdrant vector store
        self.db = Qdrant(
            client = self.client,
            embeddings=self.embeddings,
            collection_name= self.collection_name
        )

        # Initialize the prompt
        self.prompt = PromptTemplate(
            template = self.prompt_template,
            input_variables=['context', 'question']
        )

        # Initialize the retriever
        self.retriever = self.db.as_retriever(search_kwargs={"k": 1})

        # Define chain type kwargs
        self.chain_type_kwargs = {"prompt": self.prompt}

        # Initialize the RetrievalQA chain

        self.qa = RetrievalQA.from_chain_type(
            llm = self.llm,
            chain_type = "stuff",
            
            retriever = self.retriever,
            chain_type_kwargs = self.chain_type_kwargs,
            verbose=False,
            return_source_documents = False
        )

    def get_response(self, query: str) -> str:

        try:
            response = self.qa.invoke(query)

            return response['result']

        except Exception as e:
            st.error(f"An error occured: {e}")
            return "Sorry, I could'nt process your request at the moment."