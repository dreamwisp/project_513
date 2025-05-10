import os
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import ollama
import sqlite3
import pandas as pd 
import numpy as np
import contractions
import re, json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_cohere import CohereEmbeddings 
import faiss
from langchain_core.documents.base import Document
import chromadb
from chromadb.config import Settings
import os
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.llms.ollama import Ollama
from langchain.schema.output_parser import StrOutputParser
import csv
import json 
class FilesManager:
    def __init__(self, json_file):
        self.json_file = json_file
        self.collected_data = {}
        self.failed_users = []
        self.stored_users= []
        if json_file and os.path.exists(json_file):
            self.collected_data = self.load_data(json_file)
        else:
            self.collected_data = {"usernames": {}} 

    def load_data(self, json_file):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f) 
            for username in data["usernames"].keys():
             
                self.stored_users.append( username )
            print("username sample : ", self.stored_users[0])                                                    
        return data
    
    def safe_json(self, response):
        if isinstance(response , list):
            new_response = {}
            for item in response:
                try:
                    new_response.update(json.loads(item))
                    
                except json.JSONDecodeError:
                    with open('failures.csv', 'a',) as csvfile:
                        response = ', '.join(response)
                        csvfile.write(response.replace('\n', ' ') + '\n')
                    return False
                

        
            return new_response
        
    def store_user(self, response):  
        username, response_txt  = response
        new_item = self.safe_json(response_txt)
        if new_item:
            # add to the existing json file 
            self.collected_data["usernames"][username]= new_item
            with open(self.json_file, 'w', encoding='utf-8') as f:
                json.dump(self.collected_data, f, indent=4, ensure_ascii=False)
        else:
            self.failed_users.append(response)
