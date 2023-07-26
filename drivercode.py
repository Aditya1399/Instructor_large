#Loading the local llm
from transformers import pipeline
from langchain.llms import HuggingFacePipeline

#from dataandembeddingcreation import retriever
from langchain.chains import RetrievalQA
#from addingdetailstoprompts import chat_prompt1
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM

from langchain import PromptTemplate, LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

template="""Question:Recommend best KOL name to the MSL based on the data provided 

"""
system_message_prompt = SystemMessagePromptTemplate.from_template(template)

example_human = HumanMessagePromptTemplate.from_template('''

KOL Tier = Regional
KOL Qualification = MBBS,MD
KOL Experience = 20 TO 23 YEARS
KOL Location = New Jersey
Company Event = Medical Asset Generation-Patient Driven Solutions
Interaction Intent = To discuss about startup would be giving pitches and presenting their medical assets and judges will be giving reviews and how these events will promote innovation in medical field
                                                         ''')

 

example_ai = AIMessagePromptTemplate.from_template('Micheal Smith')

example1_human=HumanMessagePromptTemplate.from_template('''

KOL Tier = International
KOL Qualification = MBBS,MD,PHD
KOL Experience = 11 TO 14 Years of experience
KOL Location = Boston
Company Event = Medical Asset Generation-Patient Driven Solutions
Interaction Intent = To discuss about plan of training and workshop sessions and importance of these sessions for healthcare professionals to develop innovative and patient driven solution


                                                         ''')

example1_ai = AIMessagePromptTemplate.from_template('Johnson Charles')


human_template='''
{question}'''

human_message_prompt = HumanMessagePromptTemplate.from_template(human_template,input_variables=["question"])

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, example_human, example_ai,example1_human,example1_ai, human_message_prompt])







chat_prompt1=chat_prompt.format(question='Recommend KOL name having regional tier')

import pandas as pd


import os

from getpass import getpass
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader


from InstructorEmbedding import INSTRUCTOR
from langchain.embeddings import HuggingFaceInstructEmbeddings

from langchain.document_loaders.csv_loader import CSVLoader

loader=CSVLoader(file_path="nba_msl.csv")
data=loader.load()

#splitting the text into
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(data)

from langchain.embeddings import HuggingFaceInstructEmbeddings
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
#model = INSTRUCTOR('instructor_embedding_large')
instructor_embeddings = HuggingFaceInstructEmbeddings(model_name='Users/1926222/Documents/Dockerize_NextBestAssistant/instructor_embeddingmodel', 
                                                      model_kwargs={"device": "cpu"})

embedding = instructor_embeddings

vectordb = FAISS.from_documents(documents=texts, 
                                 embedding=embedding
                                 )

retriever = vectordb.as_retriever(search_kwargs={"k": 1})

tokenizer=AutoTokenizer.from_pretrained('Users/1926222/Documents/Dockerize_NextBestAssistant/RedPajamaModel')
model=AutoModelForSeq2SeqLM.from_pretrained('Users/1926222/Documents/Dockerize_NextBestAssistant/RedPajamaModel')

pipe = pipeline(
    "text2text-generation",
    model=model, 
    tokenizer=tokenizer,
    max_length=1024,
    temperature=0.3,
    top_p=0.95,
    repetition_penalty=1.15
)

local_llm = HuggingFacePipeline(pipeline=pipe)

# create the chain to answer questions 
#chain_type_kwargs = {"prompt": PROMPT}
qa_chain = RetrievalQA.from_chain_type(llm=local_llm, 
                                  chain_type="stuff", 
                                  retriever=retriever, 
                                  return_source_documents=True,
                                  
                                  )

query=chat_prompt1
llm_response=qa_chain(query)

print(llm_response)