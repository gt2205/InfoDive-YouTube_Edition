

!pip install -qU langchain-google-genai

!pip install -qU langchain-google-vertexai

!pip install -qU langchain

!pip install -qU langchain-chroma

!pip install -qU langchainhub

# Commented out IPython magic to ensure Python compatibility.
# %pip install --upgrade --quiet  youtube-transcript-api

import os
from google.colab import userdata

os.environ["GOOGLE_API_KEY"] = userdata.get('gemini')

from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-pro")

from langchain_community.document_loaders import YoutubeLoader

from langchain import hub

prompt = hub.pull("rlm/rag-prompt")

loader = YoutubeLoader.from_youtube_url(
    "https://www.youtube.com/watch?v=U9BfED6P4DU", add_video_info=False
)

pages=loader.load()

from langchain_text_splitters import CharacterTextSplitter

text_splitter = CharacterTextSplitter(chunk_size=20, chunk_overlap=10)
docs = text_splitter.split_documents(pages)

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
db = Chroma.from_documents(docs, embeddings)

retriever = db.as_retriever(search_kwargs={"k": 2})

from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

rag_chain = (
    {"context": retriever,  "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

query = "tell me 3 facts about porsche?"
rag_chain.invoke(query)
