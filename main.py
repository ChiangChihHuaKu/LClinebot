import sys
import aiohttp

from fastapi import Request, FastAPI, HTTPException

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain

from linebot.v3.webhook import WebhookParser
from linebot import AsyncLineBotApi
from linebot.aiohttp_async_http_client import AiohttpAsyncHttpClient
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,
)

import os
import openai


from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  # read local .env file

# get channel_secret and channel_access_token from your environment variable
channel_secret = os.getenv('ChannelSecret',None)
channel_access_token = os.getenv('ChannelAccessToken',None)
openai.api_key = os.getenv('OPENAI_API_KEY')
if channel_secret is None:
    print('Specify LINE_CHANNEL_SECRET as environment variable.')
    sys.exit(1)
if channel_access_token is None:
    print('Specify LINE_CHANNEL_ACCESS_TOKEN as environment variable.')
    sys.exit(1)

app = FastAPI()
session = aiohttp.ClientSession()
async_http_client = AiohttpAsyncHttpClient(session)
line_bot_api = AsyncLineBotApi(channel_access_token, async_http_client)
parser = WebhookParser(channel_secret)

documents = []
for file in os.listdir("Docs"):
    if file.endswith(".pdf"):
        pdf_path = "./Docs/" + file
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())
    elif file.endswith('.docx') or file.endswith('.doc'):
        doc_path = "./Docs/" + file
        loader = Docx2txtLoader(doc_path)
        documents.extend(loader.load())
    elif file.endswith('.txt'):
        text_path = "./Docs/" + file
        loader = TextLoader(text_path, encoding ="utf-8") #在這裡加上encoding參數否則python會報錯 目前只有txt有這個問題
        documents.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000 , chunk_overlap=10)
documents = text_splitter.split_documents(documents)

vectordb = Chroma.from_documents(documents, embedding=OpenAIEmbeddings())
#create the index in retriever interface
retriever = vectordb.as_retriever(search_type="similarity",search_kwargs={"k":4})
memory = ConversationBufferWindowMemory(k=5)
qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0.2), retriever=retriever , verbose=False , chain_type="stuff")
chat_history= []


# Langchain 串接 OpenAI ，這裡 model 可以先選 gpt-3.5-turbo
llm = ChatOpenAI(temperature=0.9, model='gpt-3.5-turbo')

# 透過 ConversationBufferWindowMemory 快速打造一個具有「記憶力」的聊天機器人，可以記住至少五回。
# 通常來說 5 回還蠻夠的
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=False
)


@app.post("/callback")
async def handle_callback(request: Request):
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = await request.body()
    body = body.decode()

    try:
        events = parser.parse(body, signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")

    for event in events:
        if not isinstance(event, MessageEvent):
            continue
        if not isinstance(event.message, TextMessage):
            continue

        # 將使用者傳來的訊息 event.message.text 當成輸入，等 LangChain 傳回結果。
        query = event.message.text
        result = qa({"question": query + '(用繁體中文回答)', "chat_history": chat_history})
        #ret = conversation.predict(input=event.message.text)

        await line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=result['answer'])

        )
        chat_history.append((query, result['answer']))

    return 'OK'
