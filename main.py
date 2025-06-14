import os
import asyncio
import requests
import nest_asyncio
from urllib.error import HTTPError
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from aiogram import Bot, Dispatcher, types, executor
from langchain_huggingface import HuggingFaceEmbeddings

from dotnev import load_dotenv


load_dotenv()

TOKEN = os.getenv('TOKEN')
OPENAI_KEY = os.getenv('OPENAI_API_KEY')

urls = []
base_url = 'https://www.consultant.ru/document/cons_doc_LAW_10699/'
responseq = requests.get(base_url)
html_code = responseq.text
web_code = BeautifulSoup(html_code, 'html.parser')

def page_search(web_code, base_url):
    links = set()
    for link in web_code.find_all('a', href=True):
        full_url = urljoin(base_url, link['href'])
        parsed = urlparse(full_url)
        if parsed.netloc == urlparse(base_url).netloc:
            if not parsed.fragment:
                links.add(full_url)
    return sorted(list(links))

urls = page_search(web_code, base_url)

def fetch_website_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    content = soup.find(class_= "nx-w-full nx-min-w-0 nx-max-w-6xl nx-px-6 nx-pt-4 md:nx-px-12")
    text = []
    if content is not None:
      for element in content.find_all(['p', 'h1', 'h2', 'h3']):
          if element.name == 'p':
              text.append(element.text)
          elif element.name in ['h1', 'h2', 'h3']:
              text.append(f"\n## {element.text.strip()}\n")

      return '\n'.join(text)
    else:
      return '\n'.join('')

documents = [Document(page_content=fetch_website_content(url)) for url in urls]



text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600, chunk_overlap=100, add_start_index=True
)
chunks = text_splitter.split_documents(documents)


embeddings = HuggingFaceEmbeddings(model_name="cointegrated/LaBSE-en-ru")

vectorstore = FAISS.from_documents(chunks, embeddings)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

llm = ChatOpenAI(api_key=OPENAI_KEY, model="gpt-4o-mini")

# system_template = f"""Ты помощник по вопросам-ответам. Используй следующую контекстную информацию,
# чтобы ответить на вопрос. Если в контексте нет ответа, ответь 'Не знаю ответа на вопрос'.
# Используй максимум три предложения и будь точным но кратким.
# Если тебя благодарят, скажи 'Пожалуйста, обращайтесь!'"""

system_template = f"""
Ты помощник по вопросам-ответам. Используй следующую контекстную информацию,
            чтобы ответить на вопрос. Если в контексте нет ответа, ответь 'Не знаю ответа на вопрос'.
            Используй максимум три предложения и будь точным но кратким.
"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("human", """Контекстная информация:

        {context}

        Вопрос: {input}
    """),
])

document_chain = create_stuff_documents_chain(llm, qa_prompt)


rag_chain = create_retrieval_chain(retriever, document_chain)

def ask_question(question):
    response = rag_chain.invoke({"input": question})
    return response["answer"]


nest_asyncio.apply()

bot = Bot(token=TOKEN)
dp = Dispatcher(bot)


@dp.message_handler()
async def chat_with_gpt(message: types.Message):
    if message.text != "/start":
        response = ask_question(message.text)
        await message.answer(response)
    else:
        await message.answer("Приветствую! Готов ответить на Ваши вопросы.")


executor.start_polling(dp, skip_updates=True)