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
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardMarkup, InlineKeyboardButton
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import logging


load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TOKEN = "7805795694:AAGsKBwBxlkoYw9Vs9wfV8wxhzAl3fk1Uwk"
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

bot = Bot(token=TOKEN)
dp = Dispatcher(bot)

main_keyboard = ReplyKeyboardMarkup(resize_keyboard=True)
main_keyboard.add(KeyboardButton('❓ Задать вопрос'))
main_keyboard.add(KeyboardButton('ℹ️ О боте'))
main_keyboard.add(KeyboardButton('📚 Примеры вопросов'))

# примеры вопросов
examples_keyboard = InlineKeyboardMarkup()
examples_keyboard.add(InlineKeyboardButton("Что такое гражданский кодекс?", callback_data="example1"))
examples_keyboard.add(InlineKeyboardButton("Какие бывают виды договоров?", callback_data="example2"))
examples_keyboard.add(InlineKeyboardButton("Что регулирует семейное право?", callback_data="example3"))

def initialize_rag_system():
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
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
                
            # Более надежный способ получения текста
            content = soup.find('article') or soup.find('main') or soup.find('body')
            if content:
                # Удаляем скрипты и стили
                for script in content(['script', 'style']):
                    script.decompose()
                return content.get_text(separator='\n', strip=True)
            return ""
        except Exception as e:
            logger.error(f"Ошибка при загрузке {url}: {e}")
            return ""

    documents = []
    for url in urls:
        content = fetch_website_content(url)
        if content and len(content) > 100:  # Отбрасываем слишком короткие документы
            documents.append(Document(page_content=content))
        
    if not documents:
        raise ValueError("Не удалось загрузить ни одного документа")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="cointegrated/LaBSE-en-ru")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    llm = ChatOpenAI(api_key=OPENAI_KEY, model="gpt-4")

    system_template = """
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
    
    return rag_chain

# Инициализация RAG-системы при старте
rag_chain = initialize_rag_system()

def ask_question(question):
    response = rag_chain.invoke({"input": question})
    return response["answer"]

# Обработчики сообщений
@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    await message.answer(
        "👋 Привет! Я бот-юрист, который может ответить на вопросы по законодательству.\n"
        "Выберите действие:",
        reply_markup=main_keyboard
    )

@dp.message_handler(lambda message: message.text == 'ℹ️ О боте')
async def about_bot(message: types.Message):
    await message.answer(
        "🤖 Я - бот-юрист, созданный для ответов на вопросы по законодательству.\n"
        "Я использую искусственный интеллект для поиска информации в документах.\n"
        "Просто задайте мне вопрос, и я постараюсь помочь!"
    )

@dp.message_handler(lambda message: message.text == '📚 Примеры вопросов')
async def show_examples(message: types.Message):
    await message.answer(
        "Вот несколько примеров вопросов, которые вы можете задать:",
        reply_markup=examples_keyboard
    )

@dp.message_handler(lambda message: message.text == '❓ Задать вопрос')
async def ask_question_prompt(message: types.Message):
    await message.answer("Пожалуйста, введите ваш вопрос:")

@dp.callback_query_handler(lambda c: c.data.startswith('example'))
async def process_example_question(callback_query: types.CallbackQuery):
    example_questions = {
        "example1": "Что такое гражданский кодекс?",
        "example2": "Какие бывают виды договоров?",
        "example3": "Что регулирует семейное право?"
    }
    question = example_questions[callback_query.data]
    answer = ask_question(question)
    await bot.send_message(callback_query.from_user.id, f"❓ Вопрос: {question}\n\n💡 Ответ: {answer}")

@dp.message_handler()
async def handle_question(message: types.Message):
    if message.text not in ['❓ Задать вопрос', 'ℹ️ О боте', '📚 Примеры вопросов']:
        await message.answer("🔍 Ищу ответ на ваш вопрос...")
        try:
            answer = ask_question(message.text)
            await message.answer(answer)
        except Exception as e:
            await message.answer("⚠️ Произошла ошибка при обработке запроса. Попробуйте позже.")
            print(f"Error: {e}")

if __name__ == '__main__':
    try:
        logger.info("Запуск бота...")
        nest_asyncio.apply()
        executor.start_polling(dp, skip_updates=True)
    except Exception as e:
        logger.error(f"Ошибка: {e}")