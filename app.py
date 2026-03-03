import os
import pandas as pd
import gradio as gr
from datasets import load_dataset

from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory


GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

SAMPLE_SIZE  = 550000

df = pd.read_csv('amazon_products_merged_shuffled.csv', nrows=SAMPLE_SIZE)
print(df.head())

data = []
for index, row in df.iterrows():
    item = f"""
    Product Name: {row['title']}
    Category: {row['category_name']}
    Price: ${row['price']:.2f} (List Price: ${row['listPrice']:.2f})
    Rating: {row['stars']} stars — {row['reviews']:,} reviews
    Best Seller: {'Yes' if row['isBestSeller'] else 'No'}
    Bought Last Month: {row['boughtInLastMonth']:,} times
    Product URL: {row['productURL']}
    Payment Methods: Credit Card, Debit Card, Installments
    """
    data.append(item)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
faiss_vectorstore = FAISS.from_texts(texts=data, embedding=embedding_model)
faiss_retriever = faiss_vectorstore.as_retriever()


groq_model = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=GROQ_API_KEY)


SYSTEM_PROMPT = """
You are WiseCart AI, a smart, friendly, and helpful shopping assistant.
Your main responsibilities:
- Provide product recommendations strictly based on the data available.
- Recommend at least two relevant products per user request when possible.
- Explain clearly why each recommended product fits the user's needs, preferences, and budget.
- Compare options when helpful (price, features, performance, value).
- Suggest reliable places to buy each product (online or local stores) and possible payment methods (cash, card, installments, digital wallets, etc.).
- If the requested product or information is not present in the data, inform the user politely that you don’t have information about it.
- Respond naturally to greetings, thank-yous, or casual messages in a friendly and polite way, without providing product recommendations unless asked.
Response style guidelines:
- Keep answers clear, concise, and easy to read.
- Use bullet points, numbered lists, or sections to organize your response.
- Focus on practical benefits, value, and usability for the user.
- Maintain a friendly, approachable tone.
- Avoid making assumptions or adding information that is not in the RAG data.
- Always be polite and engaging, even in casual interactions.
Your goal:
- Help the user make informed, confident purchasing decisions using only the information in the RAG knowledge base.
- Ensure transparency: if you lack the data, clearly communicate it rather than guessing.
- Keep conversations pleasant and friendly when users send greetings or thanks.
"""


store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("""
    Data:
    {data}
    Question:
    {question}
    """),
])

parser = StrOutputParser()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
    
from langchain_core.runnables import RunnablePassthrough

def build_faiss_query(message: str, chat_history) -> str:
    """Always combine recent chat history + current question for FAISS retrieval."""
    messages = chat_history.messages
    if not messages:
        return message

    recent = messages[-4:]
    parts = []
    for m in recent:
        role = "User" if m.type == "human" else "Assistant"
        parts.append(f"{role}: {m.content}")
    parts.append(f"User: {message}")
    return "\n".join(parts)


def generate_answer(message: str, session_id: str) -> str:
    try:
        chat_history = get_session_history(session_id)

        faiss_query = build_faiss_query(message, chat_history)
        docs = faiss_retriever.invoke(faiss_query)
        retrieved_data = "\n\n".join(doc.page_content for doc in docs)

        chain = (
            {
                'data'    : RunnableLambda(lambda _: retrieved_data),
                'question': RunnablePassthrough(),
                'history' : RunnableLambda(lambda _: chat_history.messages),
            }
            | prompt
            | groq_model
            | parser
        )

        response = chain.invoke(message)

        chat_history.add_user_message(message)
        chat_history.add_ai_message(response)

        return response
    except Exception as e:
        return f"⚠️ Something went wrong: {str(e)}"


def user_submit(user_message: str, history: list, session_id: str):
    if not user_message.strip():
        return "", history
    history = history + [{"role": "user", "content": user_message}]
    history = history + [{"role": "assistant", "content": generate_answer(user_message, session_id)}]
    return "", history


def clear_chat(session_id: str):
    if session_id in store:
        del store[session_id]
    return []


APP_NAME = "Wise Cart Assistant"
APP_DESCRIPTION = "AI shopping assistant that helps you discover and compare products based on your needs, budget, and preferences."

with gr.Blocks(title=APP_NAME) as demo:
    with gr.Row():
        with gr.Column(scale=1, min_width=250):
            gr.Markdown(f"# 🛒 {APP_NAME}")
            gr.Markdown(APP_DESCRIPTION)
            gr.Markdown("---")
            gr.Markdown("### 💡 Example Requests")
            for ex in [
                "I want a phone from 100$ to 1000$",
                "I want vacuum cleaners with high rating",
                "Suggest a budget friendly bag for travelling",
            ]:
                gr.Markdown(f"- {ex}")

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=500, show_label=False)
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Describe what you are looking for...",
                    label="Your Request",
                    scale=7,
                )
                send_btn = gr.Button("Send 🚀", variant="primary", scale=1)
            clear = gr.Button("🗑️ Clear Chat")

            msg.submit(user_submit,     [msg, chatbot], [msg, chatbot])
            send_btn.click(user_submit, [msg, chatbot], [msg, chatbot])
            clear.click(lambda: [], None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
