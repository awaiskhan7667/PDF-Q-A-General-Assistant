from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI  # 👈 instead of langchain.llms.OpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
import os
load_dotenv()  # take environment variables from .env.
api_key = os.getenv("OPENAI_API_KEY")
# Step 1: Define the prompt

prompt = PromptTemplate.from_template("Answer the following question clearly and concisely:\n\n{question}")

# Step 2: Define the LLM
llm = ChatOpenAI(
    temperature=0.7,
    openai_api_key=api_key,
    model="gpt-3.5-turbo"
    )

# Step 3: Create the chain
chain = LLMChain(llm=llm, prompt=prompt)

# Step 4: Run it
user_question = "How are you ?"
result = chain.invoke({"question": user_question}) 
print(result["text"].strip()) 





# Load API key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Global variables
vectorstore = None
qa_chain = None
chat_history = []
llm = None  # ✅ Define globally

def init_llm():
    global llm
    if llm is None:
        llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")

def upload_pdf(file_path):
    global vectorstore, qa_chain

    try:
        init_llm()  # ✅ Ensure LLM is initialized

        # Load the PDF
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)

        # Embeddings + FAISS
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vectorstore = FAISS.from_documents(chunks, embeddings)

        retriever = vectorstore.as_retriever()
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory
        )
        print("✅ qa_chain created:", qa_chain is not None)
        return "✅ PDF uploaded and indexed successfully!"
    except Exception as e:
        return f"❌ Error: {str(e)}"

def ask_question(query):
    global qa_chain
    if qa_chain is None:
        return "❗ Please upload a PDF first."
    try:
        answer = qa_chain.run(query)
        return answer
    except Exception as e:
        return f"❌ Error: {str(e)}"

def general_chatbot_response(user_input):
    try:
        init_llm()

        # Build chat history string (last 5 turns for example)
        history_limit = 5
        history_prompt = ""
        if chat_history:
            recent_history = chat_history[-history_limit:]
            history_prompt = "\n\n".join([f"{q}\n{a}" for q, a in recent_history])

        # Compose the full prompt
        prompt = (
            "You are a helpful assistant. Here's the recent conversation:\n\n"
            f"{history_prompt}\n\n"
            f"User: {user_input}\nAssistant:"
        )
        response = llm.invoke(prompt)  # works because llm is global now
        chat_history.append(("You: " + user_input, "Bot: " + str(response.content)))
        return response.content
    
    except Exception as e:
        return f"❌ Error: {str(e)}"
    
def clear_chat_history():
    global chat_history
    chat_history = []
    return "🧹 Chat history cleared!"


def get_chat_history():
    if not chat_history:
        return "No chat history yet."
    return "\n\n".join([f"{q}\n{a}" for q, a in chat_history])




# -------- Gradio UI --------
# import gradio as gr
# with gr.Blocks() as demo:
#     gr.Markdown("## 🧠 Chatbot App (PDF Q&A + General Assistant)")

#     with gr.Tab("📄 Chat with PDF"):
#         with gr.Row():
#             file_input = gr.File(label="Upload PDF", type="filepath", file_types=[".pdf"])
#             upload_output = gr.Textbox(label="Upload Status", lines=1)
#         file_input.change(upload_pdf, inputs=file_input, outputs=upload_output)

#         with gr.Row():
#             pdf_question = gr.Textbox(label="Ask a question about the PDF")
#             pdf_answer = gr.Textbox(label="Answer", lines=5)
#         pdf_question.submit(general_chatbot_response, inputs=pdf_question, outputs=pdf_answer)

#     with gr.Tab("💬 General Q&A Chatbot"):
#         with gr.Row():
#             user_question = gr.Textbox(label="Ask anything")
#             general_answer = gr.Textbox(label="Bot Answer", lines=5)
#         user_question.submit(general_chatbot_response, inputs=user_question, outputs=general_answer)
#     with gr.Row():
#         show_history_btn = gr.Button("📜 Show Chat History")
#         history_output = gr.Textbox(label="Chat History", lines=10)
#         user_question.submit(general_chatbot_response, inputs=user_question, outputs=general_answer)
#         show_history_btn.click(get_chat_history, inputs=[], outputs=history_output)
#         clear_history_btn = gr.Button("🧹 Clear Chat History")
#         clear_history_btn.click(clear_chat_history, inputs=[], outputs=history_output)

# demo.launch()

import gradio as gr

with gr.Blocks(css=".gr-button {font-size: 16px !important}") as demo:
    gr.Markdown(
        """
        # 🧠 AI Chatbot App  
        Chat with your PDF documents or ask general knowledge questions!  
        """
    )

    with gr.Tab("📄 Chat with PDF"):
        with gr.Group():
            with gr.Row():
                file_input = gr.File(
                    label="📂 Upload PDF",
                    type="filepath",
                    file_types=[".pdf"]
                )
                upload_output = gr.Textbox(label="Upload Status", lines=1)
            file_input.change(upload_pdf, inputs=file_input, outputs=upload_output)

            with gr.Row():
                pdf_question = gr.Textbox(
                    label="🔎 Ask a question about the PDF",
                    placeholder="e.g., What is the main idea of section 2?"
                )
            pdf_answer = gr.Textbox(label="📖 Answer", lines=6)
            pdf_question.submit(ask_question, inputs=pdf_question, outputs=pdf_answer)

    with gr.Tab("💬 General Q&A Chatbot"):
        with gr.Group():
            with gr.Row():
                user_question = gr.Textbox(
                    label="💡 Ask anything",
                    placeholder="e.g., What is AI?"
                )
            general_answer = gr.Textbox(label="🤖 Bot Answer", lines=6)
            user_question.submit(general_chatbot_response, inputs=user_question, outputs=general_answer)

        with gr.Row():
            show_history_btn = gr.Button("📜 Show Chat History")
            history_output = gr.Textbox(label="Chat History", lines=10)
            show_history_btn.click(get_chat_history, inputs=[], outputs=history_output)

            clear_history_btn = gr.Button("🧹 Clear Chat History")
            clear_history_btn.click(clear_chat_history, inputs=[], outputs=history_output)

demo.launch()
