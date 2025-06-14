import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Load API key from environment file
_ = load_dotenv(find_dotenv())
api_key = os.getenv("OPENAI_API_KEY")

# Initialize language model
language_model = ChatOpenAI(model="gpt-3.5-turbo-0125")

# Load and preprocess document
doc_loader = TextLoader("./data/be-good.txt")
raw_documents = doc_loader.load()

# Split document into manageable parts
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
document_chunks = splitter.split_documents(raw_documents)

# Create vector store for semantic search
embedding_fn = OpenAIEmbeddings()
vector_store = Chroma.from_documents(documents=document_chunks, embedding=embedding_fn)
doc_retriever = vector_store.as_retriever()

# Prompt template for answering questions
base_system_prompt = (
    "You are a helpful assistant for answering user queries. "
    "Refer only to the provided context to formulate your answer. "
    "If unsure, say you don't know. Limit answers to three sentences."
    "\n\n{context}"
)

qa_prompt_template = ChatPromptTemplate.from_messages([
    ("system", base_system_prompt),
    ("human", "{input}"),
])

# Basic retrieval-augmented QA pipeline
doc_chain = create_stuff_documents_chain(language_model, qa_prompt_template)
basic_rag = create_retrieval_chain(doc_retriever, doc_chain)

# Execute first question
initial_response = basic_rag.invoke({"input": "What is this article about?"})
print("\n----------\n")
print("What is this article about?")
print("\n----------\n")
print(initial_response["answer"])

# Follow-up question
follow_up = basic_rag.invoke({"input": "What was my previous question about?"})
print("\n----------\n")
print("What was my previous question about?")
print("\n----------\n")
print(follow_up["answer"])

# Prompt for converting chat queries to standalone questions
contextual_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Transform the latest user message into a self-contained question, "
     "using the prior chat history for reference. Do not answer, just rephrase."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

# History-aware retriever
smart_retriever = create_history_aware_retriever(language_model, doc_retriever, contextual_prompt)

# Enhanced QA chain with chat history awareness
conversational_qa_prompt = ChatPromptTemplate.from_messages([
    ("system", base_system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

contextual_doc_chain = create_stuff_documents_chain(language_model, conversational_qa_prompt)
advanced_rag = create_retrieval_chain(smart_retriever, contextual_doc_chain)

# Maintain session-based chat memory
chat_memory = []
user_input = "What is this article about?"
first_reply = advanced_rag.invoke({"input": user_input, "chat_history": chat_memory})

chat_memory.extend([
    HumanMessage(content=user_input),
    AIMessage(content=first_reply["answer"]),
])

next_input = "What was my previous question about?"
second_reply = advanced_rag.invoke({"input": next_input, "chat_history": chat_memory})

print("\n----------\n")
print("What was my previous question about?")
print("\n----------\n")
print(second_reply["answer"])

# Define message history store
memory_storage = {}

def fetch_chat_history(session_key: str) -> BaseChatMessageHistory:
    if session_key not in memory_storage:
        memory_storage[session_key] = ChatMessageHistory()
    return memory_storage[session_key]

# Build stateful RAG pipeline
session_rag = RunnableWithMessageHistory(
    advanced_rag,
    get_session_history=fetch_chat_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)

# First query in session
session_response_1 = session_rag.invoke(
    {"input": "What is this article about?"},
    config={"configurable": {"session_id": "001"}}
)
print("\n----------\n")
print("What is this article about?")
print("\n----------\n")
print(session_response_1["answer"])

# Second query in same session
session_response_2 = session_rag.invoke(
    {"input": "What was my previous question about?"},
    config={"configurable": {"session_id": "001"}}
)
print("\n----------\n")
print("What was my previous question about?")
print("\n----------\n")
print(session_response_2["answer"])

# Display full conversation log
print("\n----------\n")
print("Conversation History:")
print("\n----------\n")

for msg in memory_storage["001"].messages:
    speaker = "AI" if isinstance(msg, AIMessage) else "User"
    print(f"{speaker}: {msg.content}\n")

print("\n----------\n")
