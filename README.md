

# 📄 Retrieval-Augmented Generation with LangChain & OpenAI

This project demonstrates a complete implementation of a **Retrieval-Augmented Generation (RAG)** system using **LangChain**, **Chroma vector store**, and **OpenAI's GPT models**. It supports both basic and **conversational memory-aware querying**, providing a robust framework for working with long or complex documents.

---

## 🧠 What is RAG?

**Retrieval-Augmented Generation** enhances LLM responses by first retrieving relevant contextual data from external sources (like documents or databases) and then using that context to generate informed and grounded responses.

---

## 📦 Features

* 🔍 Vector-based document retrieval using **Chroma DB**
* 🧾 Document chunking and embeddings with **OpenAI Embeddings**
* 💬 Conversational memory handling using **LangChain Message History**
* 🤖 Question rephrasing to create context-independent queries
* 🧠 Session-based memory to persist conversations across multiple inputs

---

## 🛠️ Tech Stack

* [LangChain](https://github.com/langchain-ai/langchain)
* [OpenAI GPT](https://platform.openai.com/)
* [Chroma Vector Store](https://www.trychroma.com/)
* [Python Dotenv](https://pypi.org/project/python-dotenv/)
* [RecursiveCharacterTextSplitter](https://docs.langchain.com/docs/components/text-splitters/)

---

## 📁 Project Structure

```bash
├── data/
│   └── be-good.txt             # The input document used for retrieval
├── main.py                     # Main execution script
├── .env                        # API key storage (not committed)
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/langchain-rag-assistant.git
cd langchain-rag-assistant
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your OpenAI API key

Create a `.env` file in the root directory with the following content:

```
OPENAI_API_KEY=your_openai_api_key_here
```

---

## 📚 How It Works

1. **Document Loading**: Loads the document from `./data/be-good.txt`.
2. **Chunking**: Splits the document into overlapping chunks for efficient retrieval.
3. **Embedding & Indexing**: Each chunk is embedded using OpenAI and stored in Chroma DB.
4. **Retrieval Chain**: Sets up LangChain RAG pipeline to retrieve relevant chunks for a user query.
5. **Conversational RAG**: Adds memory/history support to handle follow-up questions using contextual understanding.

---

## 💻 Usage

Run the main script:

```bash
python main.py
```

### Example Output

```
What is this article about?
→ The article discusses...

What was my previous question about?
→ Your last question asked about the general topic of the article.
```

You’ll also see the full conversation history printed at the end.

---

## 🗣️ Session-Based Interaction

This system supports session-based memory. By assigning a `session_id`, the system retains chat history across invocations and can interpret context-aware queries accordingly.

---

## 📌 Notes

* The `.env` file must not be checked into version control (it’s ignored via `.gitignore`).
* You can change the input document by replacing `be-good.txt` with your own `.txt` file in the `data/` directory.

---

## ✅ TODO

* [ ] Add support for PDF, HTML, or web-based loaders
* [ ] Integrate UI with Streamlit or Gradio
* [ ] Support multi-turn summarization
* [ ] Add unit and integration tests

---
