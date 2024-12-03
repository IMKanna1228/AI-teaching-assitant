# PDF Conversational AI with LangChain and FAISS

![image](https://github.com/user-attachments/assets/b035e6f4-ad49-4f7a-8eca-76b6a3614323)


## **Overview**
This project allows you to interact with PDF documents using a conversational AI powered by OpenAI's GPT model. It utilizes **LangChain** for conversational retrieval and **FAISS** as a vector store for storing embeddings. You can upload a PDF, extract text, and query the document in a chat-like interface using a **Streamlit** app.

## **Features**
- Extract text from PDF files.
- Split text into manageable chunks for processing.
- Generate vector embeddings and store them in a FAISS database.
- Create a conversational retrieval system for interactive Q&A.
- Customized chat interface for enhanced user experience.

## **Requirements**
- Python 3.7 or higher.
- OpenAI API key (stored in a `.env` file).

## **Installation**

1. **Set Environment Variables**:
   - Create a file named `.env` in the root directory of your project.
   - Add the following line to the `.env` file:
     ```plaintext
     OPENAI_API_KEY=your_openai_api_key
     ```

2. **Install Dependencies**:
   Install all required Python packages by running:
   ```bash
   pip install -r requirements.txt
