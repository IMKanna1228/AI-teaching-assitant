# PDF Conversational AI with LangChain and FAISS

![image](https://github.com/user-attachments/assets/e0985bb6-83fd-441c-a56a-e4fe7b03b917)


## **Overview**
This project provides an AI-powered conversational system to interact with PDF documents. Leveraging OpenAI's GPT model, **LangChain** for conversational retrieval, and **FAISS** as a vector store for embeddings, this application enables users to upload a PDF, extract its content, and engage in an interactive Q&A session via a streamlined **Streamlit** interface.

## **Features**
- Extract text from uploaded PDF files.
- Split text into manageable chunks for efficient processing.
- Generate vector embeddings and store them in a FAISS database for retrieval.
- Create a conversational retrieval system for interactive Q&A with the document.
- User-friendly chat interface with customizable templates.

---

## **Requirements**
- Python 3.7 or higher.
- OpenAI API key (stored in a `.env` file).

---

## **Installation and Setup**

1. **Set Up Environment Variables**:
   - Create a `.env` file in the root directory of your project.
   - Add your OpenAI API key to the file:
     ```plaintext
     OPENAI_API_KEY=your_openai_api_key
     ```

2. **Install Dependencies**:
   Install the required Python packages by running:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the HTML Templates Setup**:
   Before launching the main application, set up the chat templates by executing:
   ```bash
   python htmlTemplates.py
   ```

4. **Run the Streamlit Application**:
   Start the application using Streamlit:
   ```bash
   streamlit run app.py
   ```

---

## **Usage**

### **Step 1: Upload a PDF**
- In the Streamlit app, use the file uploader to upload a PDF document.

### **Step 2: Processing the PDF**
- The system extracts text from the uploaded PDF and splits it into chunks for efficient retrieval.

### **Step 3: Interactive Chat**
- Once processing is complete, you can interact with the PDF content by typing questions in the chat box.
- Example:
  ```plaintext
  You: What is the main topic of the document?
  Assistant: The document primarily discusses the principles of AI and its applications.
  ```

### **Step 4: Exit**
- To stop the chat session, type `exit`.

---

## **Project Architecture**

1. **PDF Upload and Text Extraction**:
   - The app reads and extracts text from uploaded PDF files using `PyPDF2`.

2. **Text Chunking**:
   - The extracted text is split into smaller chunks using `LangChain`'s `CharacterTextSplitter`.

3. **Embeddings Generation**:
   - The text chunks are embedded into vector space using `OpenAIEmbeddings`.

4. **FAISS Vector Store**:
   - The embeddings are stored in a FAISS database for quick and efficient retrieval.

5. **Conversational Retrieval Chain**:
   - A LangChain-based conversational retrieval chain connects the vector database to the chatbot interface.

6. **Streamlit Chat Interface**:
   - Users interact with the chatbot through a web-based interface powered by Streamlit.

---

## **Project Structure**

```plaintext
.
├── app.py                # Main Streamlit application
├── htmlTemplates.py      # Contains the CSS and HTML templates for the chat interface
├── requirements.txt      # List of dependencies
├── README.md             # Documentation
├── .env                  # API key for OpenAI (not included in the repository)
```

---

## **Features and Customizations**

1. **Temperature Setting**:
   - The chatbot's creativity can be adjusted by modifying the `temperature` parameter in the `app.py`.

2. **Chunk Size and Overlap**:
   - Adjust `chunk_size` and `chunk_overlap` in the `CharacterTextSplitter` to handle documents of varying lengths.

3. **CSS and HTML Customization**:
   - Modify `htmlTemplates.py` for personalized chatbot styles.

---

## **Example Workflow**

### Uploading and Interacting with a PDF

1. **Upload**:
   - Drag and drop a PDF or select one using the file uploader in the Streamlit interface.

2. **Interact**:
   - Ask questions such as:
     ```plaintext
     You: What is the primary focus of the document?
     Assistant: The document is focused on the implementation of AI in education.
     ```

3. **Exit**:
   - To stop the interaction, simply type `exit`.

---

## **Dependencies**

Add the following to your `requirements.txt` file:
```plaintext
streamlit
python-dotenv
PyPDF2
langchain
openai
faiss-cpu
```

Install all dependencies using:
```bash
pip install -r requirements.txt
```

---

## **License**
This project is licensed under the MIT License.

---

## **Acknowledgments**
- **OpenAI**: For GPT-powered conversational AI.
- **LangChain**: For providing tools for conversational retrieval.
- **FAISS**: For efficient vector-based storage and retrieval.
- **Streamlit**: For an easy-to-use interactive interface.
```

This **README.md** provides a comprehensive explanation of your project, including setup, usage, and customization options, while maintaining a user-friendly structure. Let me know if you need any further modifications!
