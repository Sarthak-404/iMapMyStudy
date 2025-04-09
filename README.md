# 📚 iMapMyStudy – AI-Powered Learning Assistant

Welcome to **iMapMyStudy**, your one-stop solution for personalized and intelligent learning assistance. This project combines cutting-edge Large Language Models (LLMs), web scraping, embeddings, and interactive features to help students navigate and understand their study material effortlessly.

🔗 **Deployed Link**: [http://imapmystudy.vercel.app/](http://imapmystudy.vercel.app/)

---

## 🚀 Features

- **📄 Document Q&A**
  - Upload a PDF and ask questions directly.
  - Language detection (English or Hindi) for context-aware responses.
  
- **🌐 Web Page Summarizer**
  - Provide a URL to generate a comprehensive summary of the webpage.

- **📝 Quiz Generator**
  - Upload a study document and auto-generate multiple-choice quizzes with answers.

- **✅ Quiz Evaluator**
  - Submit answers and get instant evaluation with correct answers and scores.

- **📅 Study Plan Generator**
  - Get a structured study plan with topic breakdowns, brief explanations, and estimated completion times.

- **🧠 Mind Map Generator**
  - Generate markdown-based mind maps from documents for visual learning using Mermaid.js format.

- **💬 Gemini LLM Chat**
  - A lightweight chatbot powered by Google Gemini Pro to interact in natural language.

---

## 🛠️ Tech Stack

- **Backend**: Flask (Python)
- **LLMs**: Groq (LLaMA3-8B), Google Gemini
- **Embeddings**: Google Generative AI Embeddings
- **Vector Store**: FAISS
- **Document Parsing**: Langchain, PyPDFLoader
- **Frontend Deployment**: Vercel

---

## 📂 API Endpoints

### `GET /`
- Returns a welcome message.

### `POST /document_qa/`
- **Params**: `uploaded_file`, `prompt1`
- Upload a PDF and ask a question.

### `POST /web_summarizer`
- **Params**: `input` (URL)
- Summarizes the content of a webpage.

### `POST /generate_quiz`
- **Params**: `file`, `num` (number of questions)
- Generates a multiple-choice quiz.

### `POST /submit_quiz`
- **Params**: JSON with `quiz` and `answers`
- Submits the quiz answers and returns results.

### `POST or GET /generate_study_plan/`
- **Params**: `uploaded_file`
- Creates a personalized study plan.

### `POST /process_pdf`
- **Params**: `file` or `document_text`
- Generates a Mermaid.js-based mind map.

### `POST or GET /gemini_llm_chat/`
- **Params**: `input`
- Chatbot powered by Gemini.

---

## 🌍 Deployment

- **Frontend**: [http://imapmystudy.vercel.app/](http://imapmystudy.vercel.app/)
- **Backend**: Flask app running on AWS

---

## 🧪 Running Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/imapmystudy.git
   cd imapmystudy
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. Set up your `.env` file:
   ```env
   FLASK_SECRET_KEY=your_secret_key
   GROQ_API_KEY=your_groq_key
   GOOGLE_API_KEY=your_google_api_key
   ```

4. Run the server:
   ```bash
   python app.py
   ```

---

## 👥 Authors

Built with by Sarthak Sachan
