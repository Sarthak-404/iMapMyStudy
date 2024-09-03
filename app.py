from flask import Flask, request, jsonify
from langdetect import detect
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
import time
import shutil
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

@app.route("/", methods=["GET"])
def home():
    return "Welcome to the Flask API!"

@app.route('/favicon.ico')
def favicon():
    return '', 204


# Prompt templates
prompt_hindi = ChatPromptTemplate.from_template("""
दिए गए संदर्भ के आधार पर प्रश्नों का उत्तर दें।
कृपया प्रश्न के आधार पर 100 शब्दों से अधिक का सबसे सटीक उत्तर प्रदान करें।
<context>
{context}
<context>
प्रश्न: {input}
""")

prompt_english = ChatPromptTemplate.from_template("""
Answer the questions based on the provided context only.
Please provide the most accurate response of at least 100 words based on the question.
If it is a mathematics question and not given in context then solve it by your own method and give correct answer.
<context>
{context}
<context>
Questions: {input}
""")


def vector_embedding(uploaded_pdf):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    with open("temp_uploaded_file.pdf", "wb") as f:
        shutil.copyfileobj(uploaded_pdf, f)
    
    loader = PyPDFLoader("temp_uploaded_file.pdf")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(docs)
    vectors = FAISS.from_documents(final_documents, embeddings)
    doc_lang = detect(final_documents[0].page_content)
    os.remove("temp_uploaded_file.pdf")
    return vectors, final_documents, doc_lang


@app.route("/document_qa/", methods=["POST"])
def document_qa():
    if 'uploaded_file' in request.files:
        uploaded_file = request.files['uploaded_file']
        prompt1 = request.form.get('prompt1')
        vectors, final_documents, doc_lang = vector_embedding(uploaded_file)

        question_lang = detect(prompt1)
        if doc_lang == "hi" and question_lang == "hi":
            selected_prompt = prompt_hindi
        else:
            selected_prompt = prompt_english

        llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")
        document_chain = create_stuff_documents_chain(llm, selected_prompt)
        retriever = vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        response_time = time.process_time() - start

        return jsonify({
            "response_time": response_time,
            "answer": response['answer']
        })

    return jsonify({"error": "No file uploaded"})


@app.route("/generate_study_plan/", methods=["POST"])
def generate_study_plan():
    if 'uploaded_file' in request.files:
        uploaded_file = request.files['uploaded_file']
        vectors, final_documents, doc_lang = vector_embedding(uploaded_file)

        if doc_lang == "hi":
            study_plan_prompt = ChatPromptTemplate.from_template("""
            दिए गए संदर्भ के आधार पर एक अध्ययन योजना बनाएं। निम्नलिखित शामिल करें:
            1. महत्वपूर्ण विषय और उपविषय।
            2. प्रत्येक विषय का संक्षिप्त विवरण।
            3. प्रत्येक विषय को पूरा करने के लिए आवश्यक अनुमानित समय।
            <context>
            {context}
            <context>
            """)
        else:
            study_plan_prompt = ChatPromptTemplate.from_template("""
            Based on the context provided, create a study plan. Include the following:
            1. Important topics and subtopics.
            2. A brief description of each topic.
            3. Estimated time required to complete each topic.
            <context>
            {context}
            <context>
            """)

        llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")
        retriever = vectors.as_retriever()
        study_plan_chain = create_stuff_documents_chain(llm, study_plan_prompt)
        study_plan_retrieval_chain = create_retrieval_chain(retriever, study_plan_chain)
        study_plan_response = study_plan_retrieval_chain.invoke({'input': "Generate a study plan"})

        return jsonify({
            "study_plan": study_plan_response['answer']
        })

    return jsonify({"error": "No file uploaded"})


@app.route("/gemini_llm_chat/", methods=["POST"])
def gemini_llm_chat():
    input_text = request.form.get('input')
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel("gemini-pro")
    chat = model.start_chat(history=[])

    response = chat.send_message(input_text, stream=True)

    chat_history = []
    for chunk in response:
        chat_history.append({"role": "Bot", "text": chunk.text})

    return jsonify({"chat_history": chat_history})


@app.route("/youtube_summarizer/", methods=["POST"])
def youtube_summarizer():
    from youtube_transcript_api import YouTubeTranscriptApi

    youtube_link = request.form.get('youtube_link')
    language = request.form.get('language')

    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    prompt = """You are a YouTube video summarizer. You will be taking the transcript text
    and summarizing the entire video and providing the important summary in points
    within 250 words. Please provide the summary of the text given here:  """

    language_code = {"English": "en", "Hindi": "hi", "Japanese": "ja"}[language]

    def extract_transcript_details(youtube_video_url, language_code):
        try:
            video_id = youtube_video_url.split("=")[1]
            transcript_text = YouTubeTranscriptApi.get_transcript(video_id, languages=[language_code])
            transcript = " ".join([i["text"] for i in transcript_text])
            return transcript
        except Exception as e:
            return f"Error retrieving transcript: {str(e)}"

    def generate_gemini_content(transcript_text, prompt):
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt + transcript_text)
        return response.text

    transcript_text = extract_transcript_details(youtube_link, language_code)

    if "Error" not in transcript_text:
        summary = generate_gemini_content(transcript_text, prompt)
        return jsonify({"summary": summary})
    else:
        return jsonify({"error": transcript_text})


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
