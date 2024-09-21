import random
from flask import Flask, request, jsonify, send_file, session
from flask_cors import CORS
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
import io
import shutil
from dotenv import load_dotenv
import google.generativeai as genai
import uuid
import logging
load_dotenv()

app = Flask(__name__)
CORS(app)

app.secret_key = os.getenv('FLASK_SECRET_KEY')
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

vector_cache = {}

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
Answer the question in a way that a teacher should address.
<context>
{context}
<context>
Questions: {input}
""")



#  Vector Embedding of the document
def vector_embedding(uploaded_pdf=None, document_text=None):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    if uploaded_pdf:
        with open("temp_uploaded_file.pdf", "wb") as f:
            shutil.copyfileobj(uploaded_pdf, f)
        
        loader = PyPDFLoader("temp_uploaded_file.pdf")
        docs = loader.load()
        os.remove("temp_uploaded_file.pdf")
    else:
        docs = [{"page_content": document_text}]
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(docs)
    vectors = FAISS.from_documents(final_documents, embeddings)
    doc_lang = detect(final_documents[0].page_content)
    return vectors, final_documents, doc_lang



# Document QA
@app.route("/document_qa/", methods=["POST", "GET"])
def document_qa():
    if request.method == "POST":
        if 'uploaded_file' in request.files:
            uploaded_file = request.files['uploaded_file']
            vectors, final_documents, doc_lang = vector_embedding(uploaded_pdf=uploaded_file)
            
            # Store the vectors in the cache with a unique ID
            cache_id = str(uuid.uuid4())  # Generate a unique ID for the file
            vector_cache[cache_id] = vectors
            session['cache_id'] = cache_id  # Store the cache ID in the session
            session['stored_doc_lang'] = doc_lang

            return jsonify({"message": "File uploaded and vectors prepared successfully."})

        return jsonify({"error": "No file uploaded"})

    elif request.method == "GET":
        if 'cache_id' not in session or 'stored_doc_lang' not in session:
            return jsonify({"error": "File not uploaded or vectors not prepared."})

        cache_id = session['cache_id']
        vectors = vector_cache.get(cache_id)  # Retrieve vectors from the cache
        
        if not vectors:
            return jsonify({"error": "Vectors not found."})

        doc_lang = session['stored_doc_lang']
        prompt1 = request.args.get('prompt1')
        
        if not prompt1:
            return jsonify({"error": "No prompt provided in the query."})

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





# Quiz generator 
@app.route('/generate_quiz', methods=['POST'])
def generate_quiz():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    uploaded_file = request.files['file']
    num_questions = request.form.get('num', 0)
    
    if not num_questions.isdigit() or int(num_questions) <= 0:
        return jsonify({"error": "Please enter a valid number of questions"}), 400
    
    num_questions = int(num_questions)
    
    try:
        vectors, final_documents, doc_lang = vector_embedding(uploaded_pdf=uploaded_file)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    prompt = ChatPromptTemplate.from_template("""
    Create a quiz containing {num} questions related to the provided context.
    Questions must be in multiple choice questions format.
    <context>
    {context}
    <context>""")
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")
    retriever = vectors.as_retriever()
    quiz_chain = create_stuff_documents_chain(llm, prompt)
    quiz_retrieval_chain = create_retrieval_chain(retriever, quiz_chain)
    quiz_response = quiz_retrieval_chain.invoke({'input': "Generate Quiz", 'num': num_questions})
    quiz_text = quiz_response['answer']
    
    # Process quiz to structured format
    quiz = []
    questions = quiz_text.split("\n\n")  # Assuming each question is separated by double new lines

    for question in questions:
        if '?' in question:
            q_text = question.split('?')[0] + '?'
            options = question.split('\n')[1:]
            correct_answer_index = random.randint(0, len(options) - 1)
            correct_answer = options[correct_answer_index]  # Randomly choose one of the options as the correct answer

            quiz.append({
                "question": q_text,
                "options": options,  # Keep options in the original order
                "answer": correct_answer  # Use the randomly selected option as the correct answer
            })
            
    return jsonify({"quiz": quiz})



# Submit the quiz
@app.route('/submit_quiz', methods=['POST'])
def submit_quiz():
    data = request.get_json()
    quiz = data.get('quiz', [])
    user_answers = data.get('answers', {})

    if not quiz or not user_answers:
        return jsonify({"error": "Invalid quiz or answers provided"}), 400
    
    correct_answers = 0
    results = []
    
    for i, q_data in enumerate(quiz):
        question = q_data['question']
        options = q_data['options']
        correct_option = q_data['answer']
        selected_option = user_answers.get(str(i), None)
        
        if selected_option == correct_option:
            result = {"question": question, "result": "correct", "correct_answer": correct_option}
            correct_answers += 1
        else:
            result = {"question": question, "result": "incorrect", "correct_answer": correct_option}
        results.append(result)
    
    score = f"{correct_answers}/{len(quiz)}"
    return jsonify({"score": score, "results": results})




# Study Planner
@app.route("/generate_study_plan/", methods=["POST", "GET"])
def generate_study_plan():
    if request.method == "POST":
        if 'uploaded_file' in request.files:
            uploaded_file = request.files['uploaded_file']
            vectors, final_documents, doc_lang = vector_embedding(uploaded_pdf=uploaded_file)

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

    if request.method == "GET":
        uploaded_file = request.files['uploaded_file']
        vectors, final_documents, doc_lang = vector_embedding(uploaded_pdf=uploaded_file)

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




# Mind Map Generator 
# In-memory storage (acts as a temporary replacement for session)
in_memory_data = {
    'vectors': None,
    'docs': None,
    'doc_lang': None
}
# Define the prompt template
prompt_template = ChatPromptTemplate.from_template("""
    You are an expert teacher tasked with generating a well-structured mind map for students to understand the given material.
    Your response should only contain a hierarchical mind map format containing Main Topic Subtopic and description in markdown language use "**" "###" and "* **" only.
    Ensure the text is easy to parse and does not include additional comments or unrelated information.
    Don't include exercises or questions in the mind map.
    <context>
    {context}
    <context>
""")

def generate_mermaid_from_text(context):
    """Generate a Mermaid-based mind map from markdown content provided by LLM."""
    mermaid_code = "```mermaid\nmindmap\n"

    current_topic = None
    current_subtopic = None

    # Parse markdown content
    lines = context.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Detect main topics (Markdown header level 1 - "**")
        if line.startswith("**") and line.endswith("**"):
            current_topic = line.replace("**", "").strip()
            mermaid_code += f"  {current_topic}\n"
        
        # Detect subtopics (Markdown header level 2 - "###")
        elif line.startswith("###") and current_topic:
            current_subtopic = line.replace("###", "").strip()
            mermaid_code += f"    {current_subtopic}\n"
        
        # Detect details (Markdown list or other text following subtopics)
        elif line.startswith("* **") and current_subtopic:
            detail = line.replace("* **", "").strip()
            mermaid_code += f"      {detail}\n"
        elif current_subtopic:
            # Treat non-header lines as details (catch any extra details)
            mermaid_code += f"      {line}\n"

    mermaid_code += "```\n"
    return mermaid_code

@app.route('/process_pdf', methods=['POST'])
def process_pdf():
    uploaded_file = request.files.get('file')
    document_text = request.form.get('document_text')
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")
    # Check if a file was uploaded or text was provided
    if uploaded_file or document_text:
        # Use your custom vector_embedding function
        vectors, docs, doc_lang = vector_embedding(uploaded_pdf=uploaded_file, document_text=document_text)

        # Store the results in memory
        in_memory_data['vectors'] = vectors
        in_memory_data['docs'] = docs
        in_memory_data['doc_lang'] = doc_lang

        # Step 2: Generate Mind Map if embeddings were created successfully
        if in_memory_data['vectors']:
            # Set up the document chain and retriever
            document_chain = create_stuff_documents_chain(llm, prompt_template)
            retriever = in_memory_data['vectors'].as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            
            # Measure response time
            start = time.process_time()
            response = retrieval_chain.invoke({'input': "generate mind map"})  # Request the LLM to generate a mind map
            response_time = time.process_time() - start
            
            # Check if the response is valid before generating the mind map
            if response and 'answer' in response and response['answer'].strip():
                # Generate mind map based on the LLM response context
                mermaid_mind_map = generate_mermaid_from_text(response['answer'])
                
                # Return the Mermaid code as a response (you can serve this to a frontend that renders Mermaid)
                return jsonify({"mermaid_code": mermaid_mind_map})
            else:
                return "The response from the LLM is empty or invalid. Please try again."
        else:
            return "Vector embeddings are not set up. Please upload a PDF file."
    else:
        return "No file or text input provided."






# Gemini api call answer
@app.route("/gemini_llm_chat/", methods=["POST", "GET"])
def gemini_llm_chat():
    if request.method == "POST":
        input_text = request.form.get('input')
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        model = genai.GenerativeModel("gemini-pro")
        chat = model.start_chat(history=[])

        response = chat.send_message(input_text, stream=True)

        chat_history = []
        for chunk in response:
            chat_history.append({"role": "Bot", "text": chunk.text})

        return jsonify({"chat_history": chat_history})

    if request.method == "GET":
        input_text = request.args.get('input', 'Default input text')
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        model = genai.GenerativeModel("gemini-pro")
        chat = model.start_chat(history=[])

        response = chat.send_message(input_text, stream=True)

        chat_history = []
        for chunk in response:
            chat_history.append({"role": "Bot", "text": chunk.text})

        return jsonify({"chat_history": chat_history})



# Youtube Summarizer
@app.route("/youtube_summarizer/", methods=["POST", "GET"])
def youtube_summarizer():
    from youtube_transcript_api import YouTubeTranscriptApi
    
    if request.method == "POST":
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
        if "Error" in transcript_text:
            return jsonify({"error": transcript_text})

        summary = generate_gemini_content(transcript_text, prompt)
        return jsonify({"summary": summary})

    if request.method == "GET":
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
        if "Error" in transcript_text:
            return jsonify({"error": transcript_text})

        summary = generate_gemini_content(transcript_text, prompt)
        return jsonify({"summary": summary})

if __name__ == "__main__":
    app.run(debug=True)
