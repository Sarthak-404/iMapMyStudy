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
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
import os
import bs4
import time
import io
import shutil
from dotenv import load_dotenv
import google.generativeai as genai
import uuid

load_dotenv()

app = Flask(__name__)
CORS(app, supports_credentials = True)

app.secret_key = os.getenv('FLASK_SECRET_KEY')
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
@app.route("/document_qa/", methods=["POST"])
def document_qa():
        print(request.files)  
        print(request.form)   
        if 'uploaded_file' not in request.files:
            return jsonify({"error": "No file uploaded"})
        uploaded_file = request.files['uploaded_file']
        prompt1 = request.form.get('prompt1')
        vectors, final_documents, doc_lang = vector_embedding(uploaded_pdf=uploaded_file)
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

        



# Link Summarizer
@app.route('/web_summarizer', methods=['POST'])
def web_summarizer():
    input_text = request.form.get('input')
    if not input_text :
        return jsonify({"error": "Please enter a valid URL"}), 400
    
    try:
        url = input_text 
        loader = WebBaseLoader(url)
        docs = loader.load()
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.split_documents(docs)
        vectors = FAISS.from_documents(final_documents, embeddings)
        prompt = ChatPromptTemplate.from_template('''
                Your job is to summarize the given context.
                Give a proper summary including all main points, topics and description.
                Summary must include all the details of the context and suggest some further ons.
                {context}
        ''')
        llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        response = retrieval_chain.invoke({'input': 'Generate Summary'})

        return jsonify({"summary": response['answer']}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500




# Quiz generator 
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
    
    # Modify the prompt to explicitly ask for correct answers
    prompt = ChatPromptTemplate.from_template("""
    Create a quiz containing {num} multiple choice questions based on the provided context.
    Each question should have 4 options, and one of the options should be explicitly marked as the correct answer.
    Provide the questions in the following format:
    
    Question 1: [question]
    Options:
    a) [option 1]
    b) [option 2]
    c) [option 3]
    d) [option 4]
    Answer: [correct option]
    
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
        if 'Question' in question and 'Answer:' in question:
            # Extract question text, options, and correct answer
            q_text = question.split("Options:")[0].strip()
            options = [opt.strip() for opt in question.split("Options:")[1].split("\n") if opt.strip()]
            
            # Extract the correct answer
            correct_answer_line = [line for line in question.split("\n") if 'Answer:' in line]
            if correct_answer_line:
                correct_answer = correct_answer_line[0].split('Answer:')[-1].strip()
            
            # Remove any option that starts with "Answer" from the options list
            options = [opt for opt in options if not opt.startswith("Answer")]
            
            quiz.append({
                "question": q_text,
                "options": options,  # Filtered options without "Answer"
                "answer": correct_answer  # Use the correct answer from the LLM
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
# @app.route("/youtube_summarizer/", methods=["POST"])
# def youtube_summarizer():
#     if request.method == "POST":
        
#         summarize = ChatPromptTemplate.from_template('''
#                         Summarize the content given.
#                         Content: {input}
#                     ''')
#         def open_url_in_chrome(url, mode='headed'):
#             try:
#                 options = webdriver.ChromeOptions()

#                 if mode == 'headless':
#                     options.add_argument('--headless')
#                     options.add_argument('--no-sandbox')
#                     options.add_argument('--disable-dev-shm-usage')
#                     options.add_argument('--disable-gpu')  
#                     options.add_argument('--remote-debugging-port=9222')

#                 driver = webdriver.Chrome(options=options)
#                 driver.get(url)
#                 return driver
#             except Exception as e:
#                 print(f"Failed to open chrome: {e}")
#                 return ""
        
#         def scroll_into_view_and_click(driver, element):
#             driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", element)
#             element.click()
        
#         def expand_description(driver):
#             try:
#                 # page_html = driver.page_source
#                 # with open("index.html", "w", encoding="utf-8") as f:
#                 #     f.write(page_html)
#                 # print("Saved page HTML to index.html.")
#                 # Locate the "More" button and scroll into view before clicking
#                 more_button = WebDriverWait(driver, 10).until(
#                     EC.element_to_be_clickable((By.XPATH, "//yt-formatted-string[text()='more']"))
#                 )
#                 driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", more_button)
#                 more_button.click()
#                 print("Expanded video description.")
#             except Exception as e:
#                 print(f"Failed to expand video description: {e}")

#         def show_transcript(driver):
#             try:
#                 # Locate the "Show transcript" button using its aria-label
#                 show_transcript_button = WebDriverWait(driver, 20).until(
#                     EC.element_to_be_clickable((By.XPATH, "//button[@aria-label='Show transcript']"))
#                 )
#                 driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", show_transcript_button)
#                 show_transcript_button.click()
#                 print("Clicked 'Show transcript' button.")
#             except Exception as e:
#                 print(f"Failed to click 'Show transcript': {e}")

#         def get_transcript(driver):
#             try:
#                 # Wait for the transcript section to be visible
#                 transcript_element = WebDriverWait(driver, 10).until(
#                     EC.presence_of_element_located((By.XPATH, "//ytd-transcript-segment-list-renderer"))
#                 )
#                 transcript = transcript_element.text
#                 return transcript
#             except Exception as e:
#                 print(f"Failed to retrieve transcript: {e}")
#                 return ""

#         def transcript_text_only(transcript):
#             # Remove timestamps and only return the text
#             transcript_lines = transcript.split('\n')
#             text_lines = transcript_lines[1::2]  # Extract only the text lines, skipping timestamps
#             return " ".join(text_lines)

#         def summarizer(transcript):
#             # Use LLM to generate a summary of the transcript
#             main = summarize.invoke({'input': transcript})
#             response = llm.invoke(main)
#             return response.content

#         url = request.form.get('youtube_link')
#         mode = 'headed'
#         llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")
#         # Open YouTube video in Chrome
#         try:
#             driver = open_url_in_chrome(url, mode)

#             # Expand description
#             expand_description(driver)

#             # Show transcript
#             show_transcript(driver)

#             # Get the transcript
#             transcript = get_transcript(driver)

#             # Close the browser
#             driver.quit()

#             # If transcript was successfully retrieved
#             if transcript:
#                 text_only = transcript_text_only(transcript)

#                 summary = summarizer(text_only)
#                 return jsonify({"Summary": str(summary)})
#             # else:
#             #     return "Transcript unavailable"
#         except Exception as e:
#             print(f"Error occurred: {str(e)}")
#             return jsonify({"error": str(e)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)  # , ssl_context=('cert.pem', 'key.pem')  paste it for https
    


# in tmux
# source venv/bin/activate
# tmux ls (to check how many servers are running)

# to access aws vs code 
# press ctr + shift + p 
# click on the instance 
# enter to the directory iMapMyStudy