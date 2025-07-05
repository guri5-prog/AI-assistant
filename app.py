import os
import re
import uuid
from datetime import timedelta
from pytube import YouTube
from flask import Flask, request, render_template, url_for, redirect, session
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from langchain_community.vectorstores import FAISS
import shutil
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
from langchain_community.document_loaders import PyPDFLoader
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound

UPLOAD_FOLDER = 'upload'
app = Flask(__name__)
app.secret_key = os.urandom(24)
app.permanent_session_lifetime = timedelta(minutes=30)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.before_request
def before_request():
    make_session_permanent()

def make_session_permanent():
    session.permanent = True
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())

def get_user_upload_folder():
    user_id = session.get('user_id', 'default')
    user_folder = os.path.join(app.config['UPLOAD_FOLDER'], user_id)
    os.makedirs(user_folder, exist_ok=True)
    return user_folder

def chunk_and_split(raw_text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_text(raw_text)

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    if os.path.exists('index'):
        shutil.rmtree('index')
    texts = [t for t in texts if t.strip() != ""]

    db = FAISS.from_texts(texts, embeddings)
    index_path = os.path.join('index', session['user_id'])
    if os.path.exists(index_path):
        shutil.rmtree(index_path)
    db.save_local(index_path)
    return db, texts

def clean_transcript(text: str) -> str:
    text = re.sub(r'\b(\w+\s*)\1{3,}', r'\1', text)

    # Remove extra newlines and spaces
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s{2,}', ' ', text)

    lines = text.splitlines()
    lines = [line.strip() for line in lines if len(line.strip()) > 10]

    cleaned = '\n'.join(lines)

    return cleaned[:5000]  

def chunk_and_split_document(filepath):
    loader = PyPDFLoader(filepath)
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(pages)

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    if os.path.exists('index'):
        shutil.rmtree('index')

    db = FAISS.from_documents(texts, embeddings)
    index_path = os.path.join('index', session['user_id'])
    if os.path.exists(index_path):
        shutil.rmtree(index_path)
    db.save_local(index_path)
    return db, texts

def model_documents(question, text, pdf):
    answer = ""
    sources = []
    
    db, text = chunk_and_split_document(pdf)
    retriever = db.as_retriever(search_kwargs={"k": 4})

    pipe = pipeline('text2text-generation', model='google/flan-t5-base')
    llm = HuggingFacePipeline(pipeline=pipe)

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type='map_reduce', retriever=retriever, return_source_documents=True)

    result = qa.invoke({'query': question})
    answer = result['result']
    sources = [doc.page_content for doc in result['source_documents']]
    return question, answer, sources, text


def get_video_id(url):
    query = urlparse(url)
    if query.hostname == 'youtu.be':
        return query.path[1:]
    if query.hostname in ('www.youtube.com', 'youtube.com'):
        if query.path == '/watch':
            return parse_qs(query.query)['v'][0]
        if query.path.startswith('/embed/'):
            return query.path.split('/')[2]
        if query.path.startswith('/v/'):
            return query.path.split('/')[2]
    return None

def transcript_with_whisper(url):
    yt = YouTube(url)
    audio_stream = yt.streams.filter(only_audio=True).first()
    audio_path = 'audio.mp4'
    audio_stream.download(filename=audio_path)

    whisper_pipe = pipeline('automatic-speech-recognition', model='openai/whisper-small')
    transcript = whisper_pipe(audio_path)['text']
    os.remove(audio_path)

    return transcript


def get_transcript_from_url(url):
    video_id = get_video_id(url)

    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        full_transcript = " ".join(t['text'] for t in transcript_list)
        return clean_transcript(full_transcript)
    except (TranscriptsDisabled, NoTranscriptFound):
        print("No caption found. Falling back to audio transcription")
        raw = transcript_with_whisper(url)
        return clean_transcript(raw)


def model(question, text):
    answer = ""
    sources = []

    db, text = chunk_and_split(text)
    retriever = db.as_retriever(search_kwargs={"k": 4})

    pipe = pipeline('text2text-generation', model='google/flan-t5-base')
    llm = HuggingFacePipeline(pipeline=pipe)

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type='map_reduce', retriever=retriever, return_source_documents=True)

    result = qa.invoke({'query': question})
    answer = result['result']
    sources = [doc.page_content for doc in result['source_documents']]
    return question, answer, sources, text


@app.route('/text', methods=['POST', "GET"])
def text():
    question = ""
    answer = ""
    pasted_text = ""
    sources = []

    if request.method == 'POST':
        pasted_text = request.form.get('pasted_text')
        question = request.form.get('question')

        question, answer, sources, pasted_text = model(question, pasted_text)

    return render_template('text.html', question=question, answer=answer, sources=sources, pasted_text=pasted_text)

@app.route('/youtube', methods=["POST", "GET"])
def youtube():
    link = ""
    answer = ""
    question = ""
    text = ""
    sources = []
    if request.method == 'POST':
        link = request.form.get('youtube')
        if not get_video_id(link):
            return 'Invalid or unsupported YouTube link.'
        question = request.form.get('question')
        text = get_transcript_from_url(link)

        question, answer, sources, text = model(question, text)
    return render_template('youtube.html', question=question, answer=answer, sources=sources)
        
@app.route('/pdf', methods=['POST', 'GET'])
def pdf():
    answer = ''
    sources = []
    text = ''
    question = ''
    filepath = ''

    if request.method == 'POST':
        file = request.files['pdf']
        question = request.form.get('question')
        
        if file:
            user_folder = get_user_upload_folder()
            filepath = os.path.join(user_folder, file.filename)
            file.save(filepath)
            
            question, answer, sources, text = model_documents(question, text, filepath)
    
    return render_template('pdf.html', question=question, answer=answer, sources=sources)


@app.route('/', methods=["POST", "GET"])
def home():
    if request.method == "POST":
        choice = request.form.get('data')
        if choice == "Text":
            return redirect(url_for('text'))
        elif choice == "Youtube":
            return redirect(url_for('youtube'))
        else:
            return redirect(url_for('pdf'))
    
    return render_template('choice.html')
        
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

