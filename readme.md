Here’s a complete and polished `README.md` for your **AI Study Assistant** project:

---

# 📚 AI Study Assistant

**An AI-powered web app that answers your questions based on pasted text, YouTube videos, or uploaded PDFs.**
It uses NLP, embeddings, vector databases, and large language models to understand and respond to your queries.

---

## ✨ Features

✅ **Ask Questions From:**

* 🔤 Pasted Text
* 📺 YouTube Videos (with or without subtitles)
* 📄 PDF Documents

✅ **Technologies Used:**

* **Flask** – Lightweight backend web framework
* **LangChain** – Framework for building language model apps
* **Hugging Face Transformers** – For running LLMs like `Flan-T5`
* **Whisper** – Converts YouTube audio to text when subtitles are unavailable
* **FAISS** – Vector database for document retrieval
* **Sentence Transformers** – For generating text embeddings
* **YouTube Transcript API** – For extracting YouTube captions

---

## 🧠 How It Works

1. **Text/PDF/YouTube** input is taken from the user.
2. The content is **split into chunks** and converted to **vector embeddings**.
3. FAISS performs **semantic search** over the embedded content.
4. A **retriever-augmented generation (RAG)** pipeline powered by a **Flan-T5 model** answers the question.
5. Sources and extracted documents are shown alongside the answer.


### Example Render Build Settings:

* **Build Command:** `pip install -r requirements.txt`
* **Start Command:** `python app.py`
* **Environment:** Python 3.x

---

## 🛠️ Installation

1. **Clone the repository:**

```bash
git clone https://github.com/your-username/ai-study-assistant.git
cd ai-study-assistant
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Run the app locally:**

```bash
python app.py
```

4. Open in browser:
   `http://localhost:5000`

---

## 📁 Project Structure

```
.
├── app.py
├── requirements.txt
├── templates/
│   ├── choice.html
│   ├── text.html
│   ├── pdf.html
│   └── youtube.html
├── upload/
├── index/
└── README.md
```

---

## 📦 Requirements

* Python 3.8+
* `Flask`
* `transformers`
* `torch`
* `sentence-transformers`
* `langchain`
* `faiss-cpu`
* `PyMuPDF`
* `pytube`
* `youtube-transcript-api`

---

## 🙌 Acknowledgments

* [LangChain](https://www.langchain.com/)
* [Hugging Face Transformers](https://huggingface.co/models)
* [FAISS](https://github.com/facebookresearch/faiss)
* [Whisper by OpenAI](https://github.com/openai/whisper)
* [YouTube Transcript API](https://github.com/jdepoix/youtube-transcript-api)

---

## 📢 Feedback & Contributions

I'm actively improving this project.
Feel free to open **issues**, suggest features, or submit **pull requests**!

---

## 🧑‍💻 Author

**Gourav Shekhar**
🔗 [LinkedIn](https://www.linkedin.com/in/your-profile)


