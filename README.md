
## ⚠️ IMPORTANT (READ FIRST)

This program **WILL NOT RUN** unless **ALL** requirements below are met.

### ✅ REQUIRED

1. **Local ChromaDB folder**

   * Path (must exist):

     ```text
     ./faculty_chromadb/
     ```
   * Used for faculty name matching and information retrieval

2. **OpenAI API key**

   * Create a file named `.env` in the project root:

     ```env
     OPENAI_API_KEY=your_api_key_here
     ```

3. **Synthiam ARC**

   * Must be running
   * TCP Script Server enabled
   * Address:

     ```
     127.0.0.1 : 8080
     ```

4. **Working microphone**

   * Required for real-time voice activation

❌ **NO fine-tuned QA model is required**
❌ **NO local LLM inference is used**

---

## 📁 REQUIRED PROJECT STRUCTURE

```text
project_root/
│
├── main.py                  # Main program
├── faculty_chromadb/         # REQUIRED
│   ├── chroma.sqlite3
│   └── index/
│
├── .env                     # REQUIRED
├── music/                   # REQUIRED (Maestro mode)
│   ├── mozart.mp3
│   ├── beethoven.mp3
│   └── vivaldi.mp3
│
└── README.md
```

Folder names **must NOT be changed**.

---


pip install --upgrade pip



pip install asyncio websockets sounddevice numpy python-dotenv torch \
transformers sentence-transformers chromadb fuzzywuzzy python-Levenshtein



