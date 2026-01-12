⚠️ IMPORTANT (READ FIRST)

Before running the main file, you MUST have:

A local fine-tuned QA model folder

A local ChromaDB database folder

A valid OpenAI API key

ARC (Synthiam) running and reachable on TCP

If any of these are missing, the program will not run.

project_root/
│
├── main_With_button.py          # Main program (this file)
│
├── neu_faculty_qa_model/        # ✅ REQUIRED
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── vocab.txt
│
├── neu_faculty_db/              # ✅ REQUIRED
│   ├── chroma.sqlite3
│   └── index/
│
├── .env                         # OpenAI API key
└── README.md


///////////////////////////////////////////

pip install --upgrade pip

pip install \
asyncio \
websockets \
sounddevice \
numpy \
python-dotenv \
torch \
transformers \
sentence-transformers \
chromadb \
fuzzywuzzy \
python-Levenshtein
