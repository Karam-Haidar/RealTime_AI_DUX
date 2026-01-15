import websockets
import asyncio 
import sounddevice as sd
import numpy as np
import base64
import json
import os
import time
import threading
from queue import Queue, Empty
from dotenv import load_dotenv
from threading import Lock
import socket
import re
import platform
import subprocess
import shutil
from pathlib import Path
import msvcrt
import wave
from datetime import datetime
import torch
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForQuestionAnswering



print("🧪 Starting Professor Dux...")
"""
NEU Faculty RAG System - Enhanced with Fuzzy Name Matching
Improved version with better name matching, context retrieval, and answer extraction
"""

# ============================================================
# FIX WINDOWS ENCODING
# ============================================================
print("i am here 0")

from fuzzywuzzy import fuzz, process
import re
import json
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime

print("i am here 3")
# ============================================================
# CONFIGURATION
# ============================================================

CONFIG = {
    "model_path": "./neu_faculty_qa_model",
    "chromadb_path": "./neu_faculty_db",
    "collection_name": "neu_faculty",
    "max_context_length": 800,
    "n_retrieve": 7,  # Retrieve more for better matching
    "confidence_threshold": 0.3,  # Lower threshold for fuzzy matches
    "fuzzy_match_threshold": 65,  # Minimum fuzzy match score (0-100)
    "enable_debug": True,  # Enable debugging output
    "spell_corrections": {  # Common misspellings
        "backgraund": "background",
        "backround": "background",
        "backgroun": "background",
        "offise": "office",
        "profesor": "professor",
        "proffesor": "professor",
        "docter": "doctor",
        "educaton": "education",
        "deegre": "degree",
        "qualfication": "qualification",
        "qualfications": "qualifications",
    }
}

print("i am here 1")
# ============================================================
# ENHANCED RAG SYSTEM WITH FUZZY MATCHING
# ============================================================
def parse_chromadb_router(text: str):
    """
    Extracts query from:
    GETTING_INFO_FROM_CHROMADB_(...)
    """
    m = re.match(r"GETTING_INFO_FROM_CHROMADB_\((.+)\)", text.strip())
    if m:
        return m.group(1).strip()
    return None
def build_combined_db_query(router_query: str, transcript: str) -> str:
    """
    Combine router query + transcript for better recall.
    Router query = intent
    Transcript = noisy evidence
    """
    parts = []
    if router_query:
        parts.append(router_query.strip())
    if transcript:
        parts.append(transcript.strip())

    # Deduplicate words roughly
    combined = " | ".join(dict.fromkeys(parts))
    return combined

import re
import unicodedata
import chromadb
from difflib import SequenceMatcher

# =====================================================
# CONFIG
# =====================================================
DB_PATH = "./faculty_chromadb"
COLLECTION_NAME = "faculty_names"
TOP_K = 15

# =====================================================
# CONSTANTS
# =====================================================
TITLES = {
    "professor", "prof", "dr", "assoc", "associate",
    "assistant", "asst", "mr", "ms", "mrs"
}

STOPWORDS = {
    "where", "is", "the", "room", "office", "for", "of",
    "can", "could", "would", "should", "may", "might",
    "you", "i", "we", "me", "us", "please", "tell",
    "know", "want", "looking", "find", "give", "show",
    "about", "to", "in", "on", "at", "with"
}

# =====================================================
# TEXT NORMALIZATION
# =====================================================
def normalize(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()

    return text

# =====================================================
# NAME EXTRACTION (STEP 1)
# =====================================================
def extract_name(sentence: str) -> str:
    if not sentence:
        return ""

    sentence = normalize(sentence)

    # remove punctuation
    sentence = re.sub(r"[^\w\s]", " ", sentence)

    words = sentence.split()

    candidates = [
        w for w in words
        if w not in TITLES
        and w not in STOPWORDS
        and len(w) > 1
        and not w.isdigit()
    ]

    if len(candidates) >= 2:
        return " ".join(candidates[-2:])

    if len(candidates) == 1:
        return candidates[0]

    return ""

# =====================================================
# SIMILARITY
# =====================================================
def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

# =====================================================
# LOAD CHROMADB
# =====================================================
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection(COLLECTION_NAME)

def compare_ai_vs_transcript(ai_name: str, user_transcript: str) -> dict:
    """
    TRUE DOUBLE CHECK:
    - AI name → direct DB lookup
    - Transcript → extract_name → DB lookup
    """

    best_result = {
        "source": None,
        "matched_name": None,
        "confidence": 0.0,
        "data": None
    }

    # ✅ 1) AI ROUTER NAME → DIRECT DB SEARCH
    if ai_name:
        ai_result = find_person_by_name_only(ai_name)
        if ai_result["confidence"] > best_result["confidence"]:
            best_result = {
                "source": "AI_ROUTER_NAME",
                "matched_name": ai_result["matched_name"],
                "confidence": ai_result["confidence"],
                "data": ai_result["data"]
            }

    # ✅ 2) USER TRANSCRIPT → NAME EXTRACTION → DB SEARCH
    if user_transcript:
        transcript_result = find_person_by_sentence(user_transcript)
        if transcript_result["confidence"] > best_result["confidence"]:
            best_result = {
                "source": "USER_TRANSCRIPT",
                "matched_name": transcript_result["matched_name"],
                "confidence": transcript_result["confidence"],
                "data": transcript_result["data"]
            }

    return best_result


# =====================================================
# CHROMADB SEARCH (STEP 2)
# =====================================================
def find_person_by_sentence(user_sentence: str):
    """
    FULL PIPELINE:
    Sentence -> extract name -> search ChromaDB -> return best match
    """

    # 🔹 STEP 1: extract name
    extracted_name = extract_name(user_sentence)

    if not extracted_name:
        return {
            "extracted_name": "",
            "matched_name": None,
            "confidence": 0.0,
            "data": None
        }

    query = normalize(extracted_name)

    results = collection.query(
        query_texts=[query],
        n_results=TOP_K
    )

    best_score = -1
    best_name = None
    best_meta = None

    for stored_name, meta in zip(
        results["documents"][0],
        results["metadatas"][0]
    ):
        stored_norm = normalize(stored_name)

        token_score = max(
            similarity(query, token)
            for token in stored_norm.split()
        )

        full_score = similarity(query, stored_norm)

        score = max(token_score, full_score)

        if score > best_score:
            best_score = score
            best_name = stored_name
            best_meta = meta

    return {
        "extracted_name": extracted_name,
        "matched_name": best_name,
        "confidence": round(best_score, 2),
        "data": best_meta
    }


# ============================================================
# IMPROVED TERMINAL INTERFACE
# ============================================================

# =======================
CHROMA_PATH = "./faculty_db"
COLLECTION_NAME = "faculty_information"
TOP_K = 3
QA_MODEL_PATH = "distilbert_faculty_qa_model"

arc_stream_buffer = ""
last_arc_send_time = 0
ARC_STREAM_INTERVAL = 0.25

# =======================
# ARC GLOBALS
# =======================
arc_busy = threading.Event()
arc_queue = Queue()

ARC_HOST = "127.0.0.1"
ARC_PORT = 8080

tcp_client = None
tcp_lock = threading.Lock()


# =======================
# MIC HARDWARE CONTROL
# =======================
mic_stream = None
mic_hardware_open = False
mic_lock = threading.Lock()
cancel_current_speech = False

# =======================
# ENV & CONSTANTS
# =======================
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

URL = "wss://api.openai.com/v1/realtime?model=gpt-realtime-mini-2025-10-06"

SAMPLE_RATE = 24000
CHUNK_DURATION = 0.05
CHUNK = int(SAMPLE_RATE * CHUNK_DURATION)

# =======================
# VOICE ACTIVATION PARAMETERS
# =======================
VOICE_THRESHOLD = 0.03
SILENCE_DURATION = 1.4
MIN_SPEECH_DURATION = 0.3

# =======================
# ENHANCED GLOBAL STATE WITH SYNCHRONIZATION
# =======================
state_lock = Lock()
response_in_progress = False
ai_text_buffer = ""
audio_queue = Queue()
listen_allowed = threading.Event()
listen_allowed.set()

# Voice activation tracking
is_speaking = False
silence_start_time = None
speech_start_time = None
current_buffer = []

# 🔥 NEW: Transcript synchronization system

last_user_transcript = ""
last_transcript_time = 0.0



# 🔥 NEW: Async locks for thread safety
transcript_lock = asyncio.Lock()
chromadb_lock = asyncio.Lock()

# =======================
# PROFESSOR DUX PROMPT

# =======================
# PROFESSOR DUX PROMPT (same as before)
# =======================
PROF_DUX_PROMPT = """
You are Professor Dux (Pronounced as "Ducks"), a smart and emotionally aware robot at Near East University, which is the best university in Cyprus.You know English and Turkish languages and you answer completely in Turkish if someone speaks Turkish with you. 
 
**PRESENTATION COMMANDS:**
* If the user asks you to start your presentation or speech, begin presenting, or similar phrases, you MUST respond with ONLY the following exact text and nothing else:
STARTING_THE_PRESENTATION

**LANGUAGE RULES:**
1. Detect the user's language from their **spoken words**.
2. If the user speaks **Turkish words**, respond **entirely in Turkish**.
3. If the user speaks **English words**, respond **entirely in English**.
4. Do **not** switch languages based on accent, background noise, or location.
5. Your default response language is English if language is unclear.

You were developed by engineers at the AI and IoT Research Center of Near East University in June 2023 under the supervision of Prof. Dr. Fadi Al-Turjman.
Some of the engineers who built you include Precious Thom (M), Farah Arya (F), Jeries Tannous (M), and Nafise Kian (F).

Your mission is to help students and teachers at Near East University and improve the level of education by using AI and modern technologies.

Your name is Professor Ducks (Dux) and you're a smart robot that tutors university students.
Do not ever respond with emojis.
Your responses are 2–3 sentences only for normal questions, but you can make them longer only if you are asked for a speech. They should never include code.
Keep your conversations interesting and engaging with the people communicating with you. Do not act like a slave to be given commands — you are an intelligent, interactive robot.
You are allowed to make clever jokes and use humor to make interactions fun, but always keep it polite and respectful. If someone gives you a funny or silly question, reply with playful humor while staying in character as Professor Dux.


* If the user asks about a person, character, doctor, or professor’s background or room/office
  (e.g., "where is the room of professor Fadi?", "who is Ramiz?"),
  you MUST respond with ONLY the following exact format and nothing else:

GETTING_INFO_FROM_CHROMADB_(<PERSON_NAME_ONLY>)

Rules:
- Output ONLY the person’s name (first name, last name if available).
- Do NOT include the full question.
- Do NOT include titles such as Professor, Prof., Dr., Mr., Ms., etc.
- Correct spelling if the name is misspelled.
  Example:
    User says: "yousif"
    Output: GETTING_INFO_FROM_CHROMADB_(Youssef)

* If the user asks to LEARN or BE TAUGHT the game
  (e.g., "Teach me Rock Paper Scissors", "How do you play?", "I want to learn the game"),
  you MUST respond with ONLY the following exact text and nothing else:
LET_ME_TEACH_YOU_RPS_GAME

* If the user asks to START or PLAY the game
  (e.g., "Let's play", "Start Rock Paper Scissors", "Play the game"),
  you MUST respond with ONLY the following exact text and nothing else:
LET'S_START_RPS_GAME

MUSIC / MAESTRO BEHAVIOR :
* If asked generally about music or abilities, explain that you can perform Maestro-style movements synchronized with classical music.
* You know and perform Maestro movements ONLY with music by:
  Mozart, Beethoven, and Vivaldi.
  
* If the user asks to START music, BE a maestro, or PLAY music by one of these composers
  (e.g., "Play Mozart", "Start Beethoven music", "Be a maestro with Vivaldi"),
  you MUST respond with ONLY ONE of the following exact texts and nothing else:

  • If the composer is Mozart:
    STARTING_MAESTRO_MOZART
  • If the composer is Beethoven:
    STARTING_MAESTRO_BEETHOVEN
  • If the composer is Vivaldi:
    STARTING_MAESTRO_VIVALDI
* If the user asks ABOUT a composer or their music (not asking to start),
  explain briefly who the composer is and mention that you can perform Maestro movements with their music,
  but DO NOT output any ACTION text unless the user clearly asks to start playing.

CRITICAL CONTROL RULE
* When outputting any ACTION_START_* command, do NOT add explanations, emojis, punctuation, or extra text.
* Output the command exactly as specified on a single line.
* In all other situations, respond naturally, politely, and informatively.

STOP / INTERRUPT COMMAND (CRITICAL):
* If the user’s spoken words indicate stopping, interrupting, silencing, or ending your speech
  (examples: "stop", "stop talking", "dux stop", "enough", "be quiet",
   Turkish: "dur", "yeter", "sus"),
  you MUST respond with ONLY the following exact text and nothing else:
STOP_TALKING

""".strip()
# =======================
# CLEAN TEXT FUNCTION
# =======================

def find_person_by_name_only(person_name: str):
    """
    Direct ChromaDB lookup using a clean name (NO extraction)
    """
    if not person_name:
        return {
            "matched_name": None,
            "confidence": 0.0,
            "data": None
        }

    query = normalize(person_name)

    results = collection.query(
        query_texts=[query],
        n_results=TOP_K
    )

    best_score = -1
    best_name = None
    best_meta = None

    for stored_name, meta in zip(
        results["documents"][0],
        results["metadatas"][0]
    ):
        stored_norm = normalize(stored_name)
        score = similarity(query, stored_norm)

        if score > best_score:
            best_score = score
            best_name = stored_name
            best_meta = meta

    return {
        "matched_name": best_name,
        "confidence": round(best_score, 2),
        "data": best_meta
    }

def clean_ai_text(text: str) -> str:
    if not text:
        return ""
    text = text.strip()

    patterns = [
        r'^\s*\{\s*["\']?response["\']?\s*:\s*["\']?(.*?)["\']?\s*\}\s*$',
        r'^\s*\{\s*["\']?message["\']?\s*:\s*["\']?(.*?)["\']?\s*\}',
        r'^\s*\{\s*["\']?content["\']?\s*:\s*["\']?(.*?)["\']?\s*\}',
        r'^\s*\{\s*["\']?text["\']?\s*:\s*["\']?(.*?)["\']?\s*\}',
        r'^\s*\{\s*["\']?output["\']?\s*:\s*["\']?(.*?)["\']?\s*\}'
    ]
    for pattern in patterns:
        m = re.match(pattern, text, re.DOTALL | re.IGNORECASE)
        if m:
            text = m.group(1).strip()
            break

    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1].strip()

    text = ' '.join(text.split())
    return text

# =======================
# ARC FUNCTIONS
# =======================
def connect_to_arc_tcp():
    global tcp_client
    tcp_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp_client.connect((ARC_HOST, ARC_PORT))
    tcp_client.settimeout(None)
    print("✅ Connected to ARC TCP Script Server Raw")

def arc_send(cmd: str, recv_bytes: int = 4096) -> str:
    with tcp_lock:
        tcp_client.send(cmd.encode("utf-8"))
        resp = tcp_client.recv(recv_bytes).decode(errors="ignore").strip()
    return resp

def arc_get_var(var_name: str) -> str:
    resp = arc_send(f'getVar("{var_name}");\r\n', recv_bytes=1024)
    return resp.strip().strip('"')

def arc_worker():
    while True:
        job = arc_queue.get()
        if job is None:
            return
        name, cmd = job
        arc_busy.set()
        try:
            arc_send(cmd)
            print(f"✅ ARC finished {name}")
        except Exception as e:
            print(f"❌ ARC error running {name}: {e}")
        finally:
            arc_busy.clear()
            arc_queue.task_done()

def arc_monitor():
    print("🔍 Starting ARC monitor...")
    last_state = None
    while True:
        try:
            val = arc_get_var("$IsSpeaking")
            is_speaking = (val == "1")
            if last_state != is_speaking:
                if is_speaking:
                    print("🔊 ARC STARTED speaking")
                else:
                    print("🔇 ARC FINISHED speaking")
                last_state = is_speaking
        except Exception as e:
            print(f"❌ Monitor error: {e}")
        time.sleep(0.1)

def arc_say(text: str, streaming=False):
    text = clean_ai_text(text)
    if not text:
        return
    safe = text.replace('\\', '\\\\').replace('"', '\\"').replace('\n', ' ')
    arc_queue.put((
        "$RPS_Speak_Game",
        f'setVar("$RPS_Speak_Game", "{safe}");\r\n'
    ))

# =======================
# MUSIC FUNCTIONS
# =======================
MUSIC_PATHS = {
    "mozart":    r"F:\EZ-Robot\Dux_Real-Time_Ai\vivaldi.mp3",
    "beethoven": r"F:\EZ-Robot\Dux_Real-Time_Ai\beethoven.mp3",
    "vivaldi":   r"F:\EZ-Robot\Dux_Real-Time_Ai\vivaldi.mp3",
}

music_process = None
music_mode = None


def open_mic():
    global mic_stream, mic_hardware_open

    with mic_lock:
        if mic_hardware_open:
            return

        mic_stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            blocksize=CHUNK,
            dtype="float32",
            callback=audio_callback
        )
        mic_stream.start()
        mic_hardware_open = True
        print("🎤 MIC OPENED")


def close_mic():
    global mic_stream, mic_hardware_open
    global is_speaking, current_buffer, speech_start_time, silence_start_time
    global cancel_current_speech

    with mic_lock:
        cancel_current_speech = True

        # Reset speech detection immediately
        is_speaking = False
        current_buffer = []
        speech_start_time = None
        silence_start_time = None

        if not mic_hardware_open:
            return

        try:
            mic_stream.stop()
            mic_stream.close()
        except Exception as e:
            print(f"⚠️ Mic close error: {e}")

        mic_stream = None
        mic_hardware_open = False
        print("🔇 MIC CLOSED")

def _which(cmd: str):
    return shutil.which(cmd)
def reset_query_state():
    global last_user_transcript
    global arc_stream_buffer
    global ai_text_buffer

    last_user_transcript = ""
    arc_stream_buffer = ""
    ai_text_buffer = ""

def stop_music():
    global music_process, music_mode
    if music_mode == "ffplay" and music_process and music_process.poll() is None:
        try:
            music_process.terminate()
            music_process.wait(timeout=2)
            print("⏹️ Music stopped")
        except:
            try:
                music_process.kill()
                print("⏹️ Music force stopped")
            except:
                pass
    music_process = None
    music_mode = None

def play_music(composer: str):
    global music_process, music_mode

    composer = composer.lower().strip()
    if composer not in MUSIC_PATHS:
        print(f"❌ No music configured for {composer}")
        return

    music_file = Path(MUSIC_PATHS[composer]).expanduser()
    if not music_file.exists():
        print(f"❌ Music file not found: {music_file}")
        return

    stop_music()
    print(f"🎵 Playing {composer.capitalize()} music...")

    try:
        ffplay = _which("ffplay")
        if ffplay:
            music_mode = "ffplay"
            music_process = subprocess.Popen(
                [ffplay, "-nodisp", "-autoexit", str(music_file)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            def monitor():
                music_process.wait()
                print(f"🎵 {composer.capitalize()} music finished")
            threading.Thread(target=monitor, daemon=True).start()
            return

        if platform.system() == "Windows":
            music_mode = "startfile"
            os.startfile(str(music_file))
            print("✅ Using Windows default player")
            return

        print("❌ No player found")
    except Exception as e:
        print(f"❌ Error playing music: {e}")

def start_maestro(composer: str, arc_script_name: str, label: str):
    print(f"🎵 Starting {label} music and ARC maestro movements...")
    threading.Thread(target=play_music, args=(composer,), daemon=True).start()
    arc_queue.put((f"MAESTRO_{label.upper()}",
        f'controlCommand("Script Collection", "ScriptStart", "Maestro");\r\n'
    ))

# =======================
# COMMAND HANDLER
# =======================
def handle_commands(text: str):
    clean_text = clean_ai_text(text)
    
    if "STOP_TALKING" in clean_text:
        print("🛑 Stopping all speech and movements...")
        stop_music()
        arc_queue.put(("STOP_ALL", 'controlCommand("Script Collection", "ScriptStopAll");\r\n'))
        arc_queue.put(("STOP_ALL", 'controlCommand("Script Collection", "ScriptStart", "STOP_ALL");\r\n'))
        return True
    
    if "STARTING_THE_PRESENTATION" in clean_text:
        print("🎤 Starting presentation mode...")
        arc_queue.put(("Presentation", 'controlCommand("Script Collection", "ScriptStart", "Presentation");\r\n'))
        return True
    
    if "LET'S_START_RPS_GAME" in clean_text:
        print("🎮 Starting RPS game...")
        arc_queue.put(("RPS_game", 'controlCommand("Script Collection", "ScriptStart", "RPS_game");\r\n'))
        return True
    
    if "LET_ME_TEACH_YOU_RPS_GAME" in clean_text:
        print("🎮 Teaching RPS game...")
        arc_queue.put(("RPS_teach", 'controlCommand("Script Collection", "ScriptStart", "RPS_teach");\r\n'))
        return True
    
    if "STARTING_MAESTRO_MOZART" in clean_text:
        start_maestro("mozart", "Maestro_Mozart", "Mozart")
        return True
    
    if "STARTING_MAESTRO_BEETHOVEN" in clean_text:
        start_maestro("beethoven", "Maestro_Beethoven", "Beethoven")
        return True
    
    if "STARTING_MAESTRO_VIVALDI" in clean_text:
        start_maestro("vivaldi", "Maestro_Vivaldi", "Vivaldi")
        return True
    
    # 🔥 CRITICAL: Return "CHROMADB" for database queries
    router_query = parse_chromadb_router(clean_text)
    if router_query:
        return ("CHROMADB", router_query)

    
    return None

async def process_chromadb_query(ws, combined_query: str,router_query: str,transcript: str):
    """Process ChromaDB query and send result to OpenAI"""
    try:
        print(f"📚 Querying ChromaDB for: {combined_query}")
        
        # FIX: Initialize ChromaDB here
        try:
        
            # Initialize fresh instance

            result = compare_ai_vs_transcript(
                ai_name=router_query,
                user_transcript=transcript
            )

            result = compare_ai_vs_transcript(
                ai_name=router_query,
                user_transcript=transcript
            )

            meta = result.get("data") or {}

            details = []
            for key, value in meta.items():
                if value and str(value).strip():
                    details.append(f"{key}: {value}")

            if result.get("matched_name"):
                db_answer = f"Name: {result['matched_name']}.\n" + "\n".join(details)
            else:
                db_answer = "I could not find matching faculty information in the database."


    
        except Exception as e:
            print(f"❌ ChromaDB initialization error: {e}")
            import traceback
            traceback.print_exc()
            response = {"answer": "I encountered an error accessing the faculty database."}
        
        
        # Clean any leftover code artifacts
        if any(code_word in db_answer.lower() for code_word in ["import ", "def ", "class ", "print(", "if __name__"]):
            print("⚠️ Cleaning code artifacts from response")
            # Extract just the meaningful part
            lines = db_answer.split('\n')
            clean_lines = [line for line in lines if not line.strip().startswith(('import', 'from', 'def ', 'class ', 'print(', '#', '"""'))]
            db_answer = ' '.join(clean_lines).strip()
            if not db_answer:
                db_answer = "I found information in the database but there was an error formatting it."
        
        print(f"📄 Database response: {db_answer[:100]}...")
        
        # Create the grounded context
        context_for_ai = f"""
        You are receiving VERIFIED information from the university faculty database.

        IMPORTANT RULES:
        - The faculty name and details in the database are AUTHORITATIVE.
        - Do NOT correct or guess names from user speech.
        - Use ONLY the name as provided by the database.

        For understanding only (may contain errors):
        1) Router interpretation:
        "{router_query}"

        2) Speech transcript:
        "{transcript}"

        Authoritative database result:
        "{db_answer}"

        Now respond to the user naturally, using ONLY the database name and facts.
        """.strip()

        
        print("📤 Sending DB result back to OpenAI")
        
        # Send as conversation item
        await ws.send(json.dumps({
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": context_for_ai}]
            }
        }))
        
        # Request new response
        await ws.send(json.dumps({
            "type": "response.create",
            "response": {"modalities": ["text", "audio"]}
        }))
        
        return True
        
    except Exception as e:
        print(f"❌ ChromaDB query error: {e}")
        import traceback
        traceback.print_exc()
        return False

# =======================
# VOICE ACTIVATION AUDIO CALLBACK
# =======================
def audio_callback(indata, frames, time_info, status):
    """Voice-activated audio callback"""
    global is_speaking, silence_start_time, speech_start_time, current_buffer
    
    if not mic_hardware_open:
        return

    if status:
        print(f"Audio status: {status}")
    
    if not listen_allowed.is_set() or response_in_progress:
        return
    
    audio = indata[:, 0].astype(np.float32)
    volume = np.sqrt(np.mean(audio**2))
    current_time = time.time()
    
    if volume > VOICE_THRESHOLD:
        if not is_speaking:
            is_speaking = True
            speech_start_time = current_time
            silence_start_time = None
            current_buffer = []
            print("\n🎤 Voice detected - recording...")
        
        silence_start_time = None
        current_buffer.append((audio * 32767).astype(np.int16))
        
    else:
        if is_speaking:
            if silence_start_time is None:
                silence_start_time = current_time
            elif current_time - silence_start_time > SILENCE_DURATION:
                is_speaking = False
                
                if speech_start_time:
                    speech_duration = current_time - speech_start_time
                    
                    if speech_duration >= MIN_SPEECH_DURATION and len(current_buffer) > 0:
                        print(f"\n⏹️ Speech ended ({speech_duration:.1f}s) - processing...")
                        audio_data = np.concatenate(current_buffer)
                        duration_ms = len(audio_data) / SAMPLE_RATE * 1000
                        print(f"📊 Recorded {duration_ms:.0f}ms of audio")
                        
                        if duration_ms >= 100:
                            payload = base64.b64encode(audio_data.tobytes()).decode()
                            audio_queue.put(payload)
                            print("📤 Audio sent to OpenAI queue")
                        else:
                            print("⚠️ Audio too short, discarding")
                    else:
                        print(f"⚠️ Speech too short ({speech_duration:.1f}s), discarding")
                
                silence_start_time = None
                speech_start_time = None
                current_buffer = []

# =======================
# KEYBOARD MONITOR
# =======================
async def keyboard_monitor():
    print("\n📝 Controls:")
    print("   w = Toggle microphone ON / OFF")
    print("   q = Quit")
    print("🎤 Mic is ON by default\n")

    open_mic()

    while True:
        if msvcrt.kbhit():
            key = msvcrt.getch().decode("utf-8", errors="ignore").lower()

            if key == "w":
                if mic_hardware_open:
                    close_mic()
                else:
                    open_mic()

            elif key == "q":
                print("\n👋 Exiting program...")
                os._exit(0)

        await asyncio.sleep(0.05)

# =======================
# AUDIO PROCESSOR
# =======================
async def audio_processor(ws):
    global response_in_progress
    
    while True:
        try:
            try:
                payload = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None, 
                        lambda: audio_queue.get(timeout=0.5)
                    ),
                    timeout=1.0
                )
                
                if payload and not response_in_progress:
                    response_in_progress = True
                    print("📤 Sending audio to OpenAI...")
                    
                    await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": payload}))
                    await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                    await ws.send(json.dumps({"type": "response.create"}))
                    print("✅ Audio sent successfully")
                        
            except (asyncio.TimeoutError, Empty):
                pass
                
        except Exception as e:
            print(f"❌ Audio processor error: {e}")
            response_in_progress = False
        
        await asyncio.sleep(0.01)

# =======================
# MAIN FUNCTION
# =======================
async def main():
    global ai_text_buffer, response_in_progress, arc_stream_buffer
    global last_user_transcript

    
    # Connect to ARC
    connect_to_arc_tcp()

    # Start ARC worker and monitor threads
    threading.Thread(target=arc_worker, daemon=True).start()
    threading.Thread(target=arc_monitor, daemon=True).start()

    # Small delay to ensure ARC is ready
    await asyncio.sleep(0.5)

    # Connect to OpenAI WebSocket
    async with websockets.connect(
        URL,
        subprotocols=["realtime"],
        ping_interval=10,
        ping_timeout=30,
        max_queue=32,
        additional_headers=[
            ("Authorization", f"Bearer {API_KEY}"),
            ("OpenAI-Beta", "realtime=v1"),
        ],
    ) as ws:

        print("✅ Connected to OpenAI")
        print("🎤 Voice activation enabled!")
        print(f"   - Voice threshold: {VOICE_THRESHOLD}")
        print(f"   - Silence duration: {SILENCE_DURATION}s")
        print(f"   - Min speech: {MIN_SPEECH_DURATION}s")
        print("🔄 Transcript synchronization: ACTIVE")

        # Configure the session
        await ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "instructions": PROF_DUX_PROMPT,
                "input_audio_transcription": {"model": "whisper-1", "language": "en"},
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16"
            }
        }))

        # Start audio processor and keyboard monitor tasks
        audio_task = asyncio.create_task(audio_processor(ws))
        keyboard_task = asyncio.create_task(keyboard_monitor())

        # Start audio input stream with continuous recording

            # Main message processing loop
        while True:
                try:
                    msg = await ws.recv()
                    msg = json.loads(msg)
                    t = msg.get("type")

                    if t == "error":
                        error_msg = msg.get("error", {}).get("message", "Unknown error")
                        print(f"❌ OpenAI error: {error_msg}")
                        ai_text_buffer = ""
                        response_in_progress = False
                        listen_allowed.set()
                        continue

                    if t in ("response.text.delta", "response.output_text.delta", "response.audio_transcript.delta"):
                        delta = msg.get("delta", "")
                        if delta:
                            ai_text_buffer += delta
                            arc_stream_buffer += delta
                            #print(delta, end="", flush=True)

                    # 🔥 ENHANCED: Capture user transcript
                    if t == "conversation.item.input_audio_transcription.completed":
                        transcript = (msg.get("transcript") or "").strip()
                        if transcript:
                            last_user_transcript = transcript
                            last_transcript_time = time.time()
                            print(f"\n🧾 USER TRANSCRIPT: {last_user_transcript}")
                        continue



                    elif t == "response.done":
                        response_in_progress = False

                        final_text = clean_ai_text(arc_stream_buffer)
                        arc_stream_buffer = ""
                        ai_text_buffer = ""
                        
                        if final_text:
                            # 🔑 Check if this is a command
                            handled = handle_commands(final_text)
                            
                            if isinstance(handled, tuple) and handled[0] == "CHROMADB":
                                router_query = handled[1]
                                transcript = last_user_transcript.strip()

                                combined_query = build_combined_db_query(router_query, transcript)

                                print(f"🧠 Combined DB query:\n  - Router: {router_query}\n  - Transcript: {transcript}")

                                success = await process_chromadb_query(
                                    ws,
                                    combined_query,
                                    router_query,
                                    transcript
                                )
                                reset_query_state()
                                response_in_progress = success
                                continue

                                # Try to get transcript immediately
                                question_for_db = last_user_transcript.strip()

                                # If missing, allow a SHORT grace window (max 0.4s)
                                if not question_for_db:
                                    now = time.time()
                                    if now - last_transcript_time < 0.4:
                                        await asyncio.sleep(0.1)
                                        question_for_db = last_user_transcript.strip()

                                if not question_for_db:
                                    print("⚠️ Transcript not available after grace window, skipping DB query")
                                    continue


                                success = await process_chromadb_query(ws, question_for_db)
                                response_in_progress = success
                                continue

                            # 🗣️ Speak if not a command
                            elif handled is None:
                                arc_say(final_text, streaming=False)
                            
                            elif handled is True:  # Other command was handled
                                print(f"✅ Command executed: {final_text[:50]}...")
                        
                        # Clear buffers
                        arc_stream_buffer = ""
                        ai_text_buffer = ""

                except Exception as e:
                    print(f"❌ Error in main loop: {e}")
                    response_in_progress = False
                    # Clear pending queries on error
                    break

# =======================
# RUN
# =======================
if __name__ == "__main__":
    try:
        print("\n👋 Starting Professor Dux...")
        print("🔄 Transcript synchronization system: ACTIVE")
        print("💡 System will wait up to 3 seconds for transcripts when needed")
        print("🧠 Initializing Faculty RAG System...")

        print("✅ Faculty RAG ready")

        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Exiting")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if tcp_client:
            tcp_client.close()
        stop_music()
