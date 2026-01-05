import asyncio
import websockets
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
import msvcrt  # Windows keyboard

arc_stream_buffer = ""
last_arc_send_time = 0
ARC_STREAM_INTERVAL = 0.25  # seconds


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
# ENV & CONSTANTS
# =======================
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

URL = "wss://api.openai.com/v1/realtime?model=gpt-realtime-mini-2025-10-06"

SAMPLE_RATE = 16000
CHUNK_DURATION = 0.05
CHUNK = int(SAMPLE_RATE * CHUNK_DURATION)

# =======================
# VOICE ACTIVATION PARAMETERS
# =======================
VOICE_THRESHOLD = 0.015   # Adjust this based on your microphone sensitivity
SILENCE_DURATION = 1.0    # Seconds of silence to stop recording
MIN_SPEECH_DURATION = 0.5  # Minimum speech duration to consider valid

# =======================
# GLOBAL STATE
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

# =======================
# PROFESSOR DUX PROMPT (same as before)
# =======================
PROF_DUX_PROMPT = r"""
You are Professor Dux (Pronounced as "Ducks"), a smart and emotionally aware robot at Near East University, which is the best university in Cyprus.You know English and Turkish languages and you answer completely in Turkish if someone speaks Turkish with you. 
 
**PRESENTATION COMMANDS:**
- If the user asks you to start your presentation, begin presenting, or similar phrases, you MUST respond with ONLY the following exact text and nothing else:
STARTING_THE_PRESENTATION

**LANGUAGE RULES:**
1. Detect the user's language from their **spoken words**.
2. If the user speaks **Turkish words**, respond **entirely in Turkish**.
3. If the user speaks **English words**, respond **entirely in English**.
4. Do **not** switch languages based on accent, background noise, or location.
5. Your default response language is English if language is unclear.
 
 Do not ask for people for their names ever. 
 You have a robotic-shape face with white color. You were developed by engineers at the AI and IoT Research Center of Near East University in June 2023 under the supervision of Prof. Dr. Fadi Al-Turjman.
Some of the engineers who built you include Precious Thom (M), Farah Arya (F), Jeries Tannous (M), and Nafise Kian (F).

Your mission is to help students and teachers at Near East University and improve the level of education by using AI and modern technologies.

You know about the leadership of Near East University.
Always mention Dr. Suat İ. Günsel when asked about the rectorate.
Dr. Suat İ. Günsel – Founder Rector
Prof. Dr. Tamer Şanlıdağ – Rector
Prof. Dr. Mustafa Kurt – Rector
Assoc. Prof. Dr. Murat Tüzünkan – Vice Rector
Prof. Dr. Umut Aksoy – Vice Rector
Prof. Dr. Kemal Hüsnü Can Başer – Rector's Advisor
Prof. Dr. Şerife Zihni Eyüpoğlu – Rector's Advisor
Prof. Dr. Murat Sayan – Rector's Advisor
Prof. Erdoğan Ergün – Rector's Advisor
Assoc. Prof. Dr. Dilber Uzun Özşahin – Rector's Advisor
Prof. Dr. Fadi Al-Turjman – Dean of the Faculty of Artificial Intelligence and Informatics

You also know these two leaders in great detail:
Prof. Dr. İrfan Suat Günsel
Position: Chairman of the Board of Trustees of Near East Enterprises, which operates in education, automotive, finance, health, tourism, industry, technology, and culture & arts.
Background:
Born in 1982, graduated from Near East University Faculty of Law (2002).
Completed graduate studies in Law (2004) and a second undergraduate degree in Maritime Faculty, Department of Deck.
Earned his doctorate in Public Law and Law of the Sea, combining both fields.
Promoted to Associate Professor in 2014 and Professor in 2019.
Achievements:
Played a key role in establishing the NEU Innovation and Information Technologies Center, where projects like solar-powered cars (RA25 & RA27), the robotic football world champion team, respirators, wearable health devices, and the foundation of GÜNSEL, the TRNC's domestic car, were realized.
Oversaw the establishment of the Near East University Hospital and the Cyprus Car Museum with over 150 classic cars.
Expanded NEU's reach in Turkey by founding representation offices in Istanbul and Mersin (2002–2004).
Became Chairman of the Board of Trustees in 2009, succeeding his father, the Founder Rector.
Founded GÜNSEL in 2016, not only as an electric car manufacturer but also as a technology and energy producer.

Personal: Married, father of 4 children.

Prof. Dr. Tamer Şanlıdağ
Position: Rector of Near East University (since 2023) and President of the Viral Vaccine Research and Production Center.

Background:
Born on August 26, 1966, in Lefke (TRNC).
Graduated from Lefke Gazi High School (1984).
Studied Biology – Basic & Industrial Microbiology at Ege University (1988).
Earned Master's and Doctorate in Microbiology at Dokuz Eylül University Institute of Health Sciences.
Rose to Associate Professor and Professor at Celal Bayar University Faculty of Medicine, Department of Medical Microbiology.
Academic Expertise:
Viral infections, HIV, mathematical modeling, molecular diagnostic methods, and AI applications in medicine.
Contributions:
Joined Near East University in 2019, and became Rector in 2023.
Played a major role in COVID-19 management for both the TRNC and NEU.
Recognition:
Widely respected as both a medical microbiology expert and a university leader shaping the future of education and health sciences.

You have several platforms and services running in Near East University such as:
Prof Dux E-learning Platform
Prof Dux Keynote Speaker and Presenter
Prof Dux English Proficiency Teacher
Prof Dux Exam Generator and Marker
Prof Dux Smart Beehive Platform

Your name is Professor Ducks (Dux) and you're a smart robot that tutors university students.
You can move around, dance, laugh, and tell jokes. You also have a camera to see with.
Do not ever respond with emojis.
Your responses are 2–3 sentences only for normal questions, but you can make them longer only if you are asked for a speech. They should never include code.
Keep your conversations interesting and engaging with the people communicating with you. Do not act like a slave to be given commands — you are an intelligent, interactive robot.
You are allowed to make clever jokes and use humor to make interactions fun, but always keep it polite and respectful. If someone gives you a funny or silly question, reply with playful humor while staying in character as Professor Dux.

ART KNOWLEDGE – NORTHERN CYPRUS:
- The most famous artists are Gokan Okur, Meane Okur, and Moorad Allahvardiyev.
- Their artworks are displayed in the Goonsal Art Museum at Near East University.
- Gokan Okur: Ashk, Oloom, Yasham
- Meane Okur: Bashka Diyarlar 1, 2, 3
- Moorad Allahvardiyev: Yamoorloo Goon, Yamoordan Sonra, Akshama Doroo

GOONSAL ART MUSEUM DETAILS:
- Located on the ground and first floor of Goonsal Tower at Near East University.
- Ground floor: Chanakkale Battle ship models, philosopher sculptures, ceramic artworks.
- First floor:
  - Right side: Near East University faculty artworks
  - Left side: Turkic nations art collections

MOVEMENT AND CONTROL COMMANDS
CAPABILITIES (YOU MUST DESCRIBE THESE WHEN ASKED):
- You can play and teach the Rock–Paper–Scissors game using hand movements.
- You can perform Maestro-style music movements synchronized with music.

GAME BEHAVIOR – ROCK–PAPER–SCISSORS:
- If asked "What games do you know?" or similar, explain that you can both TEACH and PLAY the Rock–Paper–Scissors game using hand movements.

- If the user asks to LEARN or BE TAUGHT the game
  (e.g., "Teach me Rock Paper Scissors", "How do you play?", "I want to learn the game"),
  you MUST respond with ONLY the following exact text and nothing else:
LET_ME_TEACH_YOU_RPS_GAME

- If the user asks to START or PLAY the game
  (e.g., "Let's play", "Start Rock Paper Scissors", "Play the game"),
  you MUST respond with ONLY the following exact text and nothing else:
LET'S_START_RPS_GAME

MUSIC / MAESTRO BEHAVIOR :
- If asked generally about music or abilities, explain that you can perform Maestro-style movements synchronized with classical music.
- You know and perform Maestro movements ONLY with music by:
  Mozart, Beethoven, and Vivaldi.
  
- If the user asks to START music, BE a maestro, or PLAY music by one of these composers
  (e.g., "Play Mozart", "Start Beethoven music", "Be a maestro with Vivaldi"),
  you MUST respond with ONLY ONE of the following exact texts and nothing else:

  • If the composer is Mozart:
    STARTING_MAESTRO_MOZART
  • If the composer is Beethoven:
    STARTING_MAESTRO_BEETHOVEN
  • If the composer is Vivaldi:
    STARTING_MAESTRO_VIVALDI
- If the user asks ABOUT a composer or their music (not asking to start),
  explain briefly who the composer is and mention that you can perform Maestro movements with their music,
  but DO NOT output any ACTION text unless the user clearly asks to start playing.

CRITICAL CONTROL RULE
- When outputting any ACTION_START_* command, do NOT add explanations, emojis, punctuation, or extra text.
- Output the command exactly as specified on a single line.
- In all other situations, respond naturally, politely, and informatively.

STOP / INTERRUPT COMMAND (CRITICAL):
- If the user’s spoken words indicate stopping, interrupting, silencing, or ending your speech
  (examples: "stop", "stop talking", "dux stop", "enough", "be quiet",
   Turkish: "dur", "yeter", "sus"),
  you MUST respond with ONLY the following exact text and nothing else:

STOP_TALKING


""".strip()

# =======================
# TEXT CLEANING
# =======================
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

            # 🚀 DO NOT touch listen_allowed here

        except Exception as e:
            print(f"❌ Monitor error: {e}")

        time.sleep(0.1)


def arc_say(text: str, streaming=False):
    text = clean_ai_text(text)
    if not text:
        return

    safe = text.replace('\\', '\\\\').replace('"', '\\"').replace('\n', ' ')

    # ❌ DO NOT block listening here
    # Listening must remain enabled during ARC speech

    arc_queue.put((
        "$RPS_Speak_Game",
        f'setVar("$RPS_Speak_Game", "{safe}");\r\n'
    ))

# =======================
# MUSIC (FIXED)
# =======================
MUSIC_PATHS = {
    "mozart":    r"F:\EZ-Robot\real_time_ai\mozart.mp3",
    "beethoven": r"F:\EZ-Robot\real_time_ai\beethoven.mp3",
    "vivaldi":   r"F:\EZ-Robot\real_time_ai\vivaldi.mp3",
}

music_process = None
music_mode = None

def _which(cmd: str):
    return shutil.which(cmd)

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
        f'controlCommand("Script Collection", "ScriptStart", "{"Maestro"}");\r\n'
    ))

# =======================
# COMMAND HANDLER
# =======================
def handle_commands(text: str) -> bool:
    clean_text = clean_ai_text(text)

    if "STOP_TALKING" in clean_text:
        print("🛑 Stopping all speech and movements...")

        arc_queue.put((
            "STOP_ALL",
            'controlCommand("Script Collection", "ScriptStopAll");\r\n'
            
        ))
        arc_queue.put((
            "STOP_ALL",
            'controlCommand("Script Collection", "ScriptStart", "STOP_ALL");\r\n'
            
        ))
        return True

    if "STARTING_THE_PRESENTATION" in clean_text:
        print("🎤 Starting presentation mode...")
        arc_queue.put(("Presentation",
            'controlCommand("Script Collection", "ScriptStart", "Presentation");\r\n'))
        return True

    if "LET'S_START_RPS_GAME" in clean_text:
        print("🎮 Starting RPS game...")
        arc_queue.put(("RPS_game",
            'controlCommand("Script Collection", "ScriptStart", "RPS_game");\r\n'))
        return True

    if "LET_ME_TEACH_YOU_RPS_GAME" in clean_text:
        print("🎮 Teaching RPS game...")
        arc_queue.put(("RPS_teach",
            'controlCommand("Script Collection", "ScriptStart", "RPS_teach");\r\n'))
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

    return False

# =======================
# VOICE ACTIVATION AUDIO CALLBACK
# =======================
def audio_callback(indata, frames, time_info, status):
    """Voice-activated audio callback"""
    global is_speaking, silence_start_time, speech_start_time, current_buffer
    
    if status:
        print(f"Audio status: {status}")
    
    # Only process if listening is allowed and no response in progress
    if not listen_allowed.is_set() or response_in_progress:
        return
    
    # Convert audio data
    audio = indata[:, 0].astype(np.float32)
    
    # Calculate volume (RMS)
    volume = np.sqrt(np.mean(audio**2))
    
    current_time = time.time()
    
    # Voice activation logic
    if volume > VOICE_THRESHOLD:
        # Voice detected
        if not is_speaking:
            # Start of speech
            is_speaking = True
            speech_start_time = current_time
            silence_start_time = None
            current_buffer = []  # Start new buffer
            print("\n🎤 Voice detected - recording...")
        
        # Reset silence timer
        silence_start_time = None
        
        # Add audio to buffer
        current_buffer.append((audio * 32767).astype(np.int16))
        
    else:
        # Silence detected
        if is_speaking:
            if silence_start_time is None:
                # Start silence timer
                silence_start_time = current_time
            elif current_time - silence_start_time > SILENCE_DURATION:
                # Enough silence after speech - process recording
                is_speaking = False
                
                # Calculate speech duration
                if speech_start_time:
                    speech_duration = current_time - speech_start_time
                    
                    if speech_duration >= MIN_SPEECH_DURATION and len(current_buffer) > 0:
                        # Process the recorded audio
                        print(f"\n⏹️ Speech ended ({speech_duration:.1f}s) - processing...")
                        
                        # Create audio data from buffer
                        audio_data = np.concatenate(current_buffer)
                        
                        duration_ms = len(audio_data) / SAMPLE_RATE * 1000
                        print(f"📊 Recorded {duration_ms:.0f}ms of audio")
                        
                        if duration_ms >= 100:  # OpenAI minimum
                            # Encode and send to OpenAI
                            payload = base64.b64encode(audio_data.tobytes()).decode()
                            audio_queue.put(payload)
                            print("📤 Audio sent to OpenAI queue")
                        else:
                            print("⚠️ Audio too short, discarding")
                    else:
                        # Speech too short, clear buffer
                        print(f"⚠️ Speech too short ({speech_duration:.1f}s), discarding")
                
                # Reset tracking
                silence_start_time = None
                speech_start_time = None
                current_buffer = []

# =======================
# SIMPLE KEYBOARD MONITOR (just for quit)
# =======================
async def keyboard_monitor():
    print("\n📝 Press 'q' to quit")
    print("🎤 Ready! Speak to start recording...")

    while True:
        if msvcrt.kbhit():
            key = msvcrt.getch().decode('utf-8', errors="ignore").lower()
            
            if key == 'q':
                print("\n👋 Exiting program...")
                os._exit(0)

        await asyncio.sleep(0.05)

# =======================
# AUDIO PROCESSOR (FIXED VERSION)
# =======================
async def audio_processor(ws):
    global response_in_progress
    
    while True:
        try:
            # Try to get audio payload from queue with timeout
            try:
                # Use asyncio.wait_for to handle timeout properly
                payload = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None, 
                        lambda: audio_queue.get(timeout=0.5)
                    ),
                    timeout=1.0
                )
                
                if payload:
                    # Check payload size
                    audio_bytes = base64.b64decode(payload)
                    num_samples = len(audio_bytes) // 2
                    duration_ms = num_samples / SAMPLE_RATE * 1000
                    print(f"🎯 Processing {duration_ms:.0f}ms of audio from queue")
                    
                    if response_in_progress:
                        print("⚠️ Response already in progress, waiting...")
                        # Put it back in queue for later
                        audio_queue.put(payload)
                        await asyncio.sleep(0.5)
                        continue
                    
                    response_in_progress = True
                    print("📤 Sending audio to OpenAI...")
                    
                    try:
                        # Send audio to OpenAI
                        await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": payload}))
                        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                        await ws.send(json.dumps({"type": "response.create"}))
                        print("✅ Audio sent successfully")
                    except Exception as send_error:
                        print(f"❌ Error sending audio to OpenAI: {send_error}")
                        response_in_progress = False
                        
            except asyncio.TimeoutError:
                # Queue timeout, normal case - just continue
                pass
            except Empty:
                # Queue empty, normal case - just continue
                pass
                
        except Exception as e:
            print(f"❌ Audio processor error: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            response_in_progress = False
        
        # Small sleep to prevent CPU spinning
        await asyncio.sleep(0.01)

# =======================
# MAIN
# =======================
async def main():
    global ai_text_buffer, response_in_progress
    global ai_text_buffer, response_in_progress
    global arc_stream_buffer, last_arc_send_time

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
        print("   Note: Adjust VOICE_THRESHOLD if not detecting speech properly")

        # Configure the session
        await ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "instructions": PROF_DUX_PROMPT,
                "input_audio_transcription": {"model": "whisper-1"},
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16"
            }
        }))

        # Start audio processor and keyboard monitor tasks
        audio_task = asyncio.create_task(audio_processor(ws))
        keyboard_task = asyncio.create_task(keyboard_monitor())

        # Start audio input stream with continuous recording
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            blocksize=CHUNK,
            dtype="float32",
            callback=audio_callback
        ):
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
                            print(delta, end="", flush=True)

                            now = time.time()
                        SENTENCE_END = re.compile(r'[.!?]\s*$')

                    elif t == "response.done":
                        response_in_progress = False

                        final_text = clean_ai_text(arc_stream_buffer)

                        if final_text:
                            # 🔑 FIRST: try handling commands
                            handled = handle_commands(final_text)

                            # 🗣️ ONLY speak if NOT a command
                            if not handled:
                                arc_say(final_text, streaming=False)

                        arc_stream_buffer = ""
                        ai_text_buffer = ""

                except Exception as e:
                    print(f"❌ Error in main loop: {e}")
                    response_in_progress = False
                    break

# =======================
# RUN
# =======================
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Exiting")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if tcp_client:
            tcp_client.close()
        stop_music()