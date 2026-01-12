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

class EnhancedFacultyRAGSystem:
    """Enhanced RAG system with fuzzy name matching and better extraction."""
    
    def __init__(self):
        """Initialize enhanced RAG system."""
        print("Initializing Enhanced RAG System...")
        
        # Load QA model
        print(f"Loading QA model from: {CONFIG['model_path']}")
        self.tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_path"])
        self.model = AutoModelForQuestionAnswering.from_pretrained(CONFIG["model_path"])
        self.model.eval()
        print("Model loaded successfully")
        
        # Initialize ChromaDB
        print(f"Connecting to ChromaDB: {CONFIG['chromadb_path']}")
        self.client = chromadb.PersistentClient(path=CONFIG["chromadb_path"])
        self.collection = self.client.get_collection(CONFIG["collection_name"])
        total_docs = self.collection.count()
        print(f"ChromaDB connected. Documents: {total_docs}")
        
        # Pre-load all faculty names and metadata for fuzzy matching
        self.all_faculty_names = []
        self.all_faculty_data = []  # Store name + metadata
        self._load_all_faculty_data()
        print(f"Loaded {len(self.all_faculty_names)} faculty names for fuzzy matching")
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"Device: {self.device}")
    
    def _load_all_faculty_data(self):
        """Load all faculty names and metadata from ChromaDB."""
        try:
            # Get all documents with metadata
            all_docs = self.collection.get(include=["metadatas"])
            self.all_faculty_names = []
            self.all_faculty_data = []
            
            if all_docs and "metadatas" in all_docs and all_docs["metadatas"]:
                for metadata in all_docs["metadatas"]:
                    if metadata and "name" in metadata:
                        name = metadata["name"].strip()
                        if name and name not in self.all_faculty_names:
                            self.all_faculty_names.append(name)
                            self.all_faculty_data.append({
                                "name": name,
                                "name_lower": name.lower(),
                                "name_words": name.lower().split(),
                                "metadata": metadata
                            })
            
            # Also create variations for better matching
            self._create_name_variations()
            
        except Exception as e:
            print(f"Warning: Could not load faculty data: {e}")
    
    def _create_name_variations(self):
        """Create name variations for better fuzzy matching."""
        name_variations = []
        
        for faculty in self.all_faculty_data:
            name = faculty["name"]
            name_lower = faculty["name_lower"]
            
            # Extract first name and last name
            parts = name_lower.split()
            if len(parts) >= 2:
                # First name only
                first_name = parts[0]
                if first_name not in name_variations:
                    name_variations.append(first_name)
                    self.all_faculty_data.append({
                        "name": name,
                        "name_lower": first_name,
                        "name_words": [first_name],
                        "metadata": faculty["metadata"],
                        "is_variation": True
                    })
                
                # Last name only
                last_name = parts[-1]
                if last_name not in name_variations:
                    name_variations.append(last_name)
                    self.all_faculty_data.append({
                        "name": name,
                        "name_lower": last_name,
                        "name_words": [last_name],
                        "metadata": faculty["metadata"],
                        "is_variation": True
                    })
    
    def _correct_spelling(self, text: str) -> str:
        """Correct common spelling mistakes in the query."""
        if not CONFIG.get("spell_corrections"):
            return text
        
        corrected = text.lower()
        for wrong, correct in CONFIG["spell_corrections"].items():
            if wrong in corrected:
                corrected = corrected.replace(wrong, correct)
        
        # Only return corrected version if it's actually different
        if corrected != text.lower():
            return corrected
        return text
    
    def find_closest_name(self, query: str) -> Tuple[Optional[str], int, Optional[Dict]]:
        """
        Find the closest matching faculty name using multiple matching strategies.
        Returns (matched_name, match_score, matched_metadata) or (None, 0, None).
        """
        if not self.all_faculty_data:
            return None, 0, None
        
        original_query = query
        query = self._correct_spelling(query)
        
        if CONFIG["enable_debug"]:
            print(f"🔍 Name matching: '{original_query}' -> '{query}'")
        
        # Strategy 1: Direct substring match
        for faculty in self.all_faculty_data:
            if faculty["name_lower"] in query or query in faculty["name_lower"]:
                if CONFIG["enable_debug"]:
                    print(f"  ✓ Direct substring match: '{faculty['name']}'")
                return faculty["name"], 100, faculty["metadata"]
        
        # Strategy 2: Extract potential names from query
        potential_names = self._extract_name_candidates(query)
        
        # Strategy 3: Word overlap matching
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        
        best_match = None
        best_score = 0
        best_metadata = None
        
        for faculty in self.all_faculty_data:
            score = 0
            
            # Check word overlap
            name_words = set(faculty["name_words"])
            overlap = query_words.intersection(name_words)
            if overlap:
                score = len(overlap) * 30  # 30 points per overlapping word
            
            # Check for partial name matches
            for q_word in query_words:
                for n_word in faculty["name_words"]:
                    if len(q_word) > 2 and len(n_word) > 2:
                        if q_word in n_word or n_word in q_word:
                            score += 20
                        elif fuzz.ratio(q_word, n_word) > 80:
                            score += 15
            
            # Check extracted potential names
            for pot_name in potential_names:
                if pot_name.lower() in faculty["name_lower"] or faculty["name_lower"] in pot_name.lower():
                    score += 40
            
            # Use fuzzy matching as fallback
            if score < 50:  # Only use fuzzy if other methods didn't find good match
                fuzzy_score = fuzz.partial_ratio(query, faculty["name_lower"])
                if fuzzy_score > best_score:
                    score = max(score, fuzzy_score * 0.8)  # Weight fuzzy matches slightly lower
            
            if score > best_score and score >= 40:  # Minimum threshold
                best_score = score
                best_match = faculty["name"]
                best_metadata = faculty["metadata"]
        
        if best_match and best_score >= CONFIG["fuzzy_match_threshold"]:
            if CONFIG["enable_debug"]:
                print(f"  ✓ Best match found: '{best_match}' (score: {best_score})")
            return best_match, best_score, best_metadata
        
        if CONFIG["enable_debug"]:
            print(f"  ✗ No good match found (best score: {best_score})")
        return None, 0, None
    
    def _extract_name_candidates(self, query: str) -> List[str]:
        """Extract potential name candidates from query."""
        candidates = []
        
        # Remove common titles and words
        stop_words = {"prof", "dr", "professor", "doctor", "room", "office", "background", 
                     "education", "degree", "qualification", "where", "what", "is", "the", 
                     "of", "for", "and", "to", "in", "at"}
        
        words = re.findall(r'\b\w+\b', query.lower())
        filtered_words = [w for w in words if w not in stop_words and len(w) > 2]
        
        # Create combinations of filtered words
        if len(filtered_words) >= 2:
            candidates.append(' '.join(filtered_words))
        
        # Add individual words that might be names
        for word in filtered_words:
            if word[0].isupper() or len(word) > 3:
                candidates.append(word)
        
        return candidates
    
    def retrieve_with_name_focus(self, query: str, matched_name: str = None, matched_metadata: Dict = None) -> List[Dict]:
        """
        Enhanced retrieval that focuses on name matching with multiple strategies.
        """
        contexts = []
        
        try:
            # STRATEGY 1: Direct metadata filtering (if we have matched name)
            if matched_name:
                if CONFIG["enable_debug"]:
                    print(f"  📂 Searching for exact name: '{matched_name}'")
                
                try:
                    # Try exact match first
                    exact_results = self.collection.get(
                        where={"name": {"$eq": matched_name}}
                    )
                    
                    if exact_results and exact_results["documents"]:
                        for doc, meta in zip(exact_results["documents"], exact_results["metadatas"]):
                            contexts.append({
                                "text": doc,
                                "metadata": meta,
                                "distance": 0.0,
                                "relevance": 2.0,  # High relevance for exact match
                                "name_match": True,
                                "source": "exact_name_filter"
                            })
                            if CONFIG["enable_debug"]:
                                print(f"    ✓ Found {len(exact_results['documents'])} exact matches")
                except Exception as e:
                    if CONFIG["enable_debug"]:
                        print(f"    Note: Exact filter failed: {e}")
            
            # STRATEGY 2: Semantic search with the query
            if CONFIG["enable_debug"]:
                print(f"  🔍 Semantic search for: '{query}'")
            
            semantic_results = self.collection.query(
                query_texts=[query],
                n_results=CONFIG["n_retrieve"],
                include=["documents", "metadatas", "distances"]
            )
            
            if semantic_results["documents"] and semantic_results["documents"][0]:
                for doc, meta, dist in zip(
                    semantic_results["documents"][0],
                    semantic_results["metadatas"][0],
                    semantic_results["distances"][0]
                ):
                    # Skip if already in contexts
                    if any(ctx["text"] == doc for ctx in contexts):
                        continue
                    
                    relevance = 1.0 / (1.0 + float(dist))
                    
                    # Boost if name matches
                    name_match = False
                    if matched_name:
                        meta_name = meta.get("name", "").lower() if meta else ""
                        if matched_name.lower() in meta_name or meta_name in matched_name.lower():
                            relevance *= 1.8
                            name_match = True
                    
                    contexts.append({
                        "text": doc,
                        "metadata": meta,
                        "distance": float(dist),
                        "relevance": relevance,
                        "name_match": name_match,
                        "source": "semantic_search"
                    })
                
                if CONFIG["enable_debug"]:
                    print(f"    ✓ Found {len(semantic_results['documents'][0])} semantic matches")
            
            # STRATEGY 3: If still not enough, search with name only
            if matched_name and len(contexts) < 3:
                if CONFIG["enable_debug"]:
                    print(f"  🔍 Additional search with name only: '{matched_name}'")
                
                name_only_results = self.collection.query(
                    query_texts=[matched_name],
                    n_results=3,
                    include=["documents", "metadatas", "distances"]
                )
                
                if name_only_results["documents"] and name_only_results["documents"][0]:
                    for doc, meta, dist in zip(
                        name_only_results["documents"][0],
                        name_only_results["metadatas"][0],
                        name_only_results["distances"][0]
                    ):
                        # Skip if already in contexts
                        if any(ctx["text"] == doc for ctx in contexts):
                            continue
                        
                        relevance = 1.0 / (1.0 + float(dist)) * 1.5  # Boost for name search
                        
                        contexts.append({
                            "text": doc,
                            "metadata": meta,
                            "distance": float(dist),
                            "relevance": relevance,
                            "name_match": True,
                            "source": "name_only_search"
                        })
            
            # Remove duplicates and sort by relevance
            unique_contexts = []
            seen_texts = set()
            
            for ctx in contexts:
                if ctx["text"] not in seen_texts:
                    seen_texts.add(ctx["text"])
                    unique_contexts.append(ctx)
            
            # Sort by relevance (highest first)
            unique_contexts.sort(key=lambda x: x["relevance"], reverse=True)

            # 🔥 TAKE ONLY THE BEST MATCH
            best_context = unique_contexts[0] if unique_contexts else None

            if CONFIG["enable_debug"] and best_context:
                name = best_context["metadata"].get("name", "Unknown")
                print(f"  🏆 Selected BEST context: {name} (rel: {best_context['relevance']:.2f})")

            return [best_context] if best_context else []

            
        except Exception as e:
            print(f"Error in retrieval: {e}")
            if CONFIG["enable_debug"]:
                import traceback
                traceback.print_exc()
            return []
    
    def extract_better_answer(self, question: str, context: str) -> Tuple[str, float]:
        """
        Enhanced answer extraction with better post-processing.
        """
        if not context or not question:
            return "", 0.0
        
        try:
            # Prepare input for the model
            inputs = self.tokenizer(
                question,
                context,
                max_length=384,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Find the best answer across all chunks
            best_answer = ""
            best_confidence = 0.0
            best_start_probs = []
            best_end_probs = []
            
            # Process each chunk
            for i in range(inputs["input_ids"].shape[0]):
                start_logits = outputs.start_logits[i]
                end_logits = outputs.end_logits[i]
                
                # Get probabilities
                start_probs = torch.softmax(start_logits, dim=0)
                end_probs = torch.softmax(end_logits, dim=0)
                
                # Get most likely positions
                start_idx = torch.argmax(start_logits).item()
                end_idx = torch.argmax(end_logits).item()
                
                # Calculate confidence
                start_conf = start_probs[start_idx].item()
                end_conf = end_probs[end_idx].item()
                confidence = (start_conf + end_conf) / 2.0
                
                # Get answer tokens
                answer_tokens = inputs["input_ids"][i][start_idx:end_idx+1]
                answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
                
                # Keep the best answer
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_answer = answer
                    best_start_probs = start_probs
                    best_end_probs = end_probs
            
            # Enhanced cleaning
            cleaned_answer = self._enhanced_clean_answer(best_answer, context, question)
            
            if CONFIG["enable_debug"]:
                print(f"  🎯 Extraction confidence: {best_confidence:.2f}")
                print(f"  📝 Raw answer: '{best_answer}'")
                print(f"  ✨ Cleaned answer: '{cleaned_answer}'")
            
            return cleaned_answer, best_confidence
            
        except Exception as e:
            print(f"Error in answer extraction: {e}")
            if CONFIG["enable_debug"]:
                import traceback
                traceback.print_exc()
            return "", 0.0
    
    def _enhanced_clean_answer(self, answer: str, context: str, question: str) -> str:
        """
        Enhanced answer cleaning with context awareness.
        """
        if not answer:
            return ""
        
        # Clean the answer
        answer = answer.strip()
        
        # Remove artifacts
        artifacts = ["[CLS]", "[SEP]", "[PAD]", "##", "\n", "\t", "  "]
        for art in artifacts:
            answer = answer.replace(art, " ")
        
        # Fix common issues
        answer = re.sub(r'\s+([.,;!?])', r'\1', answer)  # Remove space before punctuation
        answer = re.sub(r'([.,;!?])\s+', r'\1 ', answer)  # Add space after punctuation
        
        # Fix "303. 0" type issues
        answer = re.sub(r'(\d+)\.\s+0\b', r'\1', answer)
        
        # If answer is too short or doesn't make sense, try to extract from context
        if len(answer.split()) < 2 or answer.isdigit():
            question_lower = question.lower()
            
            # Try to extract office/room information
            if "room" in question_lower or "office" in question_lower:
                room_patterns = [
                    r'room\s+(\d+[A-Z]?(?:\s*/\s*\d+[A-Z]?)?)',
                    r'office\s+(\d+[A-Z]?(?:\s*[A-Za-z\s]+)?)',
                    r'(\d+[A-Z]?)\s*(?:room|office|rm|ofc)',
                    r'at\s+(\d+[A-Z]?(?:\s*[A-Za-z\s]+)?)',
                ]
                
                for pattern in room_patterns:
                    match = re.search(pattern, context, re.IGNORECASE)
                    if match:
                        room_info = match.group(1).strip()
                        if len(room_info) > len(answer) or not answer:
                            answer = f"Room {room_info}"
                            break
            
            # Try to extract background/education information
            elif "background" in question_lower or "education" in question_lower or "degree" in question_lower:
                background_patterns = [
                    r'background[:\s]+([^\.]+\.)',
                    r'education[:\s]+([^\.]+\.)',
                    r'degree[:\s]+([^\.]+\.)',
                    r'ph\.?d\.?\s+in\s+([^\.]+)',
                    r'master[\'s]?\s+in\s+([^\.]+)',
                    r'bachelor[\'s]?\s+in\s+([^\.]+)',
                ]
                
                for pattern in background_patterns:
                    match = re.search(pattern, context, re.IGNORECASE)
                    if match:
                        bg_info = match.group(1).strip()
                        if len(bg_info.split()) > len(answer.split()):
                            answer = bg_info
                            break
        
        # Final cleanup
        answer = " ".join(answer.split())  # Normalize whitespace
        
        # Ensure proper capitalization for first letter
        if answer and answer[0].islower():
            answer = answer[0].upper() + answer[1:]
        
        # Add period if missing and it looks like a sentence
        if answer and len(answer.split()) > 3 and answer[-1] not in '.!?':
            answer += '.'
        
        return answer
    
    def generate_direct_answer_from_context(self, contexts: List[Dict], question: str) -> str:
        """
        Generate a direct answer from context when extraction fails.
        """
        if not contexts:
            return "I couldn't find any information about that in the faculty database."
        
        # Get the most relevant contexts
        top_contexts = contexts[:1]  # STRICT: only best match
 # Use top 2 for more comprehensive answer
        
        # Extract information from all relevant contexts
        all_names = []
        all_backgrounds = []
        all_offices = []
        
        for ctx in top_contexts:
            metadata = ctx.get("metadata", {})
            
            if metadata:
                name = metadata.get("name", "").strip()
                if name and name not in all_names:
                    all_names.append(name)
                
                background = metadata.get("academic_background", "").strip()
                if background and background not in all_backgrounds:
                    all_backgrounds.append(background)
                
                office = metadata.get("office", "").strip()
                if office and office not in all_offices:
                    all_offices.append(office)
        
        question_lower = question.lower()
        
        # Generate answer based on question type
        if "background" in question_lower or "degree" in question_lower or "education" in question_lower:
            if all_backgrounds:
                names_str = ", ".join(all_names) if all_names else "The faculty member"
                backgrounds_str = "; ".join(all_backgrounds)
                return f"{names_str} has the following academic background: {backgrounds_str}"
            elif all_names:
                return f"I found information about {', '.join(all_names)}, but their academic background is not specified in the database."
            else:
                return "I couldn't find specific background information for that query."
        
        elif "office" in question_lower or "room" in question_lower or "where" in question_lower:
            if all_offices:
                names_str = ", ".join(all_names) if all_names else "The faculty member"
                offices_str = "; ".join(all_offices)
                return f"{names_str}'s office is located at: {offices_str}"
            elif all_names:
                return f"I found information about {', '.join(all_names)}, but their office location is not specified in the database."
            else:
                return "I couldn't find specific office information for that query."
        
        else:
            # Generic informative answer
            parts = []
            if all_names:
                parts.append(f"Name(s): {', '.join(all_names)}")
            if all_backgrounds:
                parts.append(f"Academic Background: {'; '.join(all_backgrounds)}")
            if all_offices:
                parts.append(f"Office(s): {'; '.join(all_offices)}")
            
            if parts:
                return f"Here's what I found: {'; '.join(parts)}."
            else:
                return "I found some information, but no specific details are available in the database."
    
    def answer_question(self, question: str) -> Dict:
        """
        Main function to answer a question with enhanced capabilities.
        """
        print(f"\n{'='*60}")
        print(f"📝 PROCESSING: '{question}'")
        print(f"{'='*60}")
        
        # Step 0: Preprocess question (spelling correction)
        original_question = question
        question = self._correct_spelling(question)
        if question != original_question and CONFIG["enable_debug"]:
            print(f"  ✏️  Spelling corrected: '{original_question}' -> '{question}'")
        
        # Step 1: Find closest name match
        closest_name, match_score, matched_metadata = self.find_closest_name(question)
        
        if closest_name and match_score >= CONFIG["fuzzy_match_threshold"]:
            print(f"  ✅ Name matched: '{closest_name}' (score: {match_score}%)")
            enhanced_query = f"{question} {closest_name}"
        else:
            closest_name = None
            matched_metadata = None
            enhanced_query = question
            print(f"  ⚠️  No strong name match found")
        
        # Step 2: Retrieve with name focus
        contexts = self.retrieve_with_name_focus(enhanced_query, closest_name, matched_metadata)
        
        if not contexts:
            return {
                "answer": "I couldn't find any relevant information in the faculty database.",
                "confidence": 0.0,
                "matched_name": closest_name,
                "match_score": match_score if closest_name else 0,
                "strategy": "no_contexts",
                "contexts_found": 0,
                "timestamp": datetime.now().isoformat()
            }
        
        print(f"  📚 Retrieved {len(contexts)} relevant contexts")
        
        # Step 3: Try to extract answer from combined context
        combined_context = contexts[0]["text"]
        extracted_answer, confidence = self.extract_better_answer(question, combined_context)
        
        # Step 4: Determine response strategy
        if extracted_answer and confidence >= CONFIG["confidence_threshold"]:
            strategy = "extracted"
            answer = extracted_answer
            print(f"  ✅ Answer extracted (confidence: {confidence:.2f})")
            
        elif contexts and closest_name:
            # Use direct generation with matched name
            strategy = "generated_with_match"
            answer = self.generate_direct_answer_from_context(contexts, question)
            print(f"  📝 Generated answer based on matched name")
            
        elif contexts:
            # Use direct generation without name match
            strategy = "generated"
            answer = self.generate_direct_answer_from_context(contexts, question)
            print(f"  📝 Generated answer from retrieved contexts")
            
        else:
            strategy = "no_answer"
            answer = "I couldn't generate a specific answer based on the available information."
        
        return {
            "answer": answer,
            "confidence": confidence,
            "matched_name": closest_name,
            "match_score": match_score if closest_name else 0,
            "strategy": strategy,
            "contexts_found": len(contexts),
            "timestamp": datetime.now().isoformat()
        }

print("i am here 2")
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


* If the user asks about, person, charater, docotor, or professor's backgraund or room "office"
  (e.g., "tell where is the room for professor fadi ?", who is fadi?, or tell me what you know about ramiz for example),
  you MUST respond with ONLY the following exact text and nothing else:
GETTING_INFO_FROM_CHROMADB

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
    if "GETTING_INFO_FROM_CHROMADB" in clean_text:
        return "CHROMADB"
    
    return None

async def process_chromadb_query(ws, question_for_db: str):
    """Process ChromaDB query and send result to OpenAI"""
    try:
        print(f"📚 Querying ChromaDB for: {question_for_db}")
        
        # FIX: Initialize ChromaDB here
        try:
        
            # Initialize fresh instance

            response = rag_system.answer_question(question_for_db)
            
            # Ensure response is a dict
            if not isinstance(response, dict):
                response = {"answer": str(response)}
                
        except Exception as e:
            print(f"❌ ChromaDB initialization error: {e}")
            import traceback
            traceback.print_exc()
            response = {"answer": "I encountered an error accessing the faculty database."}
        
        # Extract answer safely
        db_answer = ""
        if isinstance(response, dict):
            db_answer = response.get("answer", "").strip()
        elif isinstance(response, str):
            db_answer = response.strip()
        else:
            db_answer = str(response).strip()
        
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
        User asked: "{question_for_db}"
        Information from faculty database: "{db_answer}"
        Please answer the user naturally using this information.
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
                            print(delta, end="", flush=True)

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
                            
                            if handled == "CHROMADB":
                                print("🔄 ChromaDB command triggered!")

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
        rag_system = EnhancedFacultyRAGSystem()
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
