import json
import os
import chromadb
from chromadb.utils import embedding_functions
from textblob import TextBlob
from typing import Dict, Any, List

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import EMBEDDING_MODEL

# --- CONFIGURATION ---
CHROMA_PATH = "~/.chroma_data"
INPUT_FILE = "facebook_messages.json"
COLLECTION_NAME = "episodic_memory"
BATCH_SIZE = 1000

# Importance keywords (heuristic)
IMPORTANCE_KEYWORDS = {
    "moved to": 0.8,
    "new job": 0.9,
    "got married": 1.0,
    "engaged": 0.9,
    "born": 0.9,
    "graduated": 0.9,
    "trip to": 0.7,
    "traveling to": 0.7,
    "started at": 0.8,
    "resigned": 0.8,
    "break up": 0.8,
    "broke up": 0.8,
    "bought a": 0.6,
    "apartment": 0.7,
    "house": 0.7,
}

# Emotion keywords mapping (simple heuristic fallback)
EMOTION_KEYWORDS = {
    "joy": ["happy", "excited", "love", "great", "wonderful", "amazing", "so nice"],
    "sadness": ["sad", "unhappy", "sorry", "miss", "grief", "bad news"],
    "anger": ["angry", "mad", "hate", "furious", "annoyed"],
    "fear": ["scared", "afraid", "worried", "nervous"],
    "surprise": ["wow", "omg", "shocked", "unbelievable"],
}

def analyze_emotion(text: str) -> Dict[str, Any]:
    """
    Analyze text for emotion, valence, and arousal.
    Returns a dictionary with emotion metrics.
    """
    blob = TextBlob(text)
    
    # Polaris (Valence): -1.0 (negative) to 1.0 (positive)
    # We normalize to 0.0 to 1.0
    valence = (blob.sentiment.polarity + 1) / 2
    
    # Subjectivity (proxy for Arousal? Not really, but often correlated with intensity)
    # 0.0 (objective) to 1.0 (subjective)
    arousal = blob.sentiment.subjectivity
    
    # Detect specific emotion category based on keywords
    detected_emotion = "neutral"
    text_lower = text.lower()
    
    for emotion, keywords in EMOTION_KEYWORDS.items():
        if any(k in text_lower for k in keywords):
            detected_emotion = emotion
            break
            
    # Adjust arousal if strong emotion detected
    if detected_emotion != "neutral" and arousal < 0.5:
        arousal = 0.5 + (arousal * 0.5)
        
    return {
        "emotion": detected_emotion,
        "emotion_arousal": round(arousal, 2),
        "emotional_valence": round(valence, 2)
    }

def calculate_importance(text: str) -> float:
    """Calculate importance score based on keywords."""
    text_lower = text.lower()
    max_score = 0.1  # Default baseline importance
    
    for keyword, score in IMPORTANCE_KEYWORDS.items():
        if keyword in text_lower:
            max_score = max(max_score, score)
            
    return max_score

def main():
    print(f"Loading messages from {INPUT_FILE}...")
    try:
        with open(INPUT_FILE, "r") as f:
            messages = json.load(f)
    except FileNotFoundError:
        print(f"Error: {INPUT_FILE} not found.")
        return

    print(f"Loaded {len(messages)} messages.")

    # Embedding function
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL, device=device
    )

    # Database
    chroma_rect_path = os.path.expanduser(CHROMA_PATH)
    client = chromadb.PersistentClient(path=chroma_rect_path)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_func
    )
    
    # Filter for potential episodes first to save processing time
    episodes_to_add = []
    
    print("Filtering and analyzing episodes...")
    for msg in messages:
        text = msg.get("text", "")
        if not text:
            continue
            
        importance = calculate_importance(text)
        
        # Only keep interesting episodes (threshold > 0.3)
        if importance > 0.3:
            emotion_data = analyze_emotion(text)
            
            episode = {
                "summary": text,
                "date": msg.get("date", "Unknown"),
                "sender_name": msg.get("sender_name", "Unknown"),
                "conversation": msg.get("conversation", "Unknown"),
                "importance": importance,
                "source": "facebook",
                **emotion_data
            }
            episodes_to_add.append(episode)

    print(f"Found {len(episodes_to_add)} significant episodes.")
    
    # Batch insert
    total_batches = (len(episodes_to_add) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for i in range(0, len(episodes_to_add), BATCH_SIZE):
        batch = episodes_to_add[i:i+BATCH_SIZE]
        
        documents = [e["summary"] for e in batch]
        metadatas = batch
        ids = [f"episode_{i+j}_{e['date']}" for j, e in enumerate(batch)]
        
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Inserted batch {i//BATCH_SIZE + 1}/{total_batches}")

    print("✅ Episodic Memory construction complete.")

if __name__ == "__main__":
    main()
