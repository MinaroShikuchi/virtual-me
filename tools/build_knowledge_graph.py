
import json
import os
from neo4j import GraphDatabase
from textblob import TextBlob
from datetime import datetime

# --- CONFIGURATION ---
NEO4J_URI = "bolt+s://neo4j.detour.team:7687"
NEO4J_AUTH = ("neo4j", "nostalgia")
MESSAGES_FILE = "facebook_messages.json"

# --- HELPERS ---

def analyze_sentiment(text):
    blob = TextBlob(text)
    return {
        "polarity": blob.sentiment.polarity,
        "subjectivity": blob.sentiment.subjectivity
    }

class KnowledgeGraphBuilder:
    def __init__(self, uri, auth):
        self.driver = GraphDatabase.driver(uri, auth=auth)

    def close(self):
        self.driver.close()

    def create_constraints(self):
        with self.driver.session() as session:
            # Create constraints for performance and data integrity
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Person) REQUIRE p.name IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (m:Message) REQUIRE m.id IS UNIQUE")
            print("✅ Constraints created.")

    def ingest_messages(self, messages, limit=None):
        with self.driver.session() as session:
            count = 0
            for msg in messages:
                if limit and count >= limit:
                    break
                
                text = msg.get("text", "")
                if not text:
                    continue

                sender_name = msg.get("sender_name", "Unknown")
                date_str = msg.get("date", "")
                conversation_id = msg.get("conversation", "unknown_conversation")
                
                # Sentiment Analysis
                sentiment = analyze_sentiment(text)
                
                # Create Graph Nodes & Relationships
                session.execute_write(self._create_message_node, 
                                      sender_name, text, date_str, conversation_id, sentiment)
                
                count += 1
                if count % 100 == 0:
                    print(f"Processed {count} messages...")

    @staticmethod
    def _create_message_node(tx, sender_name, text, date, conversation_id, sentiment):
        # We assume messages are unique by a combination of sender, date, and text hash, 
        # or simplified here by just creating a new node every time (not ideal for idempotency without an ID)
        # Using a generated ID based on attributes for basic idempotency
        msg_id = f"{sender_name}_{date}_{hash(text)}"
        
        # Single atomic query to handle nodes and relationships
        tx.run("""
            MERGE (p:Person {name: $name})
            ON CREATE SET p.normalized_name = toLower($name)
            
            MERGE (c:Conversation {id: $cid})
            
            MERGE (m:Message {id: $msg_id})
            SET m.text = $text,
                m.date = $date,
                m.polarity = $polarity,
                m.subjectivity = $subjectivity
            
            MERGE (p)-[:SENT]->(m)
            MERGE (m)-[:IN_CONVERSATION]->(c)
        """, msg_id=msg_id, text=text, date=date, 
             polarity=sentiment['polarity'], subjectivity=sentiment['subjectivity'],
             name=sender_name, cid=conversation_id)

def main():
    if not os.path.exists(MESSAGES_FILE):
        print(f"❌ File not found: {MESSAGES_FILE}")
        return

    print(f"Loading messages from {MESSAGES_FILE}...")
    with open(MESSAGES_FILE, 'r', encoding='utf-8') as f:
        messages = json.load(f)
    print(f"Loaded {len(messages)} messages.")

    kb = KnowledgeGraphBuilder(NEO4J_URI, NEO4J_AUTH)
    
    try:
        print("Connecting to Neo4j...")
        kb.create_constraints()
        
        print("Ingesting messages into Knowledge Graph...")
        # Ingest all messages
        kb.ingest_messages(messages) 
        
        print(f"✅ Knowledge Graph construction complete ({len(messages)} messages).")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        kb.close()

if __name__ == "__main__":
    main()
