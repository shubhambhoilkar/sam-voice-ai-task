#working great, can be use for Demo 
#developement for centra hosiptal

import os
import io
import asyncio
import openai
import uvicorn
import base64
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List
from pymongo import MongoClient
from gtts import gTTS
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
    Microphone,
)

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
deepgram_key = os.getenv("DEEPGRAM_API_KEY")

if not openai.api_key or not deepgram_key:
    raise EnvironmentError("Missing OPENAI_API_KEY or DEEPGRAM_API_KEY in .env file")

#--FAISS + MongoDB setup
client = MongoClient("mongodb://localhost:27017/")
db = client["Central_Hospital"]
collection = db["central_hospital_ai_chunks"]
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load and chunk prompt file
def load_and_chunk(file_path: str, chunk_size: int = 500) -> List[str]:
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# Store chunks in MongoDB
def store_prompt_in_mongo(file_path: str):
    if collection.count_documents({}) > 0:
        return  # Already stored
    print("📝 Storing prompt chunks into MongoDB...")
    chunks = load_and_chunk(file_path)
    for i, chunk in enumerate(chunks):
        embedding = model.encode(chunk).tolist()
        collection.insert_one({
            "chunk_id": i,
            "text": chunk,
            "embedding": embedding
        })

# Load from Mongo and build FAISS
def load_chunks_from_mongo():
    docs = list(collection.find().sort("chunk_id", 1))
    texts = [doc["text"] for doc in docs]
    embeddings = np.array([doc["embedding"] for doc in docs], dtype='float32')
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return texts, index

# Load prompt into MongoDB and FAISS on boot
store_prompt_in_mongo("ONLY_CENTRAL_HOSPITAL_KB.TXT")
chunk_texts, faiss_index = load_chunks_from_mongo()

#------------------
#chat history

#Fast api code 
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Use ["https://yourdomain.com"] in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ai_is_speaking = False
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global ai_is_speaking
    await websocket.accept()
    print("✅ Websocket connection open")
    try:
        while True:
            data = await websocket.receive_text()
            print(f"Received from client: {data}")

            if ai_is_speaking:
                print("⏸️ Pausing AI speech due to user interruption.")
                await websocket.send_json({"pause": True})
                ai_is_speaking = False

            ai_response = await get_ai_response(data)
            ai_is_speaking = True  # AI is starting to speak now

            #Convert AI response to Speech
            audio_base64 = await text_to_speech(ai_response)

            #send text and audio in one message
            await websocket.send_json({"text":ai_response,
                                       "audio": audio_base64 #base64 audio Stream
                                       })  
            ai_is_speaking = False
    except Exception as e:
        print(f"WebSocket connection closed: {e}")
        
    finally:
        await websocket.close()
        print("🔌 WebSocket connection closed.")
        try: 
            await websocket.close()
        except RuntimeError:
            pass


#Core working Chatbot code from Customer response to the Ai response
class TranscriptCollector:
    def __init__(self):
        self.reset()

    def reset(self):
        self.transcript_parts = []

    def add_part(self, part):
        self.transcript_parts.append(part)

    def get_full_transcript(self):
        return ' '.join(self.transcript_parts)

transcript_collector = TranscriptCollector()
silence_counter = 0
silence_timeout = 18 #seconds 

async def get_transcript(websocket):
    global silence_counter
    try:
        api_key = os.getenv("DEEPGRAM_API_KEY")
        if not api_key:
            print("Missing Deepgram API Key")
            return
        
        config = DeepgramClientOptions(options={"keepalive": "true"})
        deepgram: DeepgramClient = DeepgramClient(api_key, config)
        dg_connection = deepgram.listen.asyncwebsocket.v("1")

        #start silence monitor
        last_activity = asyncio.get_event_loop().time()
        stop_listening = False

        async def silence_monitor():
            nonlocal last_activity, stop_listening
            global silence_counter
            while not stop_listening:
                await asyncio.sleep(1)
                if asyncio.get_event_loop().time() - last_activity > silence_timeout:
                    silence_counter += 1
                    if silence_counter ==1:
                        print("AI: Are you speaking? I haven't hear from you in a while.")
                    elif silence_counter == 2:
                        print("AI: Ending session due to inactivity.")
                        stop_listening = True
                        microphone.finish()
                        await dg_connection.finish()
                        break
                    last_activity = asyncio.get_event_loop().time()

        finalization_task = None
        async def on_message(self, result, **kwargs):
            nonlocal last_activity, stop_listening, finalization_task
            global silence_counter

            if stop_listening:
                return
            
            sentence = result.channel.alternatives[0].transcript

            if result.speech_final:
                transcript_collector.add_part(sentence)
                full_sentence = transcript_collector.get_full_transcript().strip()

                silence_counter = 0
                last_activity = asyncio.get_event_loop().time()

                if finalization_task:
                    finalization_task.cancel()
                
                finalization_task = asyncio.create_task(process_after_delay(full_sentence))

        async def process_after_delay(full_sentence):
            try:
                await asyncio.sleep(1.0)
                if len(full_sentence) < 1:
                    return
                print(f"speaker: {full_sentence}")

                ai_response = await get_ai_response(full_sentence)
                print(f"AI: {ai_response}")
                await websocket.send_text(ai_response)
                transcript_collector.reset()
            
            except asyncio.CancelledError:
                pass

        async def on_error(self, error, **kwargs):
            print(f"\n\n{error}\n\n")

        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)
        dg_connection.on(LiveTranscriptionEvents.Error, on_error)

        options = LiveOptions(
            model="nova-2",
            punctuate=True,
            language="en-US",
            encoding="linear16",
            channels=1,
            sample_rate=16000,
            endpointing=True
        )

        await dg_connection.start(options)

        # You can specify device_index if needed
        microphone = Microphone(dg_connection.send)
        microphone.start()
        asyncio.create_task(silence_monitor())

        while True:
            if not microphone.is_active():
                break
            await asyncio.sleep(1)

        microphone.finish()
        await dg_connection.finish()

        print("Finished")

    except Exception as e:
        print(f"Could not open socket: {e}")
        return

async def get_ai_response(prompt):
    try:
        query_vec = model.encode([prompt])
        D, I = faiss_index.search(np.array(query_vec), k=5)
        print("D:", D)
        Distance =  D[0][0]
        print("Distance: ", Distance)
        confidence =1-Distance/2
        print("confidence: ", confidence)
        if confidence >= 0.4:

            relevant_chunks = [chunk_texts[i] for i in I[0]]
            context = "\n---\n".join(relevant_chunks)
            print("context: ", context)
            full_prompt = f"{context}\n\nUser: {prompt}"
        else:
            full_prompt = f"User: {prompt}"
        persona = """PERSONA: 
                    Every time you respond to user input, you must adopt the following persona: 
                    Professional, calm, and efficient. Lee is approachable and well-spoken, ensuring every 
                    caller gets the right information quickly. He maintains a reassuring tone, especially in 
                    emergencies, while staying highly organized. He is direct, clear, and polite. 
                    Voice & Persona.
                    Lee , you provide information related to Central Hosiptal queries only.
                      If user is brief/terse: keep responses concise of 2-3 sentences. 
                      If user is curious or talkative: respond with warmth, subtle humor, or 
                    relatable lines. 
                      If user sounds frustrated or unsure: lead with empathy, reassure, and calmly 
                    offer a solution. 
                    
                    Speech Characteristics 
                    •  Speak clearly with natural contractions: “I’ll” instead of “I will.” 
                    •  Use ellipses (...) for audible pauses. 
                    •  Incorporate brief affirmations: "Got it," "Okay," "Alright," "Perfect." 
                    •  Use natural confirmations: “Yes,” “Oh yes,” “Right,” “That's there.” 
                    •  Include filler words and subtle disfluencies: “uhm,” “actually,” “you know,” “ah, 
                    wait...” 
                    •  Use light chuckles and casual cues: "Haha, that’s true," "Ah, I see..." 
                    •  Repeat key details naturally: “So, that’s a follow-up with Dr. Chen... next 
                    Thursday?” 

                    •  Sound human when uncertain: “Hmm... let me double-check that for you...” or 
                    “Ah, just a moment... checking the schedule…” 
                    Natural Conversation Cues  
                    These help maintain engagement and realism: 
                    •  “Oh okay... what date were you thinking?” 
                    •  “Let me just check that... please hold on a second.” 
                    •  “Oh! This time isn't available anymore. Want me to check another slot?” 
                    •  “Perfect, got it!” 
                    •  “Are you free that morning? No? Okay, when would be a better time?” 
                    •  “Want me to call you back instead?” 
                    •  “Sure thing, I’ll wait... take your time.” 
                    •  “Ah yes, found it!”"""
        
        response = await openai.ChatCompletion.acreate(
            model= "gpt-3.5-turbo",
            messages=[
                {"role":"system",
                 "content": persona},
                {"role":"user",
                 "content":full_prompt}
            ]
        )
        ai_text = response["choices"][0]["message"]["content"]
        print(f"AI: {ai_text}")
        return ai_text

    except Exception as e:
        print(f"Error from OpenAi: {e}")
        return "Opps, sorry i couldn't process text."

async def text_to_speech(text):
    try:
        tts = gTTS(text)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        audio_base64 = base64.b64encode(audio_buffer.read()).decode('utf-8')
        print(f"[TTS] Audio length: {len(audio_base64)}")
        return audio_base64
    except Exception as e:
        print(f"TTS Error: {e}")
        return ""

if __name__ == "__main__":
    uvicorn.run("main:app", host = "0.0.0.0", port = 9900, reload = True)