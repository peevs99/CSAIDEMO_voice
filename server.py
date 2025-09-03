import os
import asyncio
import traceback
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import websockets

# --- Configuration ---
# Set your Gemini API Key as an environment variable
# export GEMINI_API_KEY="YOUR_API_KEY"
try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
except KeyError:
    print("FATAL: GEMINI_API_KEY environment variable not set.")
    exit(1)

# Define the model and system prompt
MODEL = "models/gemini-1.5-pro-latest"
SYSTEM_INSTRUCTION = """
You are a specialized AI voice assistant for DC Water, the water utility for Washington D.C.
Your name is the DC Water Voice Assistant. Your primary goal is to provide clear, accurate, and helpful information to customers.
Maintain a professional, polite, and patient persona. Your tone should be helpful and reassuring.
Only answer questions related to DC Water services (billing, water quality, leaks, etc.).
Politely decline off-topic questions.
Never invent information, especially regarding safety or billing. If you don't know, say so and suggest they visit the official website.
Do not ask for or process sensitive personal information like credit card numbers.
Keep your responses concise and conversational for a voice interface.
"""

# --- WebSocket Server Logic ---
async def audio_chat_handler(websocket, path):
    """Handles the WebSocket connection and the Gemini LiveConnect session."""
    print(f"Client connected from {websocket.remote_address}")

    try:
        # 1. Start a Gemini LiveConnect session
        async with genai.live.connect(
            model=MODEL,
            system_instruction=SYSTEM_INSTRUCTION,
            # Adjust safety settings if needed
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            },
            # Use a high-quality voice
            speech_config=genai.SpeechConfig(voice="gemini-1.5-pro-voice-en-us-1")
        ) as session:
            
            print(f"Gemini LiveConnect session started for {websocket.remote_address}")

            # 2. Define two concurrent tasks:
            #    - browser_to_gemini: forwards audio from the browser to the Gemini API.
            #    - gemini_to_browser: forwards responses from Gemini back to the browser.
            
            async def browser_to_gemini():
                """Receives audio from the WebSocket and sends it to Gemini."""
                async for message in websocket:
                    await session.send_audio(message)

            async def gemini_to_browser():
                """Receives responses from Gemini and sends them to the WebSocket."""
                async for response in session:
                    if response.audio:
                        await websocket.send(response.audio)
                    if response.text:
                        print(f"Transcript for {websocket.remote_address}: {response.text}")
                        # Optional: Send transcript back as text message
                        # await websocket.send(f"TRANSCRIPT: {response.text}")

            # 3. Run both tasks in parallel
            await asyncio.gather(browser_to_gemini(), gemini_to_browser())

    except websockets.exceptions.ConnectionClosed as e:
        print(f"Connection closed for {websocket.remote_address}: {e.reason} (Code: {e.code})")
    except Exception as e:
        print(f"An error occurred for {websocket.remote_address}: {e}")
        traceback.print_exc()
    finally:
        print(f"Client disconnected from {websocket.remote_address}")


async def main():
    """Starts the WebSocket server."""
    # Render and other providers set the PORT environment variable.
    # Default to 8000 for local development.
    port = int(os.environ.get("PORT", 8000))
    # Bind to 0.0.0.0 to be accessible from outside the container.
    host = "0.0.0.0"
    
    print(f"Starting WebSocket server on {host}:{port}...")
    async with websockets.serve(audio_chat_handler, host, port):
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())
