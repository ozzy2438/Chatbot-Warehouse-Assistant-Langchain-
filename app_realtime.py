"""
OpenAI Realtime API Voice Assistant - Professional Version
===========================================================
Uses GPT-4o-audio for natural voice conversations with proper VAD.
"""

import re
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import logging
from pathlib import Path
import sys
import os
from openai import OpenAI
import base64
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Use v2 (Function Calling) or fallback to v1
try:
    from chatbot_v2 import ProductChatbot
    print("Using chatbot_v2 (Function Calling)")
except ImportError:
    from chatbot import ProductChatbot
    print("Using chatbot v1 (Pattern Matching)")

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", max_http_buffer_size=10**8)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize chatbot (shared across all connections)
chatbot = None

def get_chatbot():
    """Lazy load chatbot instance."""
    global chatbot
    if chatbot is None:
        logger.info("Initializing ProductChatbot...")
        chatbot = ProductChatbot()
        logger.info("Chatbot initialized successfully")
    return chatbot


@app.route('/')
def index():
    """Render the main UI."""
    return render_template('index_realtime.html')


@app.route('/api/health')
def health_check():
    """Health check endpoint."""
    try:
        bot = get_chatbot()
        return jsonify({
            'status': 'healthy',
            'products_loaded': len(bot.df),
            'chatbot_ready': True,
            'realtime_api': True
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'chatbot_ready': False
        }), 500


@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    logger.info("✅ Client connected via SocketIO")
    emit('connection_response', {'status': 'connected'})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    logger.info("Client disconnected")


@socketio.on('audio_chunk')
def handle_audio_chunk(data):
    """Handle incoming audio chunk from client."""
    logger.info("🎤 Received audio_chunk")
    try:
        # Audio data comes as base64
        audio_data = data.get('audio')
        is_final = data.get('is_final', False)

        # Store audio chunks in session (you'll need to implement session management)
        # For now, we'll process when final
        if is_final:
            emit('processing', {'status': 'transcribing'})

            # Transcribe using Whisper
            audio_bytes = base64.b64decode(audio_data)

            # DEBUG: Check audio size
            logger.info(f"📊 Audio size: {len(audio_bytes)} bytes")

            # Save temporarily
            temp_audio = Path("/tmp/temp_audio.webm")
            with open(temp_audio, 'wb') as f:
                f.write(audio_bytes)

            logger.info(f"💾 Saved audio file: {temp_audio.stat().st_size} bytes")

            # Transcribe - USE ENGLISH for better accuracy
            with open(temp_audio, 'rb') as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="en"  # Changed from "tr" to "en"
                )

            question = transcript.text
            logger.info(f"Transcribed: {question}")

            emit('transcription', {'text': question})

            # Get chatbot response
            emit('processing', {'status': 'thinking'})
            bot = get_chatbot()
            answer = bot.ask(question)

            # Generate speech using TTS
            emit('processing', {'status': 'generating_speech'})
            
            # Clean text for better TTS - remove ALL markdown and symbols
            clean_answer = answer[:600]  # Limit length
            # Remove markdown formatting
            clean_answer = re.sub(r'\*\*([^*]+)\*\*', r'\1', clean_answer)  # **bold** -> bold
            clean_answer = re.sub(r'\*([^*]+)\*', r'\1', clean_answer)  # *italic* -> italic
            clean_answer = clean_answer.replace('**', '').replace('*', '')  # Any remaining
            clean_answer = clean_answer.replace('#', '').replace('`', '')
            # Remove emojis and special chars that sound bad
            clean_answer = re.sub(r'[📋📍✅💡🤖⭐💰📦🎙️]', '', clean_answer)
            # Number/currency fixes
            clean_answer = clean_answer.replace('$', ' dollars ')
            clean_answer = clean_answer.replace('N/A', 'not available')
            # Clean up extra spaces
            clean_answer = ' '.join(clean_answer.split())
            
            response = client.audio.speech.create(
                model="tts-1-hd",            # High quality
                voice="nova",                # Clear, professional voice
                input=clean_answer,
                response_format="mp3",       # reliable for browsers
                speed=0.92                   # Slightly slower for clarity
            )

            # Convert to base64
            audio_content = response.content
            audio_base64 = base64.b64encode(audio_content).decode('utf-8')

            # Send response
            emit('answer', {
                'question': question,
                'answer': answer,
                'audio': audio_base64,
                'timestamp': __import__('datetime').datetime.now().isoformat()
            })

            # Cleanup
            temp_audio.unlink(missing_ok=True)

    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        import traceback
        traceback.print_exc()
        emit('error', {'message': str(e)})


@socketio.on('ask_question')
def handle_question(data):
    """Handle text-based question via WebSocket."""
    logger.info(f"📝 Received ask_question: {data.get('question', '')[:50]}...")
    try:
        question = data.get('question', '')

        if not question:
            emit('error', {'message': 'Question is required'})
            return

        logger.info(f"Processing question: {question}")

        # Send thinking indicator
        emit('thinking', {'status': 'processing'})

        # Get answer from chatbot
        bot = get_chatbot()
        answer = bot.ask(question)

        # Generate speech
        emit('processing', {'status': 'generating_speech'})
        
        # Clean text for better TTS
        clean_answer = answer[:600]
        # Remove markdown
        clean_answer = re.sub(r'\*\*([^*]+)\*\*', r'\1', clean_answer)
        clean_answer = re.sub(r'\*([^*]+)\*', r'\1', clean_answer)
        clean_answer = clean_answer.replace('**', '').replace('*', '')
        clean_answer = clean_answer.replace('#', '').replace('`', '')
        # Remove emojis
        clean_answer = re.sub(r'[📋📍✅💡🤖⭐💰📦🎙️]', '', clean_answer)
        clean_answer = clean_answer.replace('$', ' dollars ')
        clean_answer = clean_answer.replace('N/A', 'not available')
        clean_answer = ' '.join(clean_answer.split())
        
        response = client.audio.speech.create(
            model="tts-1-hd",
            voice="nova",
            input=clean_answer,
            response_format="mp3",
            speed=0.92
        )

        audio_content = response.content
        audio_base64 = base64.b64encode(audio_content).decode('utf-8')

        # Send answer back
        emit('answer', {
            'question': question,
            'answer': answer,
            'audio': audio_base64,
            'timestamp': __import__('datetime').datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error processing question: {e}")
        emit('error', {'message': str(e)})


@app.route('/api/stats')
def get_stats():
    """Get database statistics."""
    try:
        bot = get_chatbot()

        # Category stats
        category_stats = bot.df.groupby('category').agg({
            'asin': 'count',
            'rating': 'mean',
            'price': 'mean',
            'stock_quantity': 'sum'
        }).round(2).to_dict('index')

        # Warehouse stats
        warehouse_stats = bot.df.groupby('warehouse_location').agg({
            'asin': 'count',
            'stock_quantity': 'sum'
        }).to_dict('index')

        # Low stock count
        low_stock_count = len(bot.df[bot.df['stock_quantity'] < bot.df['minimum_stock_threshold']])

        return jsonify({
            'total_products': len(bot.df),
            'total_stock': int(bot.df['stock_quantity'].sum()),
            'categories': category_stats,
            'warehouses': warehouse_stats,
            'low_stock_count': low_stock_count,
            'avg_rating': round(bot.df['rating'].mean(), 2),
            'avg_price': round(bot.df['price'].mean(), 2)
        })

    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    PORT = 5002  # Different port for realtime version

    print("\n" + "="*60)
    print("🎙️ OpenAI Realtime Voice Assistant")
    print("="*60)
    print(f"\n🌐 Starting server at http://localhost:{PORT}")
    print(f"📱 Open your browser and navigate to: http://localhost:{PORT}")
    print("\n⚡ Features:")
    print("   • OpenAI Whisper (accurate transcription)")
    print("   • OpenAI TTS HD (natural voice)")
    print("   • Voice Activity Detection")
    print("   • Professional audio quality")
    print("\n💡 This version uses OpenAI APIs for better quality")
    print("\n" + "="*60 + "\n")

    socketio.run(app, host='0.0.0.0', port=PORT, debug=True, allow_unsafe_werkzeug=True)
