"""
Voice Assistant Web UI - Flask Application
==========================================
Modern web interface for the e-commerce chatbot with voice capabilities.
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import logging
from pathlib import Path
import sys

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from chatbot import ProductChatbot

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-this'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    return render_template('index.html')


@app.route('/api/health')
def health_check():
    """Health check endpoint."""
    try:
        bot = get_chatbot()
        return jsonify({
            'status': 'healthy',
            'products_loaded': len(bot.df),
            'chatbot_ready': True
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'chatbot_ready': False
        }), 500


@app.route('/api/ask', methods=['POST'])
def ask_question():
    """Handle chatbot questions via REST API."""
    try:
        data = request.get_json()
        question = data.get('question', '')

        if not question:
            return jsonify({'error': 'Question is required'}), 400

        bot = get_chatbot()
        answer = bot.ask(question)

        return jsonify({
            'question': question,
            'answer': answer,
            'success': True
        })

    except Exception as e:
        logger.error(f"Error processing question: {e}")
        return jsonify({
            'error': str(e),
            'success': False
        }), 500


@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    logger.info("Client connected")
    emit('connection_response', {'status': 'connected'})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    logger.info("Client disconnected")


@socketio.on('ask_question')
def handle_question(data):
    """Handle real-time question via WebSocket."""
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

        # Send answer back
        emit('answer', {
            'question': question,
            'answer': answer,
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
    # Use port 5001 instead of 5000 (5000 is often used by AirPlay on macOS)
    PORT = 5001

    print("\n" + "="*60)
    print("🤖 Voice Assistant Web UI")
    print("="*60)
    print(f"\n🌐 Starting server at http://localhost:{PORT}")
    print(f"📱 Open your browser and navigate to: http://localhost:{PORT}")
    print("\n⚡ Features:")
    print("   • Voice input (speech-to-text)")
    print("   • Voice output (text-to-speech)")
    print("   • Real-time chat interface")
    print("   • Interactive visualizations")
    print("\n💡 Tip: If port is still busy, you can change PORT in app.py")
    print("\n" + "="*60 + "\n")

    # Note: allow_unsafe_werkzeug=True is for development only
    # For production, use a proper WSGI server like gunicorn
    socketio.run(app, host='0.0.0.0', port=PORT, debug=True, allow_unsafe_werkzeug=True)
