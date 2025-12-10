#!/bin/bash

# Voice Assistant Launcher
# ========================
# Quick launcher for the Voice Assistant Web UI

clear

echo "╔════════════════════════════════════════════════════════════╗"
echo "║                                                            ║"
echo "║          🤖 AI Voice Assistant Web UI                     ║"
echo "║          E-Commerce Inventory Chatbot                     ║"
echo "║                                                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run: python -m venv venv"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Check if OpenAI API key is set
if [ -z "$OPENAI_API_KEY" ]; then
    if [ -f ".env" ]; then
        echo "📝 Loading environment variables from .env..."
        export $(cat .env | grep -v '^#' | xargs)
    else
        echo "⚠️  Warning: OPENAI_API_KEY not found!"
        echo "Please create a .env file with your API key:"
        echo "  OPENAI_API_KEY=your-key-here"
        echo ""
        read -p "Press Enter to continue anyway, or Ctrl+C to exit..."
    fi
fi

# Check if required packages are installed
echo "📦 Checking dependencies..."
python -c "import flask" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Flask not installed. Installing dependencies..."
    pip install -r requirements.txt
fi

echo ""
echo "✅ All checks passed!"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "🌐 Starting Voice Assistant Web Server..."
echo ""
echo "📍 Local URL:  http://localhost:5000"
echo "📍 Network URL: http://$(ipconfig getifaddr en0 2>/dev/null || hostname):5000"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "⚡ Features:"
echo "   • 🎙️  Voice input (speech-to-text)"
echo "   • 🔊 Voice output (text-to-speech)"
echo "   • 💬 Real-time chat interface"
echo "   • 📊 Live statistics dashboard"
echo "   • 🎨 Beautiful modern UI"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "🔴 Press Ctrl+C to stop the server"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Wait a moment
sleep 2

# Start the server
python app.py
