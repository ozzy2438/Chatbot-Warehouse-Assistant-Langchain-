/**
 * AI Voice Assistant - Main JavaScript
 * =====================================
 * Handles WebSocket communication, speech recognition, and UI interactions
 */

// ============================================
// Global State
// ============================================

let socket = null;
let recognition = null;
let synthesis = window.speechSynthesis;
let isListening = false;
let conversationStarted = false;

// Settings
let autoSpeak = true;
let continuousListening = false;

// Voice visualizer
let audioContext = null;
let analyser = null;
let microphone = null;
let javascriptNode = null;
let canvas = null;
let canvasContext = null;

// ============================================
// Initialization
// ============================================

document.addEventListener('DOMContentLoaded', () => {
    initializeSocketIO();
    initializeSpeechRecognition();
    initializeEventListeners();
    loadStats();
    setupVisualizer();
});

// ============================================
// Socket.IO
// ============================================

function initializeSocketIO() {
    socket = io();

    socket.on('connect', () => {
        console.log('✓ Connected to server');
        showToast('Connected to AI Assistant', 'success');
    });

    socket.on('disconnect', () => {
        console.log('✗ Disconnected from server');
        showToast('Disconnected from server', 'error');
    });

    socket.on('thinking', () => {
        showThinking(true);
    });

    socket.on('answer', (data) => {
        showThinking(false);
        addMessage('assistant', data.answer);

        // Speak the answer if auto-speak is enabled
        if (autoSpeak) {
            speakText(data.answer);
        }

        // Continue listening if continuous mode is on
        if (continuousListening && conversationStarted) {
            setTimeout(() => {
                startListening();
            }, 1000);
        }
    });

    socket.on('error', (data) => {
        showThinking(false);
        showToast(data.message, 'error');
    });
}

// ============================================
// Speech Recognition
// ============================================

function initializeSpeechRecognition() {
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
        console.warn('Speech recognition not supported');
        document.getElementById('voice-btn').disabled = true;
        showToast('Speech recognition not supported in this browser', 'error');
        return;
    }

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    recognition = new SpeechRecognition();

    recognition.lang = 'tr-TR'; // Turkish language
    recognition.continuous = false;
    recognition.interimResults = false;

    recognition.onstart = () => {
        isListening = true;
        document.getElementById('voice-btn').classList.add('active');
        document.getElementById('voice-visualizer').style.display = 'flex';
        startVisualization();
    };

    recognition.onend = () => {
        isListening = false;
        document.getElementById('voice-btn').classList.remove('active');
        document.getElementById('voice-visualizer').style.display = 'none';
        stopVisualization();
    };

    recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        console.log('Recognized:', transcript);

        document.getElementById('question-input').value = transcript;
        sendMessage(transcript);
    };

    recognition.onerror = (event) => {
        console.error('Speech recognition error:', event.error);
        showToast('Speech recognition error: ' + event.error, 'error');
        isListening = false;
        document.getElementById('voice-btn').classList.remove('active');
        document.getElementById('voice-visualizer').style.display = 'none';
        stopVisualization();
    };
}

function startListening() {
    if (!recognition) {
        showToast('Speech recognition not available', 'error');
        return;
    }

    if (isListening) {
        recognition.stop();
    } else {
        try {
            recognition.start();
        } catch (error) {
            console.error('Error starting recognition:', error);
        }
    }
}

// ============================================
// Text-to-Speech
// ============================================

function speakText(text) {
    // Cancel any ongoing speech
    synthesis.cancel();

    // Remove markdown formatting for better speech
    const cleanText = text
        .replace(/\*\*/g, '')
        .replace(/\*/g, '')
        .replace(/#{1,6}\s/g, '')
        .replace(/\[([^\]]+)\]\([^\)]+\)/g, '$1')
        .replace(/`([^`]+)`/g, '$1')
        .replace(/---/g, '')
        .replace(/\|/g, '');

    const utterance = new SpeechSynthesisUtterance(cleanText);

    // Try to find a Turkish voice, otherwise use default
    const voices = synthesis.getVoices();
    const turkishVoice = voices.find(voice => voice.lang.startsWith('tr'));
    if (turkishVoice) {
        utterance.voice = turkishVoice;
    }

    utterance.rate = 1.0;
    utterance.pitch = 1.0;
    utterance.volume = 1.0;

    synthesis.speak(utterance);
}

// ============================================
// Voice Visualizer
// ============================================

function setupVisualizer() {
    canvas = document.getElementById('visualizer-canvas');
    if (canvas) {
        canvasContext = canvas.getContext('2d');
        canvas.width = canvas.offsetWidth;
        canvas.height = canvas.offsetHeight;
    }
}

async function startVisualization() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        analyser = audioContext.createAnalyser();
        microphone = audioContext.createMediaStreamSource(stream);
        javascriptNode = audioContext.createScriptProcessor(2048, 1, 1);

        analyser.smoothingTimeConstant = 0.8;
        analyser.fftSize = 1024;

        microphone.connect(analyser);
        analyser.connect(javascriptNode);
        javascriptNode.connect(audioContext.destination);

        javascriptNode.onaudioprocess = function() {
            const array = new Uint8Array(analyser.frequencyBinCount);
            analyser.getByteFrequencyData(array);
            drawVisualizer(array);
        };

    } catch (error) {
        console.error('Error accessing microphone:', error);
    }
}

function stopVisualization() {
    if (javascriptNode) {
        javascriptNode.disconnect();
        javascriptNode = null;
    }
    if (microphone) {
        microphone.disconnect();
        microphone = null;
    }
    if (analyser) {
        analyser.disconnect();
        analyser = null;
    }
    if (audioContext) {
        audioContext.close();
        audioContext = null;
    }
    if (canvasContext && canvas) {
        canvasContext.clearRect(0, 0, canvas.width, canvas.height);
    }
}

function drawVisualizer(array) {
    if (!canvasContext || !canvas) return;

    const width = canvas.width;
    const height = canvas.height;
    const barWidth = (width / array.length) * 2.5;
    let barHeight;
    let x = 0;

    // Clear canvas
    canvasContext.fillStyle = 'rgb(15, 23, 42)';
    canvasContext.fillRect(0, 0, width, height);

    // Draw bars
    for (let i = 0; i < array.length; i++) {
        barHeight = (array[i] / 255) * height * 0.8;

        // Gradient colors
        const gradient = canvasContext.createLinearGradient(0, height - barHeight, 0, height);
        gradient.addColorStop(0, 'rgb(99, 102, 241)');
        gradient.addColorStop(1, 'rgb(16, 185, 129)');

        canvasContext.fillStyle = gradient;
        canvasContext.fillRect(x, height - barHeight, barWidth, barHeight);

        x += barWidth + 1;
    }
}

// ============================================
// Event Listeners
// ============================================

function initializeEventListeners() {
    // Voice button
    document.getElementById('voice-btn').addEventListener('click', startListening);

    // Send button
    document.getElementById('send-btn').addEventListener('click', () => {
        const input = document.getElementById('question-input');
        if (input.value.trim()) {
            sendMessage(input.value);
            input.value = '';
        }
    });

    // Enter key to send
    document.getElementById('question-input').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            const input = e.target;
            if (input.value.trim()) {
                sendMessage(input.value);
                input.value = '';
            }
        }
    });

    // Example queries
    document.querySelectorAll('.example-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const query = btn.getAttribute('data-query');
            document.getElementById('question-input').value = query;
            sendMessage(query);
        });
    });

    // Settings toggles
    document.getElementById('auto-speak').addEventListener('change', (e) => {
        autoSpeak = e.target.checked;
        showToast(`Auto-speak ${autoSpeak ? 'enabled' : 'disabled'}`, 'info');
    });

    document.getElementById('continuous-listening').addEventListener('change', (e) => {
        continuousListening = e.target.checked;
        showToast(`Continuous listening ${continuousListening ? 'enabled' : 'disabled'}`, 'info');
    });
}

// ============================================
// Message Handling
// ============================================

function startConversation() {
    conversationStarted = true;
    document.getElementById('welcome-message').style.display = 'none';
    document.getElementById('messages').style.display = 'flex';
    showToast('Conversation started! Ask me anything.', 'success');
}

function sendMessage(question) {
    if (!conversationStarted) {
        startConversation();
    }

    // Add user message to chat
    addMessage('user', question);

    // Send to server via WebSocket
    socket.emit('ask_question', { question: question });

    // Clear input
    document.getElementById('question-input').value = '';
}

function addMessage(role, content) {
    const messagesContainer = document.getElementById('messages');

    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.innerHTML = role === 'user' ? '<i class="fas fa-user"></i>' : '<i class="fas fa-robot"></i>';

    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';

    // Parse markdown
    if (role === 'assistant') {
        messageContent.innerHTML = marked.parse(content);
    } else {
        messageContent.textContent = content;
    }

    const messageTime = document.createElement('div');
    messageTime.className = 'message-time';
    messageTime.textContent = new Date().toLocaleTimeString('tr-TR', {
        hour: '2-digit',
        minute: '2-digit'
    });

    messageDiv.appendChild(avatar);
    const contentWrapper = document.createElement('div');
    contentWrapper.appendChild(messageContent);
    contentWrapper.appendChild(messageTime);
    messageDiv.appendChild(contentWrapper);

    messagesContainer.appendChild(messageDiv);

    // Scroll to bottom
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function showThinking(show) {
    const thinkingIndicator = document.getElementById('thinking-indicator');
    thinkingIndicator.style.display = show ? 'flex' : 'none';

    if (show) {
        const messagesContainer = document.getElementById('messages');
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
}

// ============================================
// Stats Loading
// ============================================

async function loadStats() {
    try {
        const response = await fetch('/api/stats');
        const stats = await response.json();

        document.getElementById('total-products').textContent = stats.total_products.toLocaleString();
        document.getElementById('total-stock').textContent = stats.total_stock.toLocaleString();
        document.getElementById('avg-rating').textContent = stats.avg_rating.toFixed(1);
        document.getElementById('low-stock-count').textContent = stats.low_stock_count;
        document.getElementById('avg-price').textContent = `$${stats.avg_price.toFixed(2)}`;

    } catch (error) {
        console.error('Error loading stats:', error);
    }
}

// ============================================
// Toast Notifications
// ============================================

function showToast(message, type = 'info') {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.className = `toast ${type} show`;

    setTimeout(() => {
        toast.classList.remove('show');
    }, 3000);
}

// ============================================
// Voice synthesis initialization
// ============================================

// Load voices when they become available
if (synthesis.onvoiceschanged !== undefined) {
    synthesis.onvoiceschanged = () => {
        console.log('Voices loaded:', synthesis.getVoices().length);
    };
}
