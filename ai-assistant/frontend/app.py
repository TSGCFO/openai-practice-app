"""
AI Assistant Frontend Service
Flask application with WebSocket support via Flask-SocketIO
"""

from flask import Flask, render_template, request, session, jsonify, redirect, url_for
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_cors import CORS
import httpx
import asyncio
import json
import logging
from datetime import datetime
import os
from dotenv import load_dotenv
import secrets

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO'))
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', secrets.token_hex(32))
app.config['SESSION_TYPE'] = 'filesystem'

# Configure CORS
CORS(app, origins=os.getenv('ALLOWED_ORIGINS', 'http://localhost:5000').split(','))

# Configure SocketIO
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Configuration
ORCHESTRATOR_URL = os.getenv('ORCHESTRATOR_URL', 'http://localhost:8000')

# Active sessions tracking
active_sessions = {}

@app.route('/')
def index():
    """Render main chat interface"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('index.html', user=session.get('user'))

@app.route('/login')
def login():
    """Render login page"""
    return render_template('login.html')

@app.route('/chat')
def chat():
    """Render chat interface"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('chat.html', user=session.get('user'))

@app.route('/api/conversations', methods=['GET'])
def get_conversations():
    """Get user conversations"""
    if 'user_token' not in session:
        return jsonify({"error": "Not authenticated"}), 401
    
    # Fetch from orchestrator
    try:
        with httpx.Client() as client:
            response = client.get(
                f"{ORCHESTRATOR_URL}/api/v1/conversations",
                headers={"Authorization": f"Bearer {session['user_token']}"}
            )
            return jsonify(response.json()), response.status_code
    except Exception as e:
        logger.error(f"Error fetching conversations: {e}")
        return jsonify({"error": "Failed to fetch conversations"}), 500

@app.route('/api/conversations', methods=['POST'])
def create_conversation():
    """Create a new conversation"""
    if 'user_token' not in session:
        return jsonify({"error": "Not authenticated"}), 401
    
    try:
        with httpx.Client() as client:
            response = client.post(
                f"{ORCHESTRATOR_URL}/api/v1/conversations",
                headers={"Authorization": f"Bearer {session['user_token']}"},
                json=request.json
            )
            return jsonify(response.json()), response.status_code
    except Exception as e:
        logger.error(f"Error creating conversation: {e}")
        return jsonify({"error": "Failed to create conversation"}), 500

@app.route('/api/auth/login', methods=['POST'])
def api_login():
    """Handle login via API"""
    # Mock authentication for development
    # In production, this would integrate with OAuth2/OIDC
    data = request.json
    email = data.get('email')
    
    if not email:
        return jsonify({"error": "Email required"}), 400
    
    # Create mock session
    session['user_id'] = email
    session['user'] = {
        'id': email,
        'email': email,
        'name': email.split('@')[0]
    }
    session['user_token'] = secrets.token_urlsafe(32)  # Mock token
    
    return jsonify({
        "success": True,
        "user": session['user']
    })

@app.route('/api/auth/logout', methods=['POST'])
def api_logout():
    """Handle logout"""
    session.clear()
    return jsonify({"success": True})

# WebSocket event handlers
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info(f"Client connected: {request.sid}")
    emit('connected', {'data': 'Connected to server'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info(f"Client disconnected: {request.sid}")
    if request.sid in active_sessions:
        del active_sessions[request.sid]

@socketio.on('join_conversation')
def handle_join_conversation(data):
    """Join conversation room"""
    conversation_id = data.get('conversation_id')
    if not conversation_id:
        emit('error', {'error': 'conversation_id required'})
        return
    
    join_room(conversation_id)
    active_sessions[request.sid] = {
        'conversation_id': conversation_id,
        'user_id': session.get('user_id')
    }
    emit('joined', {'conversation_id': conversation_id}, room=conversation_id)
    logger.info(f"User {session.get('user_id')} joined conversation {conversation_id}")

@socketio.on('leave_conversation')
def handle_leave_conversation(data):
    """Leave conversation room"""
    conversation_id = data.get('conversation_id')
    if conversation_id:
        leave_room(conversation_id)
        emit('left', {'conversation_id': conversation_id})
        logger.info(f"User {session.get('user_id')} left conversation {conversation_id}")

@socketio.on('send_message')
def handle_message(data):
    """Handle incoming message and stream response"""
    conversation_id = data.get('conversation_id')
    message = data.get('message')
    
    if not all([conversation_id, message]):
        emit('error', {'error': 'conversation_id and message required'})
        return
    
    if 'user_token' not in session:
        emit('error', {'error': 'Not authenticated'})
        return
    
    # Start async task to process message
    socketio.start_background_task(
        stream_response,
        conversation_id,
        message,
        session['user_token'],
        request.sid
    )

def stream_response(conversation_id, message, user_token, client_sid):
    """Stream response from orchestrator"""
    try:
        # Send message to orchestrator and stream response
        with httpx.Client() as client:
            with client.stream(
                'POST',
                f"{ORCHESTRATOR_URL}/api/v1/conversations/{conversation_id}/messages",
                json={"content": message},
                headers={"Authorization": f"Bearer {user_token}"},
                timeout=60.0
            ) as response:
                for line in response.iter_lines():
                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                            
                            # Emit to client based on event type
                            if data['type'] == 'token':
                                socketio.emit('token', data, room=conversation_id)
                            elif data['type'] == 'tool_request':
                                socketio.emit('tool_request', data, room=conversation_id)
                            elif data['type'] == 'tool_result':
                                socketio.emit('tool_result', data, room=conversation_id)
                            elif data['type'] == 'complete':
                                socketio.emit('complete', data, room=conversation_id)
                            elif data['type'] == 'error':
                                socketio.emit('error', data, room=conversation_id)
                        except json.JSONDecodeError:
                            continue
                            
    except Exception as e:
        logger.error(f"Error streaming response: {e}")
        socketio.emit('error', {'error': str(e)}, room=conversation_id)

@socketio.on('cancel_stream')
def handle_cancel_stream(data):
    """Handle stream cancellation"""
    conversation_id = data.get('conversation_id')
    # In production, this would signal the orchestrator to stop streaming
    emit('stream_cancelled', {'conversation_id': conversation_id})

# Health check endpoint
@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "frontend",
        "timestamp": datetime.utcnow().isoformat()
    })

# Error handlers
@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    if request.path.startswith('/api/'):
        return jsonify({"error": "Not found"}), 404
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors"""
    logger.error(f"Server error: {e}")
    if request.path.startswith('/api/'):
        return jsonify({"error": "Internal server error"}), 500
    return render_template('500.html'), 500

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)