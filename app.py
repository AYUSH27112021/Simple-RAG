from flask import Flask, request, jsonify
import mysql.connector
from datetime import datetime
from GraphRag import GraphRag

app = Flask(__name__)

# MySQL Configuration
DB_CONFIG = {
    "host": "localhost",
    "user": "Enter-your-username",
    "password": "Enter-your-password",
    "database": "chatbot"
}

# Initialize GraphRag
graph_rag = GraphRag()

# Ensure tables exist
def init_db():
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INT AUTO_INCREMENT PRIMARY KEY,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            role ENUM('user', 'system') NOT NULL,
            content TEXT NOT NULL
        )
    """)
    conn.commit()
    cursor.close()
    conn.close()

init_db()  # Call on startup

# Store message in DB
def save_message(role, content):
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO chat_history (role, content) VALUES (%s, %s)", (role, content))
    conn.commit()
    cursor.close()
    conn.close()

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_query = data.get("query", "").strip()
    
    if not user_query:
        return jsonify({"error": "Query cannot be empty"}), 400

    # Call GraphRag for chatbot response
    inputs = {"question": user_query, "max_retries": 3}
    response = graph_rag.invoke_graph(inputs)
    # Save both user query and response
    save_message("user", user_query)
    save_message("system", response)

    return jsonify({"answer": response})

@app.route('/history', methods=['GET'])
def history():
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id, timestamp, role, content FROM chat_history ORDER BY timestamp DESC")
    history = cursor.fetchall()
    cursor.close()
    conn.close()
    
    return jsonify(history)

if __name__ == '__main__':
    app.run(debug=True)
