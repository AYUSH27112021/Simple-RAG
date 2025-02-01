# RAG Chatbot

## Overview

This project implements a Retrieval-Augmented Generation (RAG) chatbot using Flask. It leverages a vector database for semantic search and stores chat history in a MySQL database.

## Features

- Text corpus processing (chunking and cleaning)
- Embedding creation using `langchain-nomic "nomic[local]"`
- Vector store using `SKLearnVectorStore`
- Retrieval of relevant chunks based on user queries
- Generation of responses using `langgraph` implementation
- Flask API with endpoints for chat and history retrieval
- MySQL database for storing chat history
- Unit testing for validation

## Setup Instructions

### 1. Create a Virtual Environment

```sh
python -m venv venv python 3.10.10
source venv/bin/activate
```

### 2. Install Dependencies

```sh
pip install -r requirements.txt
```

### 3. Set Up MySQL Database

Ensure MySQL is installed and running. Create a database and a table to store chat history.
Replace DB_CONFIG with your db config. in `app.py`
```py
DB_CONFIG = {
    "host": "localhost",
    "user": "Enter-your-username",
    "password": "Enter-your-password",
    "database": "chatbot"
}
```

### 4. Run the Flask Server

```sh
python app.py
```

### 5. Test the RAG Chatbot

In another terminal, run:

```sh
python test.py
```

This script sends test requests to the Flask API and verifies responses.

## API Endpoints

### 1. Chat Endpoint

#### **POST **``

- **Request:**
  ```json
  {
    "query": "What are Agents"
  }
  ```
- **Response:**
  ```json
  {
    "context": "RAG (Retrieval-Augmented Generation) is a hybrid approach that combines information retrieval with generative AI models.",....
  }
  ```

### 2. History Endpoint

#### **GET **``

- **Response:**
  ```json
  [
    {"id": 1, "timestamp": "2024-02-01T12:00:00", "role": "user", "content": "What is RAG?"},
    {"id": 2, "timestamp": "2024-02-01T12:00:02", "role": "system", "content": "RAG (Retrieval-Augmented Generation) is..."}
  ]
  ```

## Project Structure

```
|-- app.py                # Flask API server
|-- test.py               # Testing script using request
|-- requirements.txt      # Dependencies
|-- database.sql          # MySQL setup script
|-- Data_preprocessing.py # Corpus files loading and splitting
|-- GraphRag.py           # `langgraph` implementation for retriever,generator,grader,hallucination checker
|-- Embed_store.py        # Stored embeddings
|-- README.md             # Project documentation
```

## Video Demonstration

A video showcasing the working chatbot is included in the project.

## Future Enhancements

- Implement user authentication
- Extend support for other vector databases (e.g., FAISS, Chroma)
- Deploy using Docker and cloud services

## Author
AYUSH KUMAR SINGH

