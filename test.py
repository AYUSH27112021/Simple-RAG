import requests
import json

BASE_URL = "http://127.0.0.1:5000"  

def ask_question(question):
    """ Sends a question to the chatbot API and returns the response. """
    url = f"{BASE_URL}/chat"
    payload = {"query": question}
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        return response.json().get("answer")
    else:
        print("Error:", response.json())
        return None

def get_chat_history():
    """ Retrieves and prints chat history from the chatbot API. """
    url = f"{BASE_URL}/history"
    response = requests.get(url)
    
    if response.status_code == 200:
        history = response.json()
        for entry in history:
            print(f"[{entry['timestamp']}] {entry['role'].capitalize()}: {entry['content']}")
    else:
        print("Error fetching history:", response.json())

if __name__ == "__main__":
    question = input("Enter your question: ")
    answer = ask_question(question)

    if answer:
        print("Chatbot Answer:", answer)
    
    print("\nChat History:")
    get_chat_history()
    
