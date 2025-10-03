"""
Example Python client for testing the Sierra Leone Legal Assistant API
"""

import requests
import json

# API endpoint
API_URL = "http://localhost:8000"


def check_health():
    """Check if the API is healthy"""
    response = requests.get(f"{API_URL}/health")
    print("Health Check:")
    print(json.dumps(response.json(), indent=2))
    print()


def ask_question(question):
    """Ask a question to the API"""
    response = requests.post(
        f"{API_URL}/ask",
        json={"question": question}
    )

    if response.status_code == 200:
        data = response.json()
        print(f"Question: {data['question']}")
        print(f"\nAnswer: {data['answer']}")
        print(f"\nSources ({len(data['sources'])}):")
        for i, source in enumerate(data['sources'], 1):
            print(f"\n  {i}. {source['source']}")
            if source['page']:
                print(f"     Page: {source['page']}")
            print(f"     Preview: {source['content_preview'][:100]}...")
    else:
        print(f"Error: {response.status_code}")
        print(response.json())

    print("\n" + "=" * 80 + "\n")


def ask_simple(question):
    """Ask a question using the simple endpoint (no sources)"""
    response = requests.post(
        f"{API_URL}/ask-simple",
        json={"question": question}
    )

    if response.status_code == 200:
        data = response.json()
        print(f"Question: {data['question']}")
        print(f"\nAnswer: {data['answer']}")
    else:
        print(f"Error: {response.status_code}")
        print(response.json())

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    print("=" * 80)
    print("Sierra Leone Legal Assistant API - Test Client")
    print("=" * 80 + "\n")

    # Check API health
    check_health()

    # Example questions
    questions = [
        "What are my rights if I'm arrested?",
        "How do I register a business in Sierra Leone?",
        "What does the Constitution say about freedom of speech?"
    ]

    for question in questions:
        ask_question(question)

    # Example of simple endpoint
    print("Testing simple endpoint (no sources):")
    print("-" * 80)
    ask_simple("What are the requirements for voting?")