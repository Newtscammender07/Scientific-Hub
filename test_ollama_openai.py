import requests
import json

def test_ollama_openai():
    base_url = "http://127.0.0.1:11434/v1"
    model = "llama3.2:latest"
    
    print(f"Testing Ollama OpenAI endpoint at {base_url}...")
    try:
        # Test basic chat completions
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "Say hi"}],
            "max_tokens": 10
        }
        resp = requests.post(f"{base_url}/chat/completions", json=payload)
        print(f"Response Status: {resp.status_code}")
        if resp.status_code == 200:
            print("SUCCESS: Ollama OpenAI-compatible endpoint is working!")
            print(f"Response: {resp.json()['choices'][0]['message']['content']}")
        else:
            print(f"FAILED: {resp.text}")
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    test_ollama_openai()
