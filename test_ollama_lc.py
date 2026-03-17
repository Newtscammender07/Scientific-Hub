from langchain_community.chat_models import ChatOllama
import requests

def test_ollama():
    model = "llama3.2:latest"
    base_url = "http://localhost:11434"
    
    print(f"Testing connectivity to {base_url}...")
    try:
        resp = requests.get(f"{base_url}/api/tags")
        print(f"API Tags Response Status: {resp.status_code}")
        print(f"Models available: {[m['name'] for m in resp.json().get('models', [])]}")
    except Exception as e:
        print(f"Failed to connect to Ollama API: {e}")
        return

    print(f"\nTesting ChatOllama with model {model}...")
    try:
        llm = ChatOllama(model=model, base_url=base_url)
        # Try a simple completion
        res = llm.predict("hi")
        print(f"ChatOllama Response: {res}")
    except Exception as e:
        print(f"ChatOllama failed: {type(e).__name__}: {e}")

if __name__ == "__main__":
    test_ollama()
