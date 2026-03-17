from crewai import LLM
import os

def test_litellm_openai_prefix():
    # Force litellm to use OpenAI handler for Ollama
    # This is often more stable than the native ollama/ handler
    model = "openai/llama3.2:latest"
    base_url = "http://127.0.0.1:11434/v1"
    
    print(f"Testing litellm with OpenAI prefix for Ollama...")
    print(f"Model: {model}")
    print(f"Base URL: {base_url}")
    
    try:
        # We need a dummy API key for the openai/ prefix to work
        llm = LLM(model=model, base_url=base_url, api_key="ollama")
        
        print("Invoking LLM...")
        # Note: CrewAI's call method is used internally, we can test it directly
        response = llm.call([{"role": "user", "content": "hi"}])
        print("SUCCESS!")
        print(f"Response: {response}")
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")

if __name__ == "__main__":
    test_litellm_openai_prefix()
