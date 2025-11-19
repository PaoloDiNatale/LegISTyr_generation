# Here you can define your call template for the API request payload
# If in doubt, refer to OpenRouter API documentation, particularly 

def get_payload(model, prompt, max_tokens=1200, temperature=0.2):
    
    payload = {
    "model": model,
    "max_tokens": max_tokens,
    "temperature": temperature,
    "top_p": 0.9,
    "no_repeat_ngram_size": 4,
    #"repetition_penalty": 1.2,
    "data_collection": "deny",
    "messages": prompt,
    "usage": {
        "include": True
    },
    "reasoning": {
        "exclude": False,
        "effort": "low"
    }
}
    return payload