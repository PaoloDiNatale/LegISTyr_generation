# Here you can define your call template for the API request payload
# If in dounbt, refer to OpenRouter API documentation, particularly 

def get_payload(model, prompt, max_tokens=1000, temperature=0.1):
    
    payload = {
    "model": model,
    "max_tokens": max_tokens,
    "temperature": temperature,
    "top_p": 0.9,
    "data_collection": "deny",
    "messages": prompt,
    "usage": {
        "include": True
    },
    "reasoning": {
        "effort": "low",
        "exclude": False,
    }
}
    return payload