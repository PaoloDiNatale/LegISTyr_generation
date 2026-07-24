# Here you can define your call template for the API request payload
# If in doubt, refer to OpenRouter API documentation, particularly 

def get_payload(model, prompt, max_tokens, temperature):
    
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
        "enabled": False,
        "max_tokens": 0
    }
}
    return payload
