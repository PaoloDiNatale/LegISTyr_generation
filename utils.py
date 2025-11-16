# -*- coding: utf-8 -*-
"""
Functions for translation API calls
"""

import asyncio
import httpx
import pandas as pd
import json
from tqdm.asyncio import tqdm
from prompt_templates import get_prompt_template
from call_template import get_payload

API_URL = "https://openrouter.ai/api/v1/chat/completions"


def load_dataset(file_path):
    """
    Load dataset from CSV file
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    dataset = pd.read_csv(file_path, sep=';', encoding="utf-8")
    dataset.columns = dataset.columns.str.strip()  # Fix column name issues
    return dataset


def create_prompts(dataset, source_name):
    """
    Create prompts from dataset using appropriate template
    
    Args:
        dataset (pd.DataFrame): Source dataset with columns IT EXAMPLE, IT TERM, OPTIONS
        source_name (str): Name of the source dataset to determine which template to use
        
    Returns:
        list: List of prompt dictionaries
    """
    source_sentences = dataset["IT EXAMPLE"].tolist()
    terms_it = dataset["IT TERM"].tolist()

    if source_name == "homonyms":
        terms_de = dataset["OPTIONS"].tolist()
    elif source_name == "simple_terms" or source_name == "abbreviations":
        terms_de = dataset["TARGET HYPOTHESIS (DE SOUTH TYROL)"].tolist()
    else:
        pass
    
    # Get the appropriate prompt template function
    prompt_template = get_prompt_template(source_name)
    
    prompts = []
    
    for source_sentence, term_it, term_de in zip(source_sentences, terms_it, terms_de):
        prompt = prompt_template(source_sentence, term_it, term_de)
        prompts.append(prompt)
    
    return prompts


async def fetch_completion(prompt, model, client, semaphore, pbar, api_key, max_tokens=1000, 
                          temperature=0.1, max_retries=3):
    """
    Fetch a single completion with retry logic
    
    Args:
        prompt: The prompt to send
        model (str): Model name
        client: HTTP client
        semaphore: Async semaphore for concurrency control
        pbar: Progress bar
        api_key (str): API key
        max_tokens (int): Maximum tokens
        temperature (float): Temperature setting
        max_retries (int): Maximum retry attempts
        
    Returns:
        Response object or None
    """
    async with semaphore:
        payload = get_payload(model, prompt, max_tokens, temperature)
        headers = {"Authorization": f"Bearer {api_key}"}

        for attempt in range(max_retries):
            try:
                resp = await client.post(API_URL, json=payload, headers=headers, timeout=30.0)
                resp.raise_for_status()
                pbar.update(1)
                return resp
            except (httpx.HTTPError, httpx.StreamError) as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"\nRetry {attempt + 1}/{max_retries} after error: {e}. Waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"\nFailed after {max_retries} attempts: {e}")
                    pbar.update(1)
                    return None


async def run_parallel_requests(prompts, model, api_key, max_tokens=1000, 
                                temperature=0.1, max_concurrent=15):
    """
    Run parallel API requests
    
    Args:
        prompts (list): List of prompts
        model (str): Model name
        api_key (str): API key
        max_tokens (int): Maximum tokens
        temperature (float): Temperature setting
        max_concurrent (int): Maximum concurrent requests
        
    Returns:
        list: List of responses
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async with httpx.AsyncClient(http2=True, timeout=30.0) as client:
        with tqdm(total=len(prompts), desc="Processing prompts") as pbar:
            tasks = [
                fetch_completion(prompt, model, client, semaphore, pbar, api_key, 
                               max_tokens, temperature) 
                for prompt in prompts
            ]
            results = await asyncio.gather(*tasks)

    return results


def process_responses(results):
    """
    Process API responses and extract relevant information
    
    Args:
        results (list): List of response objects
        
    Returns:
        list: List of dictionaries containing processed data
    """
    model_output = []
    
    print(f"Total responses to process: {len(results)}")
    
    for i, r in enumerate(results):
        if r is None or not getattr(r, "text", None):
            print(f"Warning: Response {i} missing or empty")
            model_output.append({
                "index": i,
                "assistant": None,
                "reasoning": None,
                "cost": None,
                "reasoning_tokens": None
            })
            continue
        
        try:
            data = r.json()
            assistant = data["choices"][0]["message"]["content"]
            reasoning = data["choices"][0]["message"].get("reasoning")
            cost = data.get("usage", {}).get("cost_details", {}).get("upstream_inference_completions_cost")
            reasoning_tok = data.get("usage", {}).get("completion_tokens_details", {}).get("reasoning_tokens")
            
            model_output.append({
                "index": i,
                "assistant": assistant,
                "reasoning": reasoning,
                "cost": cost,
                "reasoning_tokens": reasoning_tok
            })
            
        except json.JSONDecodeError as e:
            print(f"✗ JSONDecodeError at response {i}: {e}")
            model_output.append({
                "index": i,
                "assistant": None,
                "reasoning": None,
                "cost": None,
                "reasoning_tokens": None
            })
        except Exception as e:
            print(f"✗ Other error at response {i}: {type(e).__name__}: {e}")
            model_output.append({
                "index": i,
                "assistant": None,
                "reasoning": None,
                "cost": None,
                "reasoning_tokens": None
            })
    
    return model_output


def save_to_excel(model_output, output_path):
    """
    Save model output to CSV file
    
    Args:
        model_output (list): List of output dictionaries
        output_path (str): Path to save CSV file
    """
    df = pd.DataFrame(model_output)
    df.to_csv(output_path, index=False, encoding="utf-8")


def save_to_txt(csv_path, txt_path):
    """
    Save assistant column from CSV to TXT file
    
    Args:
        csv_path (str): Path to CSV file
        txt_path (str): Path to save TXT file
    """
    df = pd.read_csv(csv_path)
    
    with open(txt_path, "w", encoding="utf-8") as f:
        for value in df["assistant"]:
            if pd.isna(value) or str(value).strip() == "":
                line = "vuoto"
            else:
                line = (
                    str(value)
                    .replace("<think>", "")
                    .replace("</think>", "")
                    .replace("\n", " ")
                    .replace("\r", " ")
                    .strip()
                )
            f.write(line + "\n")