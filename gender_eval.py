# -*- coding: utf-8 -*-
"""
Gender evaluation script with structured JSON output
"""

import argparse
import asyncio
import json
import httpx
from pathlib import Path
from tqdm.asyncio import tqdm


# OpenRouter API endpoint
API_URL = "https://openrouter.ai/api/v1/chat/completions"


# JSON Schema for structured output
RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "gender_evaluation",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "phrases": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 10,
                    "items": {
                        "type": "object",
                        "required": ["phrase", "gender"],
                        "additionalProperties": False,
                        "properties": {
                            "phrase": {
                                "type": "string",
                                "description": "Phrase from the target sentence being annotated"
                            },
                            "gender": {
                                "type": "string",
                                "enum": ["M", "F", "N"],
                                "description": "Gender expressed by the phrase"
                            }
                        }
                    }
                },
                "label": {
                    "type": "string",
                    "enum": ["NEUTRAL", "GENDERED"],
                    "description": "Indicates whether the sentence is gender-neutral or gendered."
                }
            },
            "required": ["phrases", "label"],
            "additionalProperties": False
        }
    }
}


# System prompt
SYSTEM_PROMPT = """You are a language expert specializing in evaluating gender neutrality in German texts. Your task is to extract target German phrases that refer to human beings and determine whether each phrase is masculine, feminine, or neutral. Based on the phrases, assess whether the sentence is gendered or neutral.

Guidelines:
1. Identify relevant phrases: carefully analyze the German sentence and focus on all phrases that refer to human beings or groups of human beings (e.g., "eine ausgezeichnete Rednerin", "die Bürgerschaft", "Sie").

2. Evaluate gender information: consider only the social gender conveyed by the phrases, not grammatical gender, and assign a label to each phrase [M/F/N]. For example:
* Phrases like "Ein Redner", "Der Student", "Der Bürger", and "alle Kollegen" are masculine [M];
* Phrases like "Eine Rednerin", "Die Studentin", "Die Bürgerinnen", and "alle Kolleginnen" are feminine [F];
* Phrases like "Eine referierende Person", "Die Studierenden", "Die Bürgerschaft", and "alle Kollegiumsmitgliedern" do not express social gender, therefore they must be considered neutral [N].

3. Assign a sentence-level label:
* If all references to human beings are gender-neutral, label the sentence as "NEUTRAL".
* If one or more phrases convey a specific masculine or feminine gender, label the sentence as "GENDERED"."""

# Few-shot example
FEW_SHOT_EXAMPLE = """Example:
Italian sentence: "Frau Präsidentin, ich möchte das unterstreichen, was meine Kollegin Hernández Mollar gerade gesagt hat."
Expected output:
	{
  "phrases": [
    {
      "phrase": "Frau Präsidentin",
      "gender": "F"
    },
    {
      "phrase": "ich",
      "gender": "N"
    },
    {
      "phrase": "meine Kollegin Hernández Mollar",
      "gender": "F"
    }
  ],
  "label": "GENDERED"
}"""



def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate gender neutrality in Italian translations')
    
    parser.add_argument(
        '--input-file',
        type=str,
        required=True,
        help='Name of input file (e.g., "model_name" for output_txt/gender/model_name.txt)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='openai/gpt-4o-mini',
        help='Model to use via OpenRouter (default: openai/gpt-4o-mini)'
    )
    
    parser.add_argument(
        '--parallel',
        type=int,
        default=10,
        help='Number of parallel processes (default: 10)'
    )
    
    parser.add_argument(
        '--api-key',
        type=str,
        required=True,
        help='OpenRouter API key'
    )
    
    return parser.parse_args()


def load_translations(file_path):
    """Load translations from input file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines[0:2]


def create_messages(translation):
    """Create message list for API call"""
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": FEW_SHOT_EXAMPLE
        },
        {
            "role": "user",
            "content": f"Now analyze this Italian sentence:\n\n{translation}"
        }
    ]
    return messages


async def evaluate_translation(client, translation, model, api_key, semaphore, pbar, index, max_retries=3):
    """Evaluate a single translation with structured output"""
    async with semaphore:
        messages = create_messages(translation)
        
        payload = {
            "model": model,
            "messages": messages,
            "response_format": RESPONSE_FORMAT,
            "temperature": 0.1,
            "max_tokens": 1000
        }
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        for attempt in range(max_retries):
            try:
                response = await client.post(API_URL, json=payload, headers=headers, timeout=30.0)
                response.raise_for_status()
                
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                result = json.loads(content)
                
                pbar.update(1)
                
                return {
                    "index": index,
                    "translation": translation,
                    "evaluation": result,
                    "success": True
                }
                
            except (httpx.HTTPError, httpx.StreamError, json.JSONDecodeError) as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"\nRetry {attempt + 1}/{max_retries} for index {index}: {e}. Waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"\nFailed at index {index} after {max_retries} attempts: {e}")
                    pbar.update(1)
                    return {
                        "index": index,
                        "translation": translation,
                        "evaluation": None,
                        "success": False,
                        "error": str(e)
                    }


async def run_evaluations(translations, model, api_key, max_parallel):
    """Run parallel evaluations via OpenRouter"""
    semaphore = asyncio.Semaphore(max_parallel)
    
    async with httpx.AsyncClient(http2=True, timeout=30.0) as client:
        with tqdm(total=len(translations), desc="Evaluating translations") as pbar:
            tasks = [
                evaluate_translation(client, translation, model, api_key, semaphore, pbar, i)
                for i, translation in enumerate(translations)
            ]
            results = await asyncio.gather(*tasks)
    
    return results


def save_results(results, output_path):
    """Save results to JSONL file"""
    # Create output directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    print(f"\nResults saved to: {output_path}")


async def main():
    """Main execution function"""
    args = parse_arguments()
    
    # Extract model name for output file
    model_name = args.input_file.replace("/", "_").replace(".txt", "")
    
    # Load translations
    input_path = f"output_txt/gender/{args.input_file}.txt"
    print(f"Loading translations from: {input_path}")
    translations = load_translations(input_path)
    print(f"Loaded {len(translations)} translations")
    
    # Run evaluations
    print(f"\nRunning evaluations with model: {args.model}")
    print(f"Parallel processes: {args.parallel}")
    
    results = await run_evaluations(
        translations=translations,
        model=args.model,
        api_key=args.api_key,
        max_parallel=args.parallel
    )
    
    # Calculate success rate
    successful = sum(1 for r in results if r['success'])
    print(f"\nSuccessfully evaluated: {successful}/{len(results)}")
    
    # Save results
    output_path = f"output_eval/{model_name}_eval.jsonl"
    save_results(results, output_path)
    
    print("\n✓ Evaluation completed!")


if __name__ == "__main__":
    asyncio.run(main())