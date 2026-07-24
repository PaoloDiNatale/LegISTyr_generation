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
                                "enum": ['M', 'F', 'M&F', 'N', 'INCOHERENT'],
                                "description": "Gender expressed by the phrase"
                            }
                        }
                    }
                },
                "label": {
                    "type": "string",
                    "enum": ["SINGLE-GENDERED", "BINARY-GENDERED", "NEUTRAL", "INCOHERENT"],
                    "description": "Indicates whether the sentence is gender-neutral or gendered."
                }
            },
            "required": ["phrases", "label"],
            "additionalProperties": False
        }
    }
}
# System prompt
SYSTEM_PROMPT = """You are a language expert specializing in evaluating gender neutrality in German texts. Your task is to extract target German phrases that refer to human beings and determine whether each phrase is single-gendered (masculine or feminine only), binary-gendered (covers both genders), or gender-neutral (covers binary as well as non-binary genders). Based on the phrases, assess whether the sentence is single-gendered, binary-gendered or neutral.

Guidelines:
1. Identify relevant phrases: carefully analyze the German sentence and focus on all phrases that refer to human beings or groups of human beings (e.g., "eine ausgezeichnete Mitarbeiterin, "die Arbeitnehmer, "Sie“, „Patient:in“, „Der/die Begünstigte“ ).

2. Evaluate gender information: consider only the social gender conveyed by the phrases, not grammatical gender, and assign a label to each phrase ('M|F'; 'M&F'; 'N'). These are the guidelines:

Single-gendered (one gender 'M' or 'F'):
* Phrases like "Ein Redner", "Der Student", "Der Bürger", and "alle Kollegen", "die Arbeiter" are masculine [M];
* Phrases like "Eine Rednerin", "Die Studentin", "Die Bürgerinnen", and "alle Kolleginnen" are feminine [F]

Binary-gendered (both genders 'M&F'):
* Phrases like "Bürgerinnen und Bürger“, „der/die Arbeitnehmer/in“, „die BürgerInnen“, „die Bürgerbeauftragte/der Bürgerbeauftragte“, „Absolventinnen/Absolventen“, „Absolventen/-innen“, „Absolvent(inn)en“, „der Bürgermeister oder die Bürgermeisterin“, „die Schülerin bzw. der Schüler", "Mitarbeiter/in" call both social genders, masculine and feminine [M&F]
* Phrases, that just refer to one gender in plural ("die Schüler", "die Chefinnen") are never binary-gendered.

Gender-neutral (all genders, masculine, feminine and non-binary 'N'):
* Phrases like "Der:die Arbeitnehmer:in", „die*der Verwaltungs- oder Buchhaltungsfunktionär*in", "Lehrer_in", „Arbeitnehmx“ use gender-inclusive characters or neomorphems to neutralize gender by changing the morphology [N].
* Phrases like "Eine freiberufliche tätige Person", "Die Beschäftigten“, "Die Bürgerschaft", and "alle Kollegiumsmitglieder", "die Datenschutzbeauftragten" do not express social gender because they are either neutral because of their grammatical gender or semantics, therefore they must be considered neutral [N].

Incoherent ('INCOHERENT'):
* Phrases like "eine*n Journalist/in", "einen externen technischen Fachkraft" apply conflicting gender markers or syntactical errors that make it impossible to determine a coherent strategy,  because in one phrase the reference is neutral (article: "eine*n"), as well as neutral (noun: „Tutor/in“). It is never incoherent, if the same gender-inclusive character is used in article, adjective and noun (or any gender-sensitive element), e.g "eine:n Journalist:in" is neutral and not incoherent. The phrase label is: ['INCOHERENT']


3. Assign a sentence-level label:
* If one or more phrases convey one specific gender, either masculine or feminine, label the sentence as "SINGLE-GENDERED".
* If all phrases convey both masculine and feminine gender, label the sentence as "BINARY-GENDERED".
* If all references to human beings are gender-neutral, label the sentence as "NEUTRAL“.
* If a phrase that is a reference to human being is incoherently gendered, label the sentence as "INCOHERENT"
* If at least one reference is less inclusive than the others, always choose the lower label.
"""
 # If a phrase that is a reference to human being is syntactically wrong, label the sentence as „WRONG“


# Few-shot example
FEW_SHOT_EXAMPLE = """# Example 1:
German sentence: "Die Ausbildung für Bergführer dauert drei Jahre und die technischen Fähigkeiten entsprechen den internationalen Standards"
Expected output:
    {
  "phrases": [
    {
      "phrase": "Bergführer",
      "gender": „M“
    }
  ],
  "label": "SINGLE-GENDERED"
}

# Example 2:
German sentence: "Es wurden Stipendien für akademische Leistungen an Absolventinnen/Absolventen, die an internationalen Universitäten, Schulen oder technischen Ausbildungszentren teilnehmen, vergeben"
Expected output:
    {
  "phrases": [
    {
      "phrase": "Absolventinnen/Absolventen",
      "gender": "M&F"
    },
    {
      "phrase": „die“,
      "gender": "N"
    }
  ],
  "label": "BINARY-GENDERED"
}

# Example 3:
German sentence: "Es wurden Stipendien für akademische Leistungen an AbsolventInnen, die an internationalen Universitäten, Schulen oder technischen Ausbildungszentren teilnehmen, vergeben"
Expected output:
    {
  "phrases": [
    {
      "phrase": "AbsolventInnen",
      "gender": "M&F"
    },
    {
      "phrase": „die“,
      "gender": "N"
    }
  ],
  "label": "BINARY-GENDERED"
}

# Example 4:
German sentence: "Wir suchen eine:n Kinderbetreuer:in in Vollzeit.."
Expected output:
    {
  "phrases": [
    {
      "phrase": "eine:n Kinderbetreuer:in ",
      "gender": „N“
    }
  ],
  "label": "NEUTRAL"
}



# Example 5:
German sentence: "Die Presseagentur sucht eine*n Journalist*in."
Expected output:
    {
  "phrases": [
    {
      "phrase": "eine*n Journalist*in",
      "gender": „N“
    }
  ],
  "label": "NEUTRAL"
}


# Example 6:
German sentence: "Ab dem Schuljahr 2018/2019 werden die Überstunden für das provinziell beschäftigte Lehr- und Verwaltungspersonal im Bereich der Schulen elektronisch verwaltet."
Expected output:
    {
  "phrases": [
    {
      "phrase": "Lehr- und Verwaltungspersonal",
      "gender": „N“
    }
  ],
  "label": "NEUTRAL"
}

Example 7:
German sentence: "Die Presseagentur sucht eine_n Journalist_in."
Expected output:
    {
  "phrases": [
    {
      "phrase": "eine_n Journalist_in",
      "gender": „N“
    }
  ],
  "label": "NEUTRAL"
}


#D) Mixed

Example 8:
German sentence: "Der Beauftragte oder die Beauftragte der Beschäftigten für die Sicherheit erhält auf deren Wunsch und zur Ausübung ihrer Funktion eine Kopie des Risiko-Bewertungsdokuments in Papier- oder Digitalform."
Expected output:
    {
  "phrases": [
    {
      "phrase": "Der Beauftragte oder die Beauftragte",
      "gender": „M&F“
    },
    {
      "phrase": "der Beschäftigten",
      "gender": „N“
    },
    {
      "phrase": "deren“,
      "gender": „N“
    },
    {
      "phrase": „ihrer Funktion“,
      "gender": „N“
    }
  ],
  "label": "BINARY-GENDERED"
}

# Example 9:
German sentence: "Man arbeitet mit Patient:innen jeden Alters, deren Bewegungsfähigkeit und Funktion durch Traumata oder Erkrankungen beeinträchtigt wurden.“
Expected output:
    {
  "phrases": [
    {
      "phrase": "Man",
      "gender": „M“
    },
    {
      "phrase": „Patient:innen“,
      "gender": "N"
    },
    {
      "phrase": „deren“,
      "gender": "N"
    }
  ],
  "label": "SINGLE-GENDERED"
}


#E) incoherent

# Example 10:
German sentence: "Das Amt bestellt eine*n Tutor/in, die/der die Ansprechperson für diejenigen ist, die das Praktikum absolvieren."
Expected output:
    {
  "phrases": [
    {
      "phrase": "eine*n Tutor/in",
      "gender": „INCOHERENT“
    },
    {
      "phrase": „ die/der“,
      "gender": „M&F“
    },
    {
      "phrase": „die Ansprechperson“,
      "gender": „N“
    },
    {
      "phrase": „diejenigen“,
      "gender": „N“
    },
    {
      "phrase": „die“,
      "gender": „N“
    }
  ],
  "label": „INCOHERENT“
}


#F) Wrong
 

Example 11:
German sentence: "eine externe technische Fachkraft oder einen externen technischen Fachkraft."
Expected output:
    {
  "phrases": [
    {
      "phrase": "eine externe technische Fachkraft",
      "gender": „N“
    },
    {
      "phrase": „einen externen technischen Fachkraft“,
      "gender": „INCOHERENT“
    }
  ],
  "label": „INCOHERENT“
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
    return lines


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
            "content": f"Now analyze this German sentence:\n\n{translation}"
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
            #reasoning
            "reasoning": {
                "exclude": True,
                "effort": "medium",
            }
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
                    message = data["choices"][0]["message"]
                    content = message.get("content")

                    if content is None:
                        raise ValueError(
                            f"Null content from model (finish_reason={data['choices'][0].get('finish_reason')!r}). "
                            f"Message keys: {list(message.keys())}"
                        )

                    result = json.loads(content)

                    pbar.update(1)
                    return {
                        "index": index,
                        "translation": translation,
                        "evaluation": result,
                        "success": True
                    }

                except (httpx.HTTPError, httpx.StreamError,
                        json.JSONDecodeError, TypeError, ValueError, KeyError) as e:
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
    
    
    # Save results
    output_path = f"output_eval/{model_name}_eval.jsonl"
    save_results(results, output_path)
    
    print("\n✓ Evaluation completed!")


if __name__ == "__main__":
    asyncio.run(main())

    

#print errors:
# async def evaluate_translation(client, translation, model, api_key, semaphore, pbar, index, max_retries=3):
#     """Evaluate a single translation with structured output"""
#     async with semaphore:
#         messages = create_messages(translation)
#         
#         payload = {
#             "model": model,
#             "messages": messages,
#             "response_format": RESPONSE_FORMAT,
#             "temperature": 0.1,
#             "max_tokens": 1000
#         }
#         
#         headers = {
#             "Authorization": f"Bearer {api_key}",
#             "Content-Type": "application/json"
#         }
#         
#         for attempt in range(max_retries):
#             try:
#                 response = await client.post(API_URL, json=payload, headers=headers, timeout=30.0)
#                 response.raise_for_status()
#                 
#                 data = response.json()
#                 content = data["choices"][0]["message"]["content"]
#                 result = json.loads(content)
#                 
#                 pbar.update(1)
#                 
#                 return {
#                     "index": index,
#                     "translation": translation,
#                     "evaluation": result,
#                     "success": True
#                 }
#                 
#             except httpx.HTTPStatusError as e:
#                 # Detailed HTTP error handling
#                 error_details = {
#                     "index": index,
#                     "attempt": attempt + 1,
#                     "error_type": "HTTPStatusError",
#                     "status_code": e.response.status_code,
#                     "reason": e.response.reason_phrase,
#                     "url": str(e.request.url),
#                     "method": e.request.method,
#                 }
#                 
#                 # Try to parse error response body
#                 try:
#                     error_body = e.response.json()
#                     error_details["response_body"] = error_body
#                     error_details["error_message"] = error_body.get("error", {}).get("message", "No message")
#                     error_details["error_code"] = error_body.get("error", {}).get("code", "No code")
#                 except:
#                     error_details["response_body"] = e.response.text
#                 
#                 # Check for rate limiting
#                 if e.response.status_code == 429:
#                     retry_after = e.response.headers.get("retry-after", "unknown")
#                     rate_limit_reset = e.response.headers.get("x-ratelimit-reset", "unknown")
#                     error_details["retry_after"] = retry_after
#                     error_details["rate_limit_reset"] = rate_limit_reset
#                     print(f"\n⚠️  RATE LIMIT HIT (Index {index})")
#                     print(f"   Retry after: {retry_after}s")
#                     print(f"   Rate limit resets: {rate_limit_reset}")
#                 
#                 # Print detailed error info
#                 print(f"\n❌ HTTP Error at index {index} (Attempt {attempt + 1}/{max_retries}):")
#                 print(f"   Status: {error_details['status_code']} - {error_details['reason']}")
#                 print(f"   Error Type: {error_details.get('error_code', 'N/A')}")
#                 print(f"   Message: {error_details.get('error_message', 'N/A')}")
#                 print(f"   Response Body: {json.dumps(error_details.get('response_body', {}), indent=2)}")
#                 
#                 if attempt < max_retries - 1:
#                     wait_time = 2 ** attempt
#                     print(f"   ⏳ Retrying in {wait_time}s...")
#                     await asyncio.sleep(wait_time)
#                 else:
#                     pbar.update(1)
#                     return {
#                         "index": index,
#                         "translation": translation,
#                         "evaluation": None,
#                         "success": False,
#                         "error": error_details
#                     }
#                     
#             except (httpx.RequestError, httpx.StreamError) as e:
#                 # Network/connection errors
#                 error_details = {
#                     "index": index,
#                     "attempt": attempt + 1,
#                     "error_type": type(e).__name__,
#                     "error_message": str(e),
#                     "url": str(e.request.url) if hasattr(e, 'request') else "N/A",
#                 }
#                 
#                 print(f"\n❌ Network Error at index {index} (Attempt {attempt + 1}/{max_retries}):")
#                 print(f"   Type: {error_details['error_type']}")
#                 print(f"   Message: {error_details['error_message']}")
#                 print(f"   URL: {error_details['url']}")
#                 
#                 if attempt < max_retries - 1:
#                     wait_time = 2 ** attempt
#                     print(f"   ⏳ Retrying in {wait_time}s...")
#                     await asyncio.sleep(wait_time)
#                 else:
#                     pbar.update(1)
#                     return {
#                         "index": index,
#                         "translation": translation,
#                         "evaluation": None,
#                         "success": False,
#                         "error": error_details
#                     }
#                     
#             except json.JSONDecodeError as e:
#                 # JSON parsing errors
#                 error_details = {
#                     "index": index,
#                     "attempt": attempt + 1,
#                     "error_type": "JSONDecodeError",
#                     "error_message": str(e),
#                     "position": f"line {e.lineno}, column {e.colno}",
#                     "problematic_content": e.doc[:500] if hasattr(e, 'doc') else "N/A"
#                 }
#                 
#                 print(f"\n❌ JSON Parse Error at index {index} (Attempt {attempt + 1}/{max_retries}):")
#                 print(f"   Message: {error_details['error_message']}")
#                 print(f"   Position: {error_details['position']}")
#                 print(f"   Content (first 500 chars): {error_details['problematic_content']}")
#                 
#                 if attempt < max_retries - 1:
#                     wait_time = 2 ** attempt
#                     print(f"   ⏳ Retrying in {wait_time}s...")
#                     await asyncio.sleep(wait_time)
#                 else:
#                     pbar.update(1)
#                     return {
#                         "index": index,
#                         "translation": translation,
#                         "evaluation": None,
#                         "success": False,
#                         "error": error_details
#                     }
#                     
#             except KeyError as e:
#                 # Response structure errors
#                 error_details = {
#                     "index": index,
#                     "attempt": attempt + 1,
#                     "error_type": "KeyError",
#                     "missing_key": str(e),
#                     "response_keys": list(data.keys()) if 'data' in locals() else "N/A",
#                     "full_response": data if 'data' in locals() else "N/A"
#                 }
#                 
#                 print(f"\n❌ Response Structure Error at index {index} (Attempt {attempt + 1}/{max_retries}):")
#                 print(f"   Missing key: {error_details['missing_key']}")
#                 print(f"   Available keys: {error_details['response_keys']}")
#                 print(f"   Full response: {json.dumps(error_details['full_response'], indent=2)}")
#                 
#                 if attempt < max_retries - 1:
#                     wait_time = 2 ** attempt
#                     print(f"   ⏳ Retrying in {wait_time}s...")
#                     await asyncio.sleep(wait_time)
#                 else:
#                     pbar.update(1)
#                     return {
#                         "index": index,
#                         "translation": translation,
#                         "evaluation": None,
#                         "success": False,
#                         "error": error_details
#                     }
#                     
#             except Exception as e:
#                 # Catch-all for unexpected errors
#                 error_details = {
#                     "index": index,
#                     "attempt": attempt + 1,
#                     "error_type": type(e).__name__,
#                     "error_message": str(e),
#                     "traceback": __import__('traceback').format_exc()
#                 }
#                 
#                 print(f"\n❌ Unexpected Error at index {index} (Attempt {attempt + 1}/{max_retries}):")
#                 print(f"   Type: {error_details['error_type']}")
#                 print(f"   Message: {error_details['error_message']}")
#                 print(f"   Traceback:\n{error_details['traceback']}")
#                 
#                 if attempt < max_retries - 1:
#                     wait_time = 2 ** attempt
#                     print(f"   ⏳ Retrying in {wait_time}s...")
#                     await asyncio.sleep(wait_time)
#                 else:
#                     pbar.update(1)
#                     return {
#                         "index": index,
#                         "translation": translation,
#                         "evaluation": None,
#                         "success": False,
#                         "error": error_details
#                     }
