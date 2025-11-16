# Translation API Project

Restructured project for parallel translation API calls using OpenRouter.

## Project Structure

```
project/
│
├── main.py                 # Main execution script with CLI
├── functions.py            # Core functions
├── prompt_templates.py     # Prompt templates for different sources
├── data/                   # Input data folder
│   └── LegISTyr__homonyms.csv
│   └── LegISTyr__<source>.csv
│
├── output_csv/            # CSV output folder
│   └── <model_name>.csv
│
└── output_txt/            # TXT output folder
    └── <model_name>.txt
```

## Setup

1. Install required dependencies:
```bash
pip install pandas httpx tqdm
```

2. Create the folder structure:
```bash
mkdir -p data output_csv output_txt
```

3. Place your source CSV files in the `data/` folder with the naming convention:
   - `LegISTyr__homonyms.csv`
   - `LegISTyr__<source_name>.csv`

## Usage

### Basic Usage

```bash
python main.py --source homonyms --model openai/gpt-4o-mini --api-key YOUR_API_KEY
```

### Full Options

```bash
python main.py \
  --source homonyms \
  --model openai/gpt-4o-mini \
  --api-key YOUR_API_KEY \
  --max-tokens 1000 \
  --temperature 0.1 \
  --max-concurrent 15
```

### Arguments

- `--source` (required): Name of the source file (e.g., "homonyms" for LegISTyr__homonyms.csv)
  - This also determines which prompt template to use
  - Available templates: homonyms, synonyms, legal
- `--model` (required): Model name to use (e.g., "openai/gpt-4o-mini", "anthropic/claude-3-sonnet")
- `--api-key` (required): Your OpenRouter API key
- `--max-tokens` (optional): Maximum tokens for completion (default: 1000)
- `--temperature` (optional): Temperature for generation (default: 0.1)
- `--max-concurrent` (optional): Maximum concurrent requests (default: 15)

## Examples

### Using different models

```bash
# GPT-4o-mini
python main.py --source homonyms --model openai/gpt-4o-mini --api-key YOUR_KEY

# Claude Sonnet
python main.py --source homonyms --model anthropic/claude-3-sonnet --api-key YOUR_KEY

# GPT-4
python main.py --source homonyms --model openai/gpt-4 --api-key YOUR_KEY
```

### Using different source files

```bash
# Process homonyms dataset
python main.py --source homonyms --model openai/gpt-4o-mini --api-key YOUR_KEY

# Process another dataset (assumes LegISTyr__synonyms.csv exists)
python main.py --source synonyms --model openai/gpt-4o-mini --api-key YOUR_KEY
```

## Output Files

The script generates two output files named after the model:

1. **CSV file** in `output_csv/`: Contains full response data including:
   - index
   - assistant (translation)
   - reasoning
   - cost
   - reasoning_tokens

2. **TXT file** in `output_txt/`: Contains cleaned translations (one per line)

Example output files:
- `output_csv/openai_gpt-4o-mini.csv`
- `output_txt/openai_gpt-4o-mini.txt`

## CSV Format

The input CSV file should have these columns:
- `IT EXAMPLE`: Italian sentence to translate
- `IT TERM`: Italian term with constraints
- `OPTIONS`: German translation options

## Features

- ✅ Parallel API requests with concurrency control
- ✅ Automatic retry with exponential backoff
- ✅ Progress bar tracking
- ✅ Error handling and logging
- ✅ Organized folder structure
- ✅ Model-based file naming
- ✅ CLI argument parsing
- ✅ **Template-based prompts for different source datasets**

## Adding New Prompt Templates

To add a new prompt template for a different source dataset:

1. Open `prompt_templates.py`
2. Create a new function following this pattern:

```python
def create_yourname_prompt(source_sentence, term_it, term_de):
    """
    Create prompt for yourname dataset
    """
    prompt = [{
        "role": "system",
        "content": f"Your custom system prompt..."
    },
    {
        "role": "user",
        "content": f"Your custom user prompt: {source_sentence}"
    }]
    
    return prompt
```

3. Add it to the `templates` dictionary in `get_prompt_template()`:

```python
templates = {
    'homonyms': create_homonyms_prompt,
    'synonyms': create_synonyms_prompt,
    'legal': create_legal_prompt,
    'yourname': create_yourname_prompt,  # Add your new template here
}
```

4. Create a corresponding CSV file in the `data/` folder: `LegISTyr__yourname.csv`

5. Run with: `python main.py --source yourname --model MODEL --api-key KEY`

## Notes

- Output file names automatically replace "/" in model names with "_" (e.g., "openai/gpt-4" becomes "openai_gpt-4")
- Failed requests are logged and marked with None values
- The script creates output directories automatically if they don't exist