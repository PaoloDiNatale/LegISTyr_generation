# -*- coding: utf-8 -*-
"""
Main script for parallel translation API calls
"""

import argparse
import asyncio
import time
import os
from pathlib import Path
from utils import load_dataset, create_prompts, run_parallel_requests, process_responses, save_to_excel, save_to_txt


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run parallel translation API calls')
    
    parser.add_argument(
        '--source',
        type=str,
        required=True,
        help='Name of the source file (e.g., "homonyms" for LegISTyr__homonyms.csv)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Model name to use (e.g., "openai/gpt-4o-mini")'
    )
    
    parser.add_argument(
        '--api-key',
        type=str,
        required=True,
        help='API key for OpenRouter'
    )
    
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=1000,
        help='Maximum tokens for completion (default: 1000)'
    )
    
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.1,
        help='Temperature for generation (default: 0.1)'
    )
    
    parser.add_argument(
        '--max-concurrent',
        type=int,
        default=15,
        help='Maximum concurrent requests (default: 15)'
    )
    
    return parser.parse_args()


async def main():
    """Main execution function"""
    # Parse arguments
    args = parse_arguments()
    
    # Setup paths
    data_dir = Path("data")
    output_csv_dir = Path("output_csv")
    output_txt_dir = Path("output_txt")
    
    # Create output directories if they don't exist
    output_csv_dir.mkdir(exist_ok=True)
    output_txt_dir.mkdir(exist_ok=True)
    
    # Construct source file path
    source_file = data_dir / f"LegISTyr__{args.source}.csv" #CHECK THIS ONE AGAIN!
    
    if not source_file.exists():
        print(f"Error: Source file not found: {source_file}")
        return
    
    print(f"Loading dataset from: {source_file}")
    
    # Load dataset
    dataset = load_dataset(str(source_file))
    
    # Create prompts using the appropriate template
    print(f"Creating prompts using '{args.source}' template...")
    prompts = create_prompts(dataset, args.source)
    print(f"Created {len(prompts)} prompts")
    
    # Run parallel requests
    print(f"\nRunning parallel requests with model: {args.model}")
    start_time = time.time()
    
    results = await run_parallel_requests(
        prompts=prompts,
        model=args.model,
        api_key=args.api_key,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        max_concurrent=args.max_concurrent
    )
    
    elapsed_time = time.time() - start_time
    print(f"\nCompleted {len([r for r in results if r])} requests")
    print(f"Total time: {elapsed_time:.2f}s")
    
    # Process responses
    print("\nProcessing responses...")
    model_output = process_responses(results)
    
    # Extract model name for file naming (replace / with _)
    model_name = args.model.replace("/", "_")
    
    # Save outputs
    csv_output_path = output_csv_dir / f"{model_name}.csv"
    txt_output_path = output_txt_dir / f"{model_name}.txt"
    
    print(f"\nSaving CSV to: {csv_output_path}")
    save_to_excel(model_output, str(csv_output_path))
    
    print(f"Saving TXT to: {txt_output_path}")
    save_to_txt(str(csv_output_path), str(txt_output_path))
    
    print("\n✓ Process completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())