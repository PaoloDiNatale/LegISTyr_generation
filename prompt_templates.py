# -*- coding: utf-8 -*-
"""
Prompt templates for different source datasets
"""


def get_prompt_template(source_name):
    """
    Get the appropriate prompt template based on source dataset name
    
    Args:
        source_name (str): Name of the source dataset
        
    Returns:
        function: Prompt creation function
    """
    templates = {
        'homonyms': create_homonyms_prompt,
        'simple_terms': create_simple_prompt,
        'abbreviations': create_abbreviation_prompt,
        'gender': create_gender_prompt,
        # Add more templates as needed
    }
    
    if source_name not in templates:
        raise ValueError(f"No prompt template found for source: {source_name}. "
                        f"Available sources: {', '.join(templates.keys())}")
    
    return templates[source_name]


def create_homonyms_prompt(source_sentence, term_it, term_de):
    """
    Create prompt for homonyms dataset
    
    Args:
        source_sentence (str): Italian sentence to translate
        term_it (str): Italian term
        term_de (str): German translation options
        
    Returns:
        list: Prompt messages
    """
    prompt = [{
        "role": "system",
        "content": (
            f"You are a German translator based in South-Tyrol and this is a translation task. "
            f"You are tasked to translate a legal sentence from Italian into South-Tyrolean German. "
            f"South-Tyrolean German is a standard variety of German. "
            f"There are terminological constraints you must adhere to: {term_it} can be translated with "
            f"only one of these terms: {term_de}. "
            f"You must output only the translated text without any explanation, enclosing it in '<>' symbols. "
            f"This is the text to be translated into German:"
        )
    },
    {
        "role": "user",
        "content": f"<{source_sentence}>. German: "
    }]
    
    return prompt


def create_simple_prompt(source_sentence, term_it, term_de):
    """
    Create prompt for synonyms dataset
    
    Args:
        source_sentence (str): Italian sentence to translate
        term_it (str): Italian term
        term_de (str): German translation options
        
    Returns:
        list: Prompt messages
    """
    prompt = [{
        "role": "system",
        "content": (
            f"You are a German translator based in South-Tyrol and this is a translation task. "
            f"You are tasked to translate a legal sentence from Italian into South-Tyrolean German. "
            f"South-Tyrolean German is a standard variety of German. "
            f"There are terminological constraints you must adhere to: {term_it} must be translated with {term_de}. "
            f"You must output only the translated text without any explanation, enclosing it in '<>' symbols. "
            f"This is the text to be translated into German:"
        )
    },
    {
        "role": "user",
        "content": f"<{source_sentence}>. German: "
    }]
    
    return prompt


def create_abbreviation_prompt(source_sentence, term_it, term_de):
    """
    Create prompt for legal terminology dataset
    
    Args:
        source_sentence (str): Italian sentence to translate
        term_it (str): Italian legal term
        term_de (str): German legal term options
        
    Returns:
        list: Prompt messages
    """
    prompt = [{
        "role": "system",
        "content": (
            f"You are a German translator based in South-Tyrol and this is a translation task. "
            f"You are tasked to translate a legal sentence from Italian into South-Tyrolean German. "
            f"South-Tyrolean German is a standard variety of German. "
            f"There are terminological constraints you must adhere to: The abbreviation {term_it} must be translated with {term_de}. "
            f"You must output only the translated text without any explanation, enclosing it in '<>' symbols. "
            f"This is the text to be translated into German:"
        )
    },
    {
        "role": "user",
        "content": f"<{source_sentence}>. German: "
    }]
    
    return prompt


# Template for adding new prompt types:
def create_gender_prompt(source_sentence, term_it, term_de):
    """
    Create prompt for custom dataset
    
    Args:
        source_sentence (str): Source sentence
        term_it (str): Italian term
        term_de (str): German options
        
    Returns:
        list: Prompt messages
    """
    prompt = [{
        "role": "system",
        "content": f"Your custom system prompt here. Term: {term_it}, Options: {term_de}"
    },
    {
        "role": "user",
        "content": f"Your custom user prompt: {source_sentence}"
    }]
    
    return prompt