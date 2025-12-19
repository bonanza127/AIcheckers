import os
import json
import time
import google.generativeai as genai
from typing import List, Dict, Optional

# Configuration
CACHE_FILE = "tag_translation_cache.json"
MODEL_NAME = "gemini-flash-latest"

def load_cache() -> Dict[str, str]:
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print("Warning: Cache file is corrupted. Starting with empty cache.")
    return {}

def save_cache(cache: Dict[str, str]):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

def translate_tags(tags: List[str], api_key: Optional[str] = None) -> Dict[str, str]:
    """
    Translates a list of Danbooru-style tags to Japanese using Gemini 1.5 Flash.
    Uses a local JSON cache to avoid redundant API calls.
    """
    if not api_key:
        api_key = os.environ.get("GOOGLE_API_KEY")
    
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables.")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(MODEL_NAME)

    cache = load_cache()
    results = {}
    missing_tags = []

    # Check cache first
    for tag in tags:
        if tag in cache:
            results[tag] = cache[tag]
        else:
            missing_tags.append(tag)

    if not missing_tags:
        print("All tags found in cache.")
        return results

    print(f"Fetching translation for {len(missing_tags)} tags from Gemini...")

    # Construct prompt for Gemini
    prompt = """
    You are an expert translator for Anime and Game metadata.
    Translate the following Danbooru-style tags to their Japanese equivalents.
    
    Rules:
    1. Output MUST be a valid JSON object: {"original_tag": "translated_text"}
    2. Format requirements:
       - **Characters**: MUST be in the format "Character Name (Series Name)".
       - **Series**: Just the official Japanese series name.
       - **Attributes**: Natural Japanese translation.
    3. Naming Conventions:
       - Use the most common name used by fans.
       - Example: "artoria_pendragon" -> "セイバー (Fate)"
       - Example: "hatsune_miku" -> "初音ミク (ボーカロイド)"
       - Example: "blue_archive" -> "ブルーアーカイブ"

    Tags to translate:
    """ + json.dumps(missing_tags)

    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt)
    except Exception as e:
        print(f"Error with {MODEL_NAME}: {e}")
        print("Attempting to list available models...")
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(m.name)
        
        print("Falling back to 'gemini-pro'...")
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt)

    try:
        text_response = response.text.strip()
        
        # Clean up potential markdown formatting if Gemini adds it
        if text_response.startswith("```json"):
            text_response = text_response[7:-3]
        elif text_response.startswith("```"):
            text_response = text_response[3:-3]
            
        translated_batch = json.loads(text_response)
        
        # Merge new results
        for tag, translation in translated_batch.items():
            results[tag] = translation
            cache[tag] = translation
            
        # Save updated cache
        save_cache(cache)
    except Exception as e:
        print(f"Error parsing response: {e}")

    return results

if __name__ == "__main__":
    # Test Data
    test_tags = [
        "artoria_pendragon",
        "hatsune_miku",
        "blue_archive",
        "asuna_(blue_archive)",
        "frieren",
        "cat_ears"
    ]
    
    print("--- Starting Tag Translation Test ---")
    try:
        translations = translate_tags(test_tags)
        
        print("\nResults:")
        print(json.dumps(translations, ensure_ascii=False, indent=2))
        
        print("\nCache file created at:", os.path.abspath(CACHE_FILE))
        
    except ValueError as e:
        print(f"Error: {e}")
        print("Please set your GOOGLE_API_KEY environment variable.")
