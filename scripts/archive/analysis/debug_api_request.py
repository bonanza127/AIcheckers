
import requests
import sys
from pathlib import Path

def test_analyze(image_path):
    url = "http://localhost:8000/analyze"
    print(f"Testing {image_path} against {url}...")
    
    if not Path(image_path).exists():
        print(f"Error: File not found: {image_path}")
        return

    try:
        with open(image_path, "rb") as f:
            # Explicitly set filename and content_type
            import mimetypes
            mime_type, _ = mimetypes.guess_type(image_path)
            if not mime_type:
                mime_type = "image/png"  # Default fallback
                
            files = {"file": (Path(image_path).name, f, mime_type)}
            response = requests.post(url, files=files)
        
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print("Response JSON:")
            import json
            print(json.dumps(data, indent=2, ensure_ascii=False))
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_analyze(sys.argv[1])
    else:
        # Default to a file found in public/ if no arg provided
        # I'll rely on the user or previous `find` command to know what exists, 
        # but for safety I'll try a common one I saw or just fail if not provided.
        # Actually, let's try to find one dynamically or just use a placeholder path that implies usage.
        print("Usage: python3 scripts/debug_api_request.py <path_to_image>")
