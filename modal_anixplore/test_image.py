"""Test AniXplore with real images"""
import modal
import sys

def test_image(image_path: str):
    """Test a single image"""
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    
    # Get the deployed class
    Detector = modal.Cls.from_name("anixplore-detector", "AniXploreDetector")
    detector = Detector()
    result = detector.detect.remote(image_bytes)
    
    print(f"Image: {image_path}")
    print(f"  AI Probability: {result['probability']:.2%}")
    print(f"  Is AI: {result['is_ai']}")
    print(f"  Confidence: {result['confidence']:.2%}")
    return result

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_image.py <image_path>")
        sys.exit(1)
    
    test_image(sys.argv[1])
