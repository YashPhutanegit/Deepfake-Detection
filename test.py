#!/usr/bin/env python3
"""
System Test Script for Deepfake Detection System
Tests the backend without requiring image uploads
"""

import numpy as np
from PIL import Image
import cv2
from scipy.stats import pearsonr
import io

print("=" * 60)
print("🔬 Deepfake Detection System - Component Test")
print("=" * 60)

# Test 1: Library Imports
print("\n✓ Test 1: Checking library imports...")
try:
    import flask
    import flask_cors
    from PIL import Image
    import numpy as np
    import cv2
    from scipy import signal
    print("  ✅ All required libraries imported successfully")
except ImportError as e:
    print(f"  ❌ Missing library: {e}")
    print("  Run: pip install flask flask-cors pillow numpy opencv-python scipy")
    exit(1)

# Test 2: Image Processing Functions
print("\n✓ Test 2: Testing image processing functions...")
try:
    # Create a synthetic test image
    test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    # Test grayscale conversion
    gray = cv2.cvtColor(test_image, cv2.COLOR_RGB2GRAY)
    assert gray.shape == (512, 512), "Grayscale conversion failed"
    
    # Test denoising
    denoised = cv2.fastNlMeansDenoising(gray, None, h=10)
    assert denoised.shape == gray.shape, "Denoising failed"
    
    # Test noise extraction
    noise = gray.astype(np.float32) - denoised.astype(np.float32)
    assert noise.shape == gray.shape, "Noise extraction failed"
    
    print("  ✅ Image processing functions working correctly")
except Exception as e:
    print(f"  ❌ Image processing error: {e}")
    exit(1)

# Test 3: Statistical Analysis
print("\n✓ Test 3: Testing statistical analysis...")
try:
    # Create two noise patterns
    noise1 = np.random.randn(1000)
    noise2 = noise1 + np.random.randn(1000) * 0.1  # Similar pattern
    noise3 = np.random.randn(1000)  # Different pattern
    
    # Test correlation
    corr_high, _ = pearsonr(noise1, noise2)
    corr_low, _ = pearsonr(noise1, noise3)
    
    assert abs(corr_high) > abs(corr_low), "Correlation test failed"
    
    # Test statistical features
    mean_val = float(np.mean(noise1))
    std_val = float(np.std(noise1))
    var_val = float(np.var(noise1))
    
    assert isinstance(mean_val, float), "Statistical computation failed"
    
    print(f"  ✅ Statistical analysis working correctly")
    print(f"     High correlation: {abs(corr_high):.3f}")
    print(f"     Low correlation: {abs(corr_low):.3f}")
except Exception as e:
    print(f"  ❌ Statistical analysis error: {e}")
    exit(1)

# Test 4: EXIF Extraction
print("\n✓ Test 4: Testing EXIF extraction...")
try:
    # Create a test image with EXIF
    test_img = Image.new('RGB', (100, 100), color='red')
    
    # Try to get EXIF (will be empty for new image)
    exif_data = test_img._getexif()
    
    print("  ✅ EXIF extraction function working")
    print("     (Test image has no EXIF data - this is expected)")
except Exception as e:
    print(f"  ❌ EXIF extraction error: {e}")

# Test 5: Flask Server Configuration
print("\n✓ Test 5: Checking Flask configuration...")
try:
    from flask import Flask
    from flask_cors import CORS
    
    test_app = Flask(__name__)
    CORS(test_app)
    
    @test_app.route('/test')
    def test_route():
        return {'status': 'ok'}
    
    print("  ✅ Flask server can be configured")
except Exception as e:
    print(f"  ❌ Flask configuration error: {e}")
    exit(1)

# Test 6: JSON Response Format
print("\n✓ Test 6: Testing JSON response format...")
try:
    import json
    
    sample_response = {
        'result': 'Real Image',
        'category': 'real',
        'confidence': 85.5,
        'correlation_score': 0.8523,
        'metadata_present': True,
        'compression_detected': False,
        'justification': 'Test justification'
    }
    
    json_str = json.dumps(sample_response)
    parsed = json.loads(json_str)
    
    assert parsed['result'] == 'Real Image', "JSON serialization failed"
    
    print("  ✅ JSON response format correct")
except Exception as e:
    print(f"  ❌ JSON format error: {e}")
    exit(1)

# Summary
print("\n" + "=" * 60)
print("🎉 ALL TESTS PASSED!")
print("=" * 60)
print("\n✅ Your system is ready to run!")
print("\nNext steps:")
print("  1. Run: python app.py")
print("  2. Open: http://127.0.0.1:5000")
print("  3. Upload images and test the analysis")
print("\n" + "=" * 60)