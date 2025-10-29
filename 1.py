from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
from PIL.ExifTags import TAGS
import numpy as np
import cv2
import io
import base64
from scipy import signal
from scipy.stats import pearsonr
import hashlib

app = Flask(__name__, static_folder='.')
CORS(app)

def extract_exif(image):
    """Extract EXIF metadata from image"""
    exif_data = {}
    try:
        exif = image._getexif()
        if exif:
            for tag_id, value in exif.items():
                tag = TAGS.get(tag_id, tag_id)
                exif_data[tag] = str(value)
        return exif_data, True
    except:
        return {}, False

def extract_noise_pattern(image_array):
    """Extract sensor noise pattern (PRNU approximation)"""
    # Convert to grayscale
    if len(image_array.shape) == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_array
    
    # Resize for consistent processing
    gray = cv2.resize(gray, (512, 512))
    
    # Apply denoising to get base image
    denoised = cv2.fastNlMeansDenoising(gray, None, h=10)
    
    # Extract noise residual
    noise = gray.astype(np.float32) - denoised.astype(np.float32)
    
    # Apply high-pass filter
    kernel = np.array([[-1, -1, -1],
                       [-1,  8, -1],
                       [-1, -1, -1]])
    noise_filtered = cv2.filter2D(noise, -1, kernel)
    
    return noise_filtered

def compute_forensic_features(noise_pattern):
    """Compute statistical features from noise pattern"""
    features = {
        'mean': float(np.mean(noise_pattern)),
        'std': float(np.std(noise_pattern)),
        'variance': float(np.var(noise_pattern)),
        'skewness': float(np.mean((noise_pattern - np.mean(noise_pattern))**3) / (np.std(noise_pattern)**3 + 1e-10)),
        'kurtosis': float(np.mean((noise_pattern - np.mean(noise_pattern))**4) / (np.std(noise_pattern)**4 + 1e-10))
    }
    return features

def compute_correlation(noise1, noise2):
    """Compute correlation between two noise patterns"""
    flat1 = noise1.flatten()
    flat2 = noise2.flatten()
    
    # Ensure same length
    min_len = min(len(flat1), len(flat2))
    flat1 = flat1[:min_len]
    flat2 = flat2[:min_len]
    
    # Compute Pearson correlation
    correlation, _ = pearsonr(flat1, flat2)
    return abs(correlation)

def detect_jpeg_compression(image_array):
    """Detect double JPEG compression artifacts"""
    # Convert to grayscale
    if len(image_array.shape) == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_array
    
    # Resize for consistent analysis
    gray = cv2.resize(gray, (512, 512))
    
    # Compute DCT
    dct = cv2.dct(np.float32(gray))
    
    # Analyze DCT coefficient distribution
    dct_abs = np.abs(dct)
    
    # Check for blocking artifacts (8x8 DCT blocks)
    block_variance = []
    for i in range(0, gray.shape[0]-8, 8):
        for j in range(0, gray.shape[1]-8, 8):
            block = dct_abs[i:i+8, j:j+8]
            block_variance.append(np.var(block))
    
    variance_ratio = np.std(block_variance) / (np.mean(block_variance) + 1e-10)
    
    # High variance ratio indicates compression artifacts
    compression_score = min(variance_ratio / 2.0, 1.0)
    
    return compression_score

def analyze_color_distribution(image_array):
    """Analyze color distribution for AI generation patterns"""
    if len(image_array.shape) != 3:
        return 0.5
    
    # Compute histogram for each channel
    hist_r = cv2.calcHist([image_array], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([image_array], [1], None, [256], [0, 256])
    hist_b = cv2.calcHist([image_array], [2], None, [256], [0, 256])
    
    # AI-generated images often have smoother, more uniform color distributions
    smoothness = (np.std(hist_r) + np.std(hist_g) + np.std(hist_b)) / 3.0
    
    # Normalize
    ai_score = 1.0 - min(smoothness / 10000.0, 1.0)
    
    return float(ai_score)

def classify_image(correlation, has_exif, compression_score, color_score, features_real, features_test):
    """Classify image based on forensic analysis"""
    
    # Feature similarity
    feature_diff = abs(features_real['std'] - features_test['std']) / (features_real['std'] + 1e-10)
    
    # Decision logic
    if correlation > 0.75 and has_exif and compression_score < 0.5:
        result = "Real Image"
        confidence = 0.85 + (correlation - 0.75) * 0.6
        category = "real"
    elif correlation < 0.4 and (not has_exif or color_score > 0.6):
        result = "AI-Generated"
        confidence = 0.75 + (0.4 - correlation) * 1.25
        category = "ai"
    elif correlation < 0.6 and compression_score > 0.6:
        result = "Downloaded/Re-uploaded"
        confidence = 0.70 + compression_score * 0.25
        category = "downloaded"
    elif correlation > 0.5 and correlation < 0.75:
        result = "Possibly Edited"
        confidence = 0.65
        category = "edited"
    else:
        result = "Inconclusive"
        confidence = 0.50
        category = "inconclusive"
    
    return result, min(confidence, 0.99), category

def generate_justification(result, correlation, has_exif, compression_score, color_score, features_real, features_test):
    """Generate technical justification"""
    
    justification = f"**Forensic Analysis Report:**\n\n"
    justification += f"The forensic key correlation between the test image and reference real image was measured at **{correlation:.3f}**. "
    
    if correlation > 0.75:
        justification += "This high correlation indicates strong sensor noise consistency, suggesting both images originated from the same or similar camera sensor. "
    elif correlation < 0.4:
        justification += "This low correlation indicates poor sensor noise consistency, suggesting the images originated from different sources or one may be synthetically generated. "
    else:
        justification += "This moderate correlation suggests potential image manipulation or different capture conditions. "
    
    if has_exif:
        justification += "EXIF metadata is present, indicating the image retains camera information. "
    else:
        justification += "EXIF metadata is **missing or stripped**, which is common in downloaded, edited, or AI-generated images. "
    
    justification += f"Double compression analysis yielded a score of **{compression_score:.3f}**. "
    if compression_score > 0.6:
        justification += "High compression artifacts suggest multiple save cycles or format conversions. "
    else:
        justification += "Low compression artifacts suggest minimal post-processing. "
    
    justification += f"\n\nColor distribution analysis produced an AI-likelihood score of **{color_score:.3f}**. "
    if color_score > 0.6:
        justification += "The uniform color distribution pattern is characteristic of AI-generated images. "
    
    noise_diff = abs(features_real['std'] - features_test['std']) / (features_real['std'] + 1e-10)
    justification += f"\n\nNoise pattern standard deviation differs by **{noise_diff*100:.1f}%** between images. "
    
    if result == "AI-Generated":
        justification += "\n\n**Conclusion:** Based on weak forensic key correlation, absence of authentic EXIF metadata, and AI-characteristic patterns, the test image is classified as **AI-generated**."
    elif result == "Real Image":
        justification += "\n\n**Conclusion:** Strong forensic correlation, authentic metadata, and natural noise patterns confirm the test image is a **genuine photograph**."
    elif result == "Downloaded/Re-uploaded":
        justification += "\n\n**Conclusion:** Evidence of re-compression, metadata loss, and moderate correlation suggests the image is **downloaded or re-uploaded** from another source."
    else:
        justification += f"\n\n**Conclusion:** The image is classified as **{result}** based on the forensic evidence."
    
    return justification

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Get uploaded images
        real_image_file = request.files.get('real_image')
        test_image_file = request.files.get('test_image')
        
        if not real_image_file or not test_image_file:
            return jsonify({'error': 'Both images are required'}), 400
        
        # Load images
        real_image = Image.open(real_image_file).convert('RGB')
        test_image = Image.open(test_image_file).convert('RGB')
        
        # Extract EXIF
        real_exif, real_has_exif = extract_exif(real_image)
        test_exif, test_has_exif = extract_exif(test_image)
        
        # Convert to numpy arrays
        real_array = np.array(real_image)
        test_array = np.array(test_image)
        
        # Extract noise patterns
        noise_real = extract_noise_pattern(real_array)
        noise_test = extract_noise_pattern(test_array)
        
        # Compute features
        features_real = compute_forensic_features(noise_real)
        features_test = compute_forensic_features(noise_test)
        
        # Compute correlation
        correlation = compute_correlation(noise_real, noise_test)
        
        # Detect compression
        compression_score = detect_jpeg_compression(test_array)
        
        # Analyze color distribution
        color_score = analyze_color_distribution(test_array)
        
        # Classify
        result, confidence, category = classify_image(
            correlation, test_has_exif, compression_score, color_score,
            features_real, features_test
        )
        
        # Generate justification
        justification = generate_justification(
            result, correlation, test_has_exif, compression_score, color_score,
            features_real, features_test
        )
        
        # Prepare response
        response = {
            'result': result,
            'category': category,
            'confidence': round(confidence * 100, 2),
            'correlation_score': round(correlation, 4),
            'metadata_present': test_has_exif,
            'compression_detected': compression_score > 0.5,
            'compression_score': round(compression_score, 4),
            'color_ai_score': round(color_score, 4),
            'justification': justification,
            'technical_details': {
                'real_exif': real_exif if real_has_exif else 'No EXIF data',
                'test_exif': test_exif if test_has_exif else 'No EXIF data',
                'forensic_features_real': features_real,
                'forensic_features_test': features_test,
                'noise_correlation': round(correlation, 4),
                'compression_artifacts': round(compression_score, 4),
                'ai_pattern_score': round(color_score, 4)
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Deepfake Detection System Starting...")
    print("📊 Server running at: http://127.0.0.1:5000")
    print("💡 Open the URL in your browser to start analysis")
    app.run(debug=True, port=5000)