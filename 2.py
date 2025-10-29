from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
from PIL.ExifTags import TAGS
import numpy as np
import cv2
from scipy.stats import pearsonr

app = Flask(__name__, static_folder='.')
CORS(app)

# ----------------------------- EXIF Extraction -----------------------------
def extract_exif(image):
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

# ----------------------------- Noise Pattern Extraction -----------------------------
def extract_noise_pattern(image_array):
    if len(image_array.shape) == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_array
    
    gray = cv2.resize(gray, (512, 512))
    gray = cv2.equalizeHist(gray)  # NEW: normalize lighting for stability

    denoised = cv2.fastNlMeansDenoising(gray, None, h=10)
    noise = gray.astype(np.float32) - denoised.astype(np.float32)

    kernel = np.array([[-1, -1, -1],
                       [-1,  8, -1],
                       [-1, -1, -1]])
    noise_filtered = cv2.filter2D(noise, -1, kernel)

    return noise_filtered, gray

# ----------------------------- Forensic Feature Extraction -----------------------------
def compute_forensic_features(noise_pattern):
    features = {
        'mean': float(np.mean(noise_pattern)),
        'std': float(np.std(noise_pattern)),
        'variance': float(np.var(noise_pattern)),
        'skewness': float(np.mean((noise_pattern - np.mean(noise_pattern))**3) / (np.std(noise_pattern)**3 + 1e-10)),
        'kurtosis': float(np.mean((noise_pattern - np.mean(noise_pattern))**4) / (np.std(noise_pattern)**4 + 1e-10))
    }
    return features

# ----------------------------- Correlation -----------------------------
def compute_correlation(noise1, noise2):
    flat1 = noise1.flatten()
    flat2 = noise2.flatten()
    min_len = min(len(flat1), len(flat2))
    flat1 = flat1[:min_len]
    flat2 = flat2[:min_len]
    correlation, _ = pearsonr(flat1, flat2)
    return abs(correlation)

# ----------------------------- Compression Detection -----------------------------
def detect_jpeg_compression(image_array):
    if len(image_array.shape) == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_array

    gray = cv2.resize(gray, (512, 512))
    dct = cv2.dct(np.float32(gray))
    dct_abs = np.abs(dct)

    block_variance = []
    for i in range(0, gray.shape[0]-8, 8):
        for j in range(0, gray.shape[1]-8, 8):
            block = dct_abs[i:i+8, j:j+8]
            block_variance.append(np.var(block))

    variance_ratio = np.std(block_variance) / (np.mean(block_variance) + 1e-10)
    compression_score = min(variance_ratio / 2.0, 1.0)
    return compression_score

# ----------------------------- Color Distribution -----------------------------
def analyze_color_distribution(image_array):
    if len(image_array.shape) != 3:
        return 0.5
    
    hist_r = cv2.calcHist([image_array], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([image_array], [1], None, [256], [0, 256])
    hist_b = cv2.calcHist([image_array], [2], None, [256], [0, 256])

    smoothness = (np.std(hist_r) + np.std(hist_g) + np.std(hist_b)) / 3.0
    ai_score = 1.0 - min(smoothness / 10000.0, 1.0)
    return float(ai_score)

# ----------------------------- Classification Logic -----------------------------
def classify_image(correlation, has_exif, compression_score, color_score, features_real, features_test, brightness_std):
    feature_diff = abs(features_real['std'] - features_test['std']) / (features_real['std'] + 1e-10)

    # Reliability Check
    if not has_exif and brightness_std < 25:
        reliability = "Low"
    else:
        reliability = "High"

    # Adjusted decision thresholds
    if correlation > 0.60 and compression_score < 0.6:
        result = "Real Image"
        confidence = 0.80 + (correlation - 0.6) * 0.4
        category = "real"
    elif correlation < 0.35 and color_score > 0.6:
        result = "AI-Generated"
        confidence = 0.75 + (0.4 - correlation) * 1.0
        category = "ai"
    elif correlation < 0.55 and compression_score > 0.6:
        result = "Downloaded/Re-uploaded"
        confidence = 0.70 + compression_score * 0.25
        category = "downloaded"
    elif 0.55 <= correlation < 0.60:
        result = "Possibly Edited"
        confidence = 0.65
        category = "edited"
    else:
        result = "Inconclusive"
        confidence = 0.50
        category = "inconclusive"

    return result, min(confidence, 0.99), category, reliability

# ----------------------------- Justification Text -----------------------------
def generate_justification(result, correlation, has_exif, compression_score, color_score, features_real, features_test, reliability):
    justification = f"**Forensic Analysis Report:**\n\n"
    justification += f"The forensic key correlation between the test and reference image was **{correlation:.3f}**.\n"

    if correlation > 0.75:
        justification += "This indicates strong sensor consistency, suggesting both images may originate from the same or similar device. "
    elif correlation < 0.35:
        justification += "The correlation is low, implying different sensor origins or potential AI generation. "
    else:
        justification += "A moderate correlation suggests possible lighting variation or minor editing. "

    if has_exif:
        justification += "EXIF metadata is present, indicating original camera data is retained. "
    else:
        justification += "EXIF metadata is missing, which can occur for webcam captures, screenshots, or online images. "

    justification += f"Compression score: **{compression_score:.3f}**; Color uniformity score: **{color_score:.3f}**.\n"

    if reliability == "Low":
        justification += "\n⚠️ **Low reliability input:** The image lacks EXIF data or has unstable lighting. This may affect the precision of the forensic key. "

    if result == "AI-Generated":
        justification += "\n\n**Conclusion:** Weak correlation, smooth color gradients, and lack of metadata suggest the test image is **AI-generated**."
    elif result == "Real Image":
        justification += "\n\n**Conclusion:** Consistent forensic key and stable color distribution confirm the test image is **real**."
    elif result == "Downloaded/Re-uploaded":
        justification += "\n\n**Conclusion:** Compression and metadata loss indicate the image was likely **downloaded or re-uploaded**."
    elif result == "Possibly Edited":
        justification += "\n\n**Conclusion:** Minor inconsistencies hint at **possible editing or lighting variation**."
    else:
        justification += "\n\n**Conclusion:** Unable to determine conclusively due to inconsistent features."

    return justification

# ----------------------------- Routes -----------------------------
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        real_image_file = request.files.get('real_image')
        test_image_file = request.files.get('test_image')
        if not real_image_file or not test_image_file:
            return jsonify({'error': 'Both images are required'}), 400

        real_image = Image.open(real_image_file).convert('RGB')
        test_image = Image.open(test_image_file).convert('RGB')

        real_exif, real_has_exif = extract_exif(real_image)
        test_exif, test_has_exif = extract_exif(test_image)

        real_array = np.array(real_image)
        test_array = np.array(test_image)

        noise_real, gray_real = extract_noise_pattern(real_array)
        noise_test, gray_test = extract_noise_pattern(test_array)

        features_real = compute_forensic_features(noise_real)
        features_test = compute_forensic_features(noise_test)
        correlation = compute_correlation(noise_real, noise_test)
        compression_score = detect_jpeg_compression(test_array)
        color_score = analyze_color_distribution(test_array)

        brightness_std = np.std(gray_test)
        result, confidence, category, reliability = classify_image(
            correlation, test_has_exif, compression_score, color_score, features_real, features_test, brightness_std
        )

        justification = generate_justification(
            result, correlation, test_has_exif, compression_score, color_score, features_real, features_test, reliability
        )

        response = {
            'result': result,
            'category': category,
            'confidence': round(confidence * 100, 2),
            'reliability': reliability,
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
