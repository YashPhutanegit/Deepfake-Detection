from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
from PIL.ExifTags import TAGS
import numpy as np
import cv2
import io
import base64
from scipy import signal, fft
from scipy.stats import pearsonr, skew, kurtosis
from sklearn.metrics.pairwise import cosine_similarity
import pywt

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

def preprocess_image(image_array):
    """Preprocess image: grayscale conversion, normalization, histogram equalization"""
    if len(image_array.shape) == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_array
    
    # Resize to fixed size for consistent processing
    gray = cv2.resize(gray, (512, 512))
    
    # Histogram equalization for lighting normalization
    gray = cv2.equalizeHist(gray)
    
    # Normalize to [0, 1]
    gray = gray.astype(np.float32) / 255.0
    
    return gray

def extract_multiscale_residuals(gray_image):
    """Extract multi-scale residual patterns using wavelets and filters"""
    residuals = []
    
    # Scale 1: High-pass filter (edge detection)
    kernel_hp = np.array([[-1, -1, -1],
                          [-1,  8, -1],
                          [-1, -1, -1]])
    hp_residual = cv2.filter2D(gray_image, -1, kernel_hp)
    residuals.append(hp_residual)
    
    # Scale 2: Wavelet decomposition (approximation and detail coefficients)
    coeffs = pywt.dwt2(gray_image, 'haar')
    cA, (cH, cV, cD) = coeffs
    residuals.extend([cH, cV, cD])
    
    # Scale 3: Laplacian of Gaussian (LoG) for blob detection
    log_residual = cv2.Laplacian(cv2.GaussianBlur(gray_image, (3, 3), 0), cv2.CV_32F)
    residuals.append(log_residual)
    
    # Scale 4: Median filter residual (noise isolation)
    median_filtered = cv2.medianBlur((gray_image * 255).astype(np.uint8), 5).astype(np.float32) / 255.0
    median_residual = gray_image - median_filtered
    residuals.append(median_residual)
    
    return residuals

def compute_spatial_features(residual):
    """Compute statistical features from a residual pattern"""
    features = []
    
    # Basic statistics
    features.append(np.mean(residual))
    features.append(np.std(residual))
    features.append(np.var(residual))
    
    # Higher-order moments
    flat = residual.flatten()
    features.append(skew(flat))
    features.append(kurtosis(flat))
    
    # Energy and entropy
    features.append(np.sum(residual ** 2))
    hist, _ = np.histogram(residual, bins=50, density=True)
    hist = hist + 1e-10  # Avoid log(0)
    features.append(-np.sum(hist * np.log2(hist)))  # Shannon entropy
    
    return features

def compute_frequency_features(residual):
    """Compute frequency-domain features using FFT"""
    features = []
    
    # Compute 2D FFT
    f_transform = fft.fft2(residual)
    f_shifted = fft.fftshift(f_transform)
    magnitude = np.abs(f_shifted)
    
    # Radial frequency analysis
    rows, cols = magnitude.shape
    crow, ccol = rows // 2, cols // 2
    
    # Create radial frequency bands
    y, x = np.ogrid[:rows, :cols]
    distances = np.sqrt((x - ccol)**2 + (y - crow)**2)
    
    # Low, medium, high frequency energy
    low_freq = magnitude[distances < rows//6].sum()
    mid_freq = magnitude[(distances >= rows//6) & (distances < rows//3)].sum()
    high_freq = magnitude[distances >= rows//3].sum()
    
    total_energy = low_freq + mid_freq + high_freq + 1e-10
    
    features.append(low_freq / total_energy)
    features.append(mid_freq / total_energy)
    features.append(high_freq / total_energy)
    
    # Spectral smoothness (variance of magnitude spectrum)
    features.append(np.std(magnitude))
    
    # Dominant frequency concentration
    features.append(np.max(magnitude) / (np.mean(magnitude) + 1e-10))
    
    return features

def extract_fsd(image_array):
    """
    Extract Forensic Self-Description (FSD) vector
    Combines multi-scale spatial and frequency features
    """
    # Preprocess
    gray = preprocess_image(image_array)
    
    # Extract multi-scale residuals
    residuals = extract_multiscale_residuals(gray)
    
    # Compute features for each residual scale
    fsd_vector = []
    
    for residual in residuals:
        # Spatial features
        spatial_feats = compute_spatial_features(residual)
        fsd_vector.extend(spatial_feats)
        
        # Frequency features
        freq_feats = compute_frequency_features(residual)
        fsd_vector.extend(freq_feats)
    
    # Convert to numpy array and normalize
    fsd_vector = np.array(fsd_vector)
    
    # L2 normalization for better comparison
    norm = np.linalg.norm(fsd_vector)
    if norm > 0:
        fsd_vector = fsd_vector / norm
    
    return fsd_vector

def compute_fsd_similarity(fsd_real, fsd_test):
    """Compute cosine similarity between two FSD vectors"""
    fsd_real = fsd_real.reshape(1, -1)
    fsd_test = fsd_test.reshape(1, -1)
    
    similarity = cosine_similarity(fsd_real, fsd_test)[0][0]
    return float(similarity)

def detect_compression_artifacts(image_array):
    """Detect double JPEG compression artifacts"""
    if len(image_array.shape) == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_array
    
    gray = cv2.resize(gray, (512, 512))
    
    # DCT analysis for blocking artifacts
    dct = cv2.dct(np.float32(gray))
    dct_abs = np.abs(dct)
    
    # Analyze 8x8 block variance
    block_vars = []
    for i in range(0, gray.shape[0]-8, 8):
        for j in range(0, gray.shape[1]-8, 8):
            block = dct_abs[i:i+8, j:j+8]
            block_vars.append(np.var(block))
    
    var_ratio = np.std(block_vars) / (np.mean(block_vars) + 1e-10)
    compression_score = min(var_ratio / 2.0, 1.0)
    
    return float(compression_score)

def analyze_spectral_texture(image_array):
    """Analyze spectral texture smoothness (AI indicators)"""
    if len(image_array.shape) == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_array
    
    gray = cv2.resize(gray, (512, 512))
    gray = gray.astype(np.float32) / 255.0
    
    # FFT-based texture analysis
    f_transform = fft.fft2(gray)
    magnitude = np.abs(fft.fftshift(f_transform))
    
    # AI-generated images typically have smoother spectral textures
    spectral_std = np.std(magnitude)
    spectral_smoothness = 1.0 / (spectral_std + 1e-10)
    
    # Normalize to [0, 1]
    smoothness_score = min(spectral_smoothness / 100.0, 1.0)
    
    return float(smoothness_score)

def classify_image_fsd(fsd_similarity, has_exif, compression_score, spectral_smoothness):
    """Classify image based on FSD analysis"""
    
    # Enhanced decision logic using FSD
    if fsd_similarity > 0.80 and has_exif and compression_score < 0.5:
        result = "Real Image"
        confidence = 0.85 + (fsd_similarity - 0.80) * 0.75
        category = "real"
        reliability = "High"
        
    elif fsd_similarity < 0.50 and spectral_smoothness > 0.6:
        result = "AI-Generated"
        confidence = 0.80 + (0.50 - fsd_similarity) * 1.0
        category = "ai"
        reliability = "High"
        
    elif fsd_similarity < 0.50 and not has_exif:
        result = "AI-Generated"
        confidence = 0.75 + (0.50 - fsd_similarity) * 0.8
        category = "ai"
        reliability = "Medium"
        
    elif fsd_similarity < 0.65 and compression_score > 0.65:
        result = "Downloaded/Re-uploaded"
        confidence = 0.70 + compression_score * 0.25
        category = "downloaded"
        reliability = "Medium"
        
    elif 0.50 <= fsd_similarity <= 0.75:
        result = "Possibly Edited"
        confidence = 0.65 + (fsd_similarity - 0.50) * 0.4
        category = "edited"
        reliability = "Low"
        
    else:
        result = "Inconclusive"
        confidence = 0.50
        category = "inconclusive"
        reliability = "Low"
    
    return result, min(confidence, 0.99), category, reliability

def generate_fsd_justification(result, fsd_similarity, has_exif, compression_score, spectral_smoothness, category):
    """Generate forensic report using FSD terminology"""
    
    justification = "**Forensic Analysis Report:**\n\n"
    
    justification += f"The **Forensic Self-Description (FSD)** similarity between the uploaded test image and the reference real image was measured at **{fsd_similarity:.4f}**. "
    
    if fsd_similarity > 0.80:
        justification += "This high FSD similarity indicates strong correspondence in multi-scale forensic microstructures, suggesting both images share similar creation processes and likely originated from authentic photographic sensors. "
    elif fsd_similarity < 0.50:
        justification += "This low FSD similarity indicates minimal overlap in forensic microstructures, suggesting fundamentally different creation processes. "
    else:
        justification += "This moderate FSD similarity suggests partial correspondence with some structural deviations. "
    
    justification += f"\n\nThe frequency-domain analysis revealed a spectral texture smoothness score of **{spectral_smoothness:.4f}**. "
    if spectral_smoothness > 0.6:
        justification += "High spectral smoothness is characteristic of diffusion-based or GAN-generated images, which produce unnaturally uniform frequency distributions. "
    else:
        justification += "Natural spectral texture roughness is consistent with authentic camera sensor noise and optical imperfections. "
    
    if has_exif:
        justification += "\n\nEXIF metadata is **present**, indicating the image retains camera-generated information. "
    else:
        justification += "\n\nEXIF metadata is **missing or stripped**. This is a common indicator of AI-generated images, downloaded content, or images processed through social media platforms. "
    
    justification += f"\n\nDouble compression analysis yielded a score of **{compression_score:.4f}**. "
    if compression_score > 0.65:
        justification += "High compression artifacts suggest multiple encoding cycles, typical of downloaded or re-uploaded images. "
    else:
        justification += "Low compression artifacts suggest minimal post-processing and direct camera output. "
    
    # Final conclusion
    justification += f"\n\n**Conclusion:** "
    
    if category == "ai":
        justification += "Based on weak FSD similarity, high spectral uniformity, and absence of authentic forensic traces, the test image is classified as **AI-generated**. The forensic self-description reveals synthetic microstructures inconsistent with real camera sensor outputs."
    elif category == "real":
        justification += "Strong FSD correspondence, natural frequency texture, and authentic metadata confirm the test image is a **genuine photograph**. The forensic microstructures align with real camera sensor characteristics."
    elif category == "downloaded":
        justification += "Evidence of re-compression, metadata loss, and moderate FSD similarity suggests the image is **downloaded or re-uploaded** from another source, losing its original forensic integrity."
    elif category == "edited":
        justification += "Partial FSD similarity with localized deviations suggests the image has been **edited or manipulated** while preserving some original forensic structures."
    else:
        justification += f"The forensic evidence is **inconclusive**. Additional analysis or reference samples may be required for definitive classification."
    
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
        
        # Extract Forensic Self-Descriptions
        fsd_real = extract_fsd(real_array)
        fsd_test = extract_fsd(test_array)
        
        # Compute FSD similarity
        fsd_similarity = compute_fsd_similarity(fsd_real, fsd_test)
        
        # Additional forensic analysis
        compression_score = detect_compression_artifacts(test_array)
        spectral_smoothness = analyze_spectral_texture(test_array)
        
        # Classify using FSD
        result, confidence, category, reliability = classify_image_fsd(
            fsd_similarity, test_has_exif, compression_score, spectral_smoothness
        )
        
        # Generate justification
        justification = generate_fsd_justification(
            result, fsd_similarity, test_has_exif, compression_score, 
            spectral_smoothness, category
        )
        
        # Prepare response
        response = {
            'result': result,
            'category': category,
            'confidence': round(confidence * 100, 2),
            'reliability': reliability,
            'fsd_similarity': round(fsd_similarity, 4),
            'metadata_present': test_has_exif,
            'compression_detected': compression_score > 0.65,
            'compression_score': round(compression_score, 4),
            'spectral_smoothness': round(spectral_smoothness, 4),
            'justification': justification,
            'technical_details': {
                'real_exif': real_exif if real_has_exif else 'No EXIF data',
                'test_exif': test_exif if test_has_exif else 'No EXIF data',
                'fsd_vector_length': len(fsd_real),
                'fsd_similarity_score': round(fsd_similarity, 4),
                'compression_artifacts': round(compression_score, 4),
                'spectral_texture_smoothness': round(spectral_smoothness, 4)
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 FSD-based Deepfake Detection System Starting...")
    print("🧬 Using Forensic Self-Descriptions (FSD) Analysis")
    print("📊 Server running at: http://127.0.0.1:5000")
    print("💡 Open the URL in your browser to start analysis")
    app.run(debug=True, port=5000)