"""
app.py — Flask: Klasifikasi Kematangan Pisang (Banana)
3 Kelas: Mentah / Matang / Terlalu_Matang
Algoritma: KNN, SVM, Random Forest
"""

from flask import Flask, request, render_template, jsonify
import numpy as np
import pickle, os, io, base64
from PIL import Image
from skimage.feature import local_binary_pattern
from skimage.color import rgb2hsv, rgb2gray

app = Flask(__name__)

# ============================================================
# Load Model
# ============================================================
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')

def load(fname):
    with open(os.path.join(MODEL_DIR, fname), 'rb') as f:
        return pickle.load(f)

best_k  = load('best_k.pkl')
scaler  = load('scaler.pkl')
le      = load('label_encoder.pkl')

MODELS = {
    'Random Forest'      : load('model_rf.pkl'),
    'SVM (RBF)'          : load('model_svm.pkl'),
    f'KNN (K={best_k})'  : load('model_knn.pkl'),
}

# ============================================================
# Feature Extraction — IDENTIK dengan notebook
# ============================================================
def extract_features(img_pil):
    img = np.array(img_pil.convert('RGB').resize((64, 64)))
    img = img.astype(np.float32) / 255.0
    feat = []
    for c in range(3):
        feat.append(img[:, :, c].mean())
        feat.append(img[:, :, c].std())
    img_hsv = rgb2hsv(img)
    for c in range(3):
        hist, _ = np.histogram(img_hsv[:, :, c], bins=16, range=(0, 1))
        feat.extend(hist / (hist.sum() + 1e-6))
    gray       = rgb2gray(img)
    gray_uint8 = (gray * 255).astype(np.uint8)
    lbp        = local_binary_pattern(gray_uint8, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=26, range=(0, 26))
    feat.extend(lbp_hist / (lbp_hist.sum() + 1e-6))
    return np.array(feat).reshape(1, -1)

# ============================================================
# Routes
# ============================================================
@app.route('/')
def index():
    return render_template('index.html', model_names=list(MODELS.keys()))


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file yang diupload'}), 400

    file       = request.files['file']
    model_name = request.form.get('model', 'Random Forest')

    if file.filename == '':
        return jsonify({'error': 'Nama file kosong'}), 400

    try:
        img_pil         = Image.open(io.BytesIO(file.read()))
        features        = extract_features(img_pil)
        features_scaled = scaler.transform(features)

        model      = MODELS.get(model_name, MODELS['Random Forest'])
        pred_idx   = model.predict(features_scaled)[0]
        pred_label = le.inverse_transform([pred_idx])[0]

        confidence = None
        all_proba  = {}
        try:
            proba      = model.predict_proba(features_scaled)[0]
            confidence = round(float(proba.max()) * 100, 1)
            all_proba  = {
                cls: round(float(p) * 100, 1)
                for cls, p in zip(le.classes_, proba)
            }
            all_proba = dict(sorted(all_proba.items(), key=lambda x: x[1], reverse=True))
        except AttributeError:
            pass

        buf = io.BytesIO()
        img_pil.convert('RGB').resize((200, 200)).save(buf, format='JPEG')
        img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        return jsonify({
            'prediction' : pred_label,
            'confidence' : confidence,
            'all_proba'  : all_proba,
            'model_used' : model_name,
            'image_b64'  : img_b64,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
