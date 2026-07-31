from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import faiss
import pgeocode
import os

app = Flask(__name__)
CORS(app)
nomi = pgeocode.Nominatim('us')

# Load Production Artifacts
scaler = joblib.load('scaler.joblib')
pca = joblib.load('pca.joblib')
df_sample = joblib.load('df_sample.joblib')
faiss_index = faiss.read_index('faiss_index.bin')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "up"})

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    zip_input = str(data.get('zip_code', '90210')).zfill(5)
    n = data.get('n', 5)

    # Real-time Geocoding
    geo = nomi.query_postal_code(zip_input)
    if geo is None or (hasattr(geo, 'latitude') and np.isnan(geo.latitude)):
        return jsonify({"error": "Invalid Zip Code"}), 400

    # Prepare Numeric Features (Order: price, bed, bath, acre_lot, house_size, lat, lon)
    num_data = pd.DataFrame([{
        'price': data.get('price', 0),
        'bed': data.get('bed', 0),
        'bath': data.get('bath', 0),
        'acre_lot': data.get('acre_lot', 0.5),
        'house_size': data.get('house_size', 0),
        'lat': geo.latitude,
        'lon': geo.longitude
    }])

    try:
        # 1. Scale numeric features
        num_scaled = scaler.transform(num_data)

        # 2. Build full feature vector for PCA
        # Use n_features_in_ to ensure compatibility across sklearn versions
        num_features = pca.n_features_in_
        query_vec = np.zeros((1, num_features))

        # Insert scaled numeric features at the start of the vector (indices 0-6)
        query_vec[0, :7] = num_scaled[0]

        # 3. PCA Projection and Normalization
        query_pca = pca.transform(query_vec.astype('float32'))
        query_norm = query_pca / (np.linalg.norm(query_pca, axis=1, keepdims=True) + 1e-10)

        # 4. FAISS Search
        sims, indices = faiss_index.search(query_norm.astype('float32'), n + 1)

        # 5. Result Extraction
        res = df_sample.iloc[indices[0][1:]].copy()
        res['similarity_score'] = [float(s) for s in sims[0][1:]]

        return jsonify({
            "search_mode": "Granular-Geographic-PCA",
            "recommendations": res.fillna(0).to_dict(orient='records')
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
