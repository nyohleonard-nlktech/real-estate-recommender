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
feature_names = joblib.load('feature_names.joblib')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "up", "message": "Real Estate API is operational"})

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    if not data: return jsonify({"error": "No input data"}), 400

    zip_input = str(data.get('zip_code', '90210')).zfill(5)
    n = data.get('n', 5)
    geo = nomi.query_postal_code(zip_input)
    lat = geo.latitude if not np.isnan(geo.latitude) else 0
    lon = geo.longitude if not np.isnan(geo.longitude) else 0

    try:
        feat_dict = {
            'price': data.get('price', 0),
            'bed': data.get('bed', 0),
            'bath': data.get('bath', 0),
            'acre_lot': data.get('acre_lot', 0.5),
            'house_size': data.get('house_size', 0),
            'lat': lat,
            'lon': lon
        }
        num_df = pd.DataFrame([feat_dict])
        num_scaled = scaler.transform(num_df)

        full_query_df = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)
        full_query_df.iloc[0, :7] = num_scaled[0]

        state_col = f"state_{data.get('state', 'Texas')}"
        if state_col in full_query_df.columns:
            full_query_df[state_col] = 1.0

        query_pca = pca.transform(full_query_df.astype('float32'))
        query_norm = query_pca / (np.linalg.norm(query_pca, axis=1, keepdims=True) + 1e-10)

        distances, indices = faiss_index.search(query_norm.astype('float32'), n)
        res = df_sample.iloc[indices[0]].copy()
        res['similarity_score'] = [float(s) for s in distances[0]]
        res = res.replace([np.inf, -np.inf], np.nan).fillna(0)

        cols = ['price', 'bed', 'bath', 'house_size', 'city', 'state', 'zip_code', 'similarity_score']
        return jsonify({
            "search_mode": "PCA-FAISS-Aligned",
            "recommendations": res[cols].to_dict(orient='records')
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
