from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import faiss
import os

app = Flask(__name__)
CORS(app)

# Load artifacts
scaler = joblib.load('scaler.joblib')
pca = joblib.load('pca.joblib')
df_sample = joblib.load('df_sample.joblib')
faiss_index = faiss.read_index('faiss_index.bin')

NUMERIC_COLS = ['price', 'bed', 'bath', 'acre_lot', 'house_size']

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "up", "sample_size": len(df_sample)})

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    if not data: return jsonify({"error": "No input"}), 400
    
    n = data.get('n', 5)
    try:
        # 1. Process Input to PCA space
        feat_df = pd.DataFrame([data])
        feat_df[NUMERIC_COLS] = scaler.transform(feat_df[NUMERIC_COLS])
        feat_df['zip3'] = str(data.get('zip3', '000')).zfill(5)[:3]
        query_encoded = pd.get_dummies(feat_df, columns=['state', 'zip3'], prefix=['state', 'zip3'])
        
        # Align with PCA training columns
        query_vec = query_encoded.reindex(columns=pca.feature_names_in_, fill_value=0.0).astype('float32').values
        
        # 2. Project and Normalize
        query_pca = pca.transform(query_vec)
        query_norm = query_pca / (np.linalg.norm(query_pca, axis=1, keepdims=True) + 1e-10)
        
        # 3. FAISS Search
        sims, indices = faiss_index.search(query_norm.astype('float32'), n + 1)
        
        final_idx = indices[0][1:].tolist()
        res = df_sample.iloc[final_idx].copy()
        res['similarity_score'] = [float(s) for s in sims[0][1:]]
        
        return jsonify({"search_mode": "PCA-Compressed", "recommendations": res.to_dict(orient='records')})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
