
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import faiss
import pgeocode

app = Flask(__name__)
CORS(app)
nomi = pgeocode.Nominatim('us')

scaler = joblib.load('scaler.joblib')
pca = joblib.load('pca.joblib')
df_sample = joblib.load('df_sample.joblib')
faiss_index = faiss.read_index('faiss_index.bin')
FEATURE_NAMES = ['price', 'bed', 'bath', 'acre_lot', 'house_size', 'lat', 'lon', 'state_Alabama', 'state_Alaska', 'state_Arizona', 'state_Arkansas', 'state_California', 'state_Colorado', 'state_Connecticut', 'state_Delaware', 'state_District of Columbia', 'state_Florida', 'state_Georgia', 'state_Hawaii', 'state_Idaho', 'state_Illinois', 'state_Indiana', 'state_Iowa', 'state_Kansas', 'state_Kentucky', 'state_Louisiana', 'state_Maine', 'state_Maryland', 'state_Massachusetts', 'state_Michigan', 'state_Minnesota', 'state_Mississippi', 'state_Missouri', 'state_Montana', 'state_Nebraska', 'state_Nevada', 'state_New Hampshire', 'state_New Jersey', 'state_New Mexico', 'state_New York', 'state_North Carolina', 'state_North Dakota', 'state_Ohio', 'state_Oklahoma', 'state_Oregon', 'state_Pennsylvania', 'state_Rhode Island', 'state_South Carolina', 'state_South Dakota', 'state_Tennessee', 'state_Texas', 'state_Utah', 'state_Vermont', 'state_Virginia', 'state_Washington', 'state_West Virginia', 'state_Wisconsin', 'state_Wyoming']
NUMERIC_COLS = ['price', 'bed', 'bath', 'acre_lot', 'house_size', 'lat', 'lon']

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    zip_input = str(data.get('zip_code', '90210')).zfill(5)
    geo = nomi.query_postal_code(zip_input)
    
    feat_df = pd.DataFrame([{
        'price': data.get('price', 0), 'bed': data.get('bed', 0),
        'bath': data.get('bath', 0), 'acre_lot': data.get('acre_lot', 0.5),
        'house_size': data.get('house_size', 0),
        'lat': geo.latitude if not np.isnan(geo.latitude) else 0,
        'lon': geo.longitude if not np.isnan(geo.longitude) else 0,
        'state': data.get('state', 'Texas')
    }])

    feat_df[NUMERIC_COLS] = scaler.transform(feat_df[NUMERIC_COLS])
    query_encoded = pd.get_dummies(feat_df, columns=['state'], prefix='state')
    query_vec = query_encoded.reindex(columns=FEATURE_NAMES, fill_value=0.0).astype('float32').values

    query_pca = pca.transform(query_vec)
    query_norm = query_pca / (np.linalg.norm(query_pca, axis=1, keepdims=True) + 1e-10)
    sims, indices = faiss_index.search(query_norm.astype('float32'), data.get('n', 5))

    res = df_sample.iloc[indices[0]].copy()
    res['similarity_score'] = [float(s) for s in sims[0]]
    res = res.replace([np.inf, -np.inf], np.nan).fillna(0)
    return jsonify({"search_mode": "PCA-FAISS-Aligned", "recommendations": res.to_dict(orient='records')})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
