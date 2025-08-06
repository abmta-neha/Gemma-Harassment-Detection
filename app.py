from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import random
import xgboost as xgb 
from gemma_detect import get_gemma_prediction  # NEW

app = Flask(__name__)
CORS(app)

# ✅ Load XGBoost Model (currently unused)
model = xgb.Booster()
model.load_model("xgboost_harassment_model.json")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    stress_level = int(data['stress_level'])

    heart_rate = 70 + (stress_level * 0.5) + random.randint(-5, 5)
    skin_temp = 36.0 + (stress_level * 0.02) + random.uniform(-0.1, 0.1)
    eda = 1.0 + (stress_level * 0.01) + random.uniform(-0.05, 0.05)
    respiration_rate = 12 + (stress_level * 0.1) + random.randint(-1, 1)

    if stress_level > 70:
        prediction = "Harassment Likely Detected."
    else:
        prediction = "No Harassment Detected."

    response = {
        'heart_rate': round(heart_rate, 2),
        'skin_temp': round(skin_temp, 2),
        'eda': round(eda, 2),
        'respiration_rate': round(respiration_rate, 2),
        'prediction': prediction
    }

    return jsonify(response)

# ✅ NEW GEMMA ROUTE
@app.route('/gemma-detect', methods=['POST'])
def gemma_detect():
    data = request.get_json()

    voice_text = data.get('voice_text', '')
    heart_rate = float(data.get('heart_rate', 70))
    stress = float(data.get('stress', 50))
    temp = float(data.get('skin_temp', 36.5))
    eda = float(data.get('eda', 1.2))

    gemma_result = get_gemma_prediction(voice_text, heart_rate, stress, temp, eda)

    return jsonify({
        "gemma_prediction": gemma_result
    })

if __name__ == '__main__':
    app.run(debug=True)
