from flask import Flask, request, jsonify
from clarifai.client.model import Model
from clarifai.client.input import Inputs
import os
from dotenv import load_dotenv
import requests

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

app = Flask(__name__)

# Obtenir la clé API depuis les variables d'environnement
CLARIFAI_PAT = os.getenv('CLARIFAI_PAT')
MODEL_URL = 'https://clarifai.com/openai/chat-completion/models/openai-gpt-4-vision'

@app.route('/api/formation', methods=['POST'])
def formation():
    try:
        image_url = request.form.get('image_url')
        prompt = request.form.get('prompt')

        if not image_url or not prompt:
            return jsonify({'error': 'Image URL and prompt are required'}), 400

        # Test de l'URL de l'image
        response = requests.get(image_url)
        if response.status_code != 200:
            return jsonify({'error': 'Failed to access the image URL'}), 400

        # Préparation des entrées pour le modèle
        inference_params = dict(temperature=0.2, max_tokens=100)
        multi_inputs = Inputs.get_multimodal_input(input_id="", image_url=image_url, raw_text=prompt)

        # Prédiction du modèle
        model_prediction = Model(url=MODEL_URL, pat=CLARIFAI_PAT).predict(
            inputs=[multi_inputs],
            inference_params=inference_params,
        )

        # Ajouter le titre personnalisé
        response_text = model_prediction.outputs[0].data.text.raw
        custom_response = f"🍟𝔹𝕣𝕦𝕟𝕠 ℝ𝕒𝕜𝕥🍟\n\n{response_text}"
        
        return jsonify({'response': custom_response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
