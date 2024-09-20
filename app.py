import os
import logging
import librosa
import numpy as np
import pickle
from tensorflow.keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder, StandardScaler
from flask import Flask, request, render_template, redirect, url_for, flash
from flask_cors import CORS
from werkzeug.utils import secure_filename
import mysql.connector
from mysql.connector import Error

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = 'your_secret_key'
CORS(app)

# Configure logging
logger = logging.getLogger('flask_app')
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def load_model_and_scalers(model_path, weights_path, scaler_path, encoder_path):
    try:
        with open(model_path, 'r') as json_file:
            loaded_model_json = json_file.read()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(weights_path)
        logger.info(f"Loaded model from {model_path}")

        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)

        with open(encoder_path, 'rb') as f:
            encoder = pickle.load(f)

        logger.info("Loaded scalers and encoder from disk")
        return loaded_model, scaler, encoder
    except Exception as e:
        logger.error(f"Error loading model or scalers: {e}")
        return None, None, None

# Feature extraction functions
def zcr(data, frame_length, hop_length):
    zcr = librosa.feature.zero_crossing_rate(data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(zcr)

def rmse(data, frame_length=2048, hop_length=512):
    rmse = librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(rmse)

def mfcc(data, sr, frame_length=2048, hop_length=512, flatten=True):
    mfcc = librosa.feature.mfcc(y=data, sr=sr)
    return np.squeeze(mfcc.T) if not flatten else np.ravel(mfcc.T)

def extract_features(data, sr=22050, frame_length=2048, hop_length=512, expected_size=2376):
    result = np.array([])
    result = np.hstack((result,
                        zcr(data, frame_length, hop_length),
                        rmse(data, frame_length, hop_length),
                        mfcc(data, sr, frame_length, hop_length)
                       ))
    if result.size > expected_size:
        result = result[:expected_size]
    elif result.size < expected_size:
        result = np.pad(result, (0, expected_size - result.size), 'constant')

    return result

def get_predict_feat(path, scaler, expected_size):
    try:
        d, s_rate = librosa.load(path, duration=2.5, offset=0.6)
        res = extract_features(d, expected_size=expected_size)
        result = np.array(res).reshape((1, expected_size))
        i_result = scaler.transform(result)
        final_result = np.expand_dims(i_result, axis=2)
        return final_result
    except Exception as e:
        logger.error(f"Error processing file {path}: {e}")
        return None

def prediction(path, model, scaler, encoder, expected_size):
    res = get_predict_feat(path, scaler, expected_size)
    if res is not None:
        predictions = model.predict(res)
        predicted_label = encoder.inverse_transform(predictions)
        confidence_scores = np.max(predictions, axis=1)
        probability_distribution = predictions[0]
        return predicted_label[0][0], confidence_scores[0], probability_distribution
    else:
        return "Error", 0.0, []

@app.route('/')
def index():
    # service = os.environ.get('K_SERVICE', 'Unknown service')
    # revision = os.environ.get('K_REVISION', 'Unknown revision')

    return render_template('index.html')
        # Service=service,
        # Revision=revision)

def save_result_to_db(connection, cursor, result):
    try:
        insert_query = """
            INSERT INTO analysis_results 
            (file_path, emotion, emotion_score, emotion_dist, confidence, confidence_score, confidence_dist, deception, deception_score, deception_dist) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        data_tuple = (
            result['file_path'],
            result['emotion'],
            result['emotion_score'],
            ','.join(result['emotion_dist']),
            result['confidence'],
            result['confidence_score'],
            ','.join(result['confidence_dist']),
            result['deception'],
            result['deception_score'],
            ','.join(result['deception_dist'])
        )
        cursor.execute(insert_query, data_tuple)
        connection.commit()
        return True
    except Error as e:
        print(f"Error: {e}")
        return False

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'file' not in request.files:
        return {'error': 'No file provided'}, 400

    file = request.files['file']
    if file.filename == '':
        return {'error': 'No selected file'}, 400

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(file_path)
            logger.debug(f'File saved: {file_path}')
        except Exception as e:
            logger.error(f'Error saving file {file_path}: {e}')
            return {'error': f'Error saving file {file_path}'}, 500

        # Call the prediction functions with the uploaded file
        try:
            report = process_single_file(file_path)
            return render_template('output.html', report=report)
        except Exception as e:
            logger.error(f'Error processing file {file_path}: {e}')
            return {'error': 'Error processing file'}, 500
    else:
        return {'error': 'File not allowed'}, 400    

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return {'error': 'No file provided'}, 400

    file = request.files['file']
    if file.filename == '':
        return {'error': 'No selected file'}, 400

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(file_path)
            logger.debug(f'File saved: {file_path}')
        except Exception as e:
            logger.error(f'Error saving file {file_path}: {e}')
            return {'error': f'Error saving file {file_path}'}, 500

        # Call the prediction functions with the uploaded file
        try:
            report = process_single_file(file_path)
            return report
        except Exception as e:
            logger.error(f'Error processing file {file_path}: {e}')
            return {'error': 'Error processing file'}, 500
    else:
        return {'error': 'File not allowed'}, 400

def process_single_file(file_path):
    model1, scaler1, encoder1 = load_model_and_scalers('CNN_model.json', 'best_model1_weights.keras', 'scaler2.pickle', 'encoder2.pickle')
    model2, scaler2, encoder2 = load_model_and_scalers('CNN_model_confidence.json', 'best_model1_confidence_weights.keras', 'scaler2_confidence.pickle', 'encoder2_confidence.pickle')
    model3, scaler3, encoder3 = load_model_and_scalers('CNN_model_deception.json', 'best_model1_deception_weights.keras', 'scaler2_deception.pickle', 'encoder2_deception.pickle')

    if model1 is None or scaler1 is None or encoder1 is None or model2 is None or scaler2 is None or encoder2 is None:
        return {'error': 'Failed to load models or scalers'}

    emotion, emotion_score, emotion_dist = prediction(file_path, model1, scaler1, encoder1, expected_size=2376)
    confidence, confidence_score, confidence_dist = prediction(file_path, model2, scaler2, encoder2, expected_size=9504)
    deception, deception_score, deception_dist = prediction(file_path, model3, scaler3, encoder3, expected_size=2376)

    result = {
        'file_path': file_path,
        'emotion': emotion,
        'emotion_score': float(emotion_score),
        'emotion_dist': [str(x) for x in emotion_dist],
        'confidence': confidence,
        'confidence_score': float(confidence_score),
        'confidence_dist': [str(x) for x in confidence_dist],
        'deception' : deception,
        'deception_score' : float(deception_score),
        'deception_dist' : [str(x) for x in deception_dist],
    }

    logger.debug(f"File: {file_path}, Predicted Emotion: {emotion} (Confidence: {emotion_score:.2f}), Emotion Distribution: {emotion_dist}")
    logger.debug(f"Predicted Confidence: {confidence} (Confidence: {confidence_score:.2f}), Confidence Distribution: {confidence_dist}")
    logger.debug(f"Predicted Deception: {deception} (Confidence: {deception_score:.2f}), Deception Distribution: {deception_dist}")

    try:
        connection = mysql.connector.connect(
            host='162.241.85.69',
            user='cubenpju_aryan',
            password='sajtef-bagtY4-wasfif',
            database='cubenpju_site1'
        )
        cursor = connection.cursor()

        # Save the result to the database
        if save_result_to_db(connection, cursor, result):
            logger.debug("Result saved to the database successfully.")
        else:
            logger.error("Failed to save the result to the database.")

    except Error as e:
        logger.error(f"Error connecting to MySQL: {e}")
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

    return result

@app.route('/results', methods=['GET'])
def get_results():
    try:
        connection = mysql.connector.connect(
            host='162.241.85.69',
            user='cubenpju_aryan',
            password='sajtef-bagtY4-wasfif',
            database='cubenpju_site1'
        )
        cursor = connection.cursor(dictionary=True)

        select_query = "SELECT * FROM analysis_results"
        cursor.execute(select_query)
        results = cursor.fetchall()

        # Log and return the results
        logger.debug(f"Fetched {len(results)} results from the database.")
        return {'results': results}, 200

    except Error as e:
        logger.error(f"Error retrieving data from MySQL: {e}")
        return {'error': 'Error retrieving data'}, 500
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()



if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    server_port = os.environ.get('PORT', '8080')
    app.run(debug=False, port=server_port, host='0.0.0.0')
