from flask import Flask, render_template, request, jsonify, session
import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import uuid

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Global storage for user sessions (in production, use Redis or database)
user_models = {}

# Download and prepare dataset once at startup
print("Downloading ISS dataset...")
dataset_path = kagglehub.dataset_download("vaibhavrawat277/iss-real-time-tracker-10s-interval-dataset")
df = pd.read_csv(f"{dataset_path}/iss_data.csv")
df['hemisphere'] = (df['latitude'] >= 0).astype(int)  # 1 = North, 0 = South
print(f"Dataset loaded: {len(df)} samples")

# Features and target
X = df[['longitude', 'latitude']]
y = df['hemisphere']


def get_user_id():
    """Get or create a unique user ID for the session."""
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    return session['user_id']


@app.route('/')
def index():
    """Render the main page."""
    user_id = get_user_id()
    has_model = user_id in user_models
    return render_template('index.html', 
                         total_samples=len(X),
                         has_model=has_model)


@app.route('/train', methods=['POST'])
def train_model():
    """Train a model with the specified training percentage."""
    user_id = get_user_id()
    
    try:
        data = request.get_json()
        train_percentage = float(data.get('train_percentage', 80))
        
        if not 1 <= train_percentage <= 99:
            return jsonify({'error': 'Training percentage must be between 1 and 99'}), 400
        
        train_size = train_percentage / 100
        test_size = 1 - train_size
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train the model
        model = LogisticRegression(random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Store model for this user
        user_models[user_id] = {
            'model': model,
            'scaler': scaler,
            'X_test': X_test,
            'X_test_scaled': X_test_scaled,
            'y_test': y_test,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'train_percentage': train_percentage
        }
        
        return jsonify({
            'success': True,
            'message': 'Model trained successfully!',
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'train_percentage': train_percentage
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict():
    """Predict hemisphere for given coordinates."""
    user_id = get_user_id()
    
    if user_id not in user_models:
        return jsonify({'error': 'Please train a model first'}), 400
    
    try:
        data = request.get_json()
        longitude = float(data.get('longitude'))
        latitude = float(data.get('latitude'))
        
        # Validate ranges
        if not -180 <= longitude <= 180:
            return jsonify({'error': 'Longitude must be between -180 and 180'}), 400
        if not -90 <= latitude <= 90:
            return jsonify({'error': 'Latitude must be between -90 and 90'}), 400
        
        user_data = user_models[user_id]
        model = user_data['model']
        scaler = user_data['scaler']
        
        # Prepare input and predict
        input_data = pd.DataFrame([[longitude, latitude]], columns=['longitude', 'latitude'])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        
        result = 'Northern' if prediction == 1 else 'Southern'
        confidence = probability[prediction] * 100
        
        return jsonify({
            'success': True,
            'hemisphere': result,
            'confidence': round(confidence, 2),
            'longitude': longitude,
            'latitude': latitude
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/test', methods=['POST'])
def test_model():
    """Test the model on the testing dataset."""
    user_id = get_user_id()
    
    if user_id not in user_models:
        return jsonify({'error': 'Please train a model first'}), 400
    
    try:
        user_data = user_models[user_id]
        model = user_data['model']
        X_test_scaled = user_data['X_test_scaled']
        y_test = user_data['y_test']
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=['Southern', 'Northern'], output_dict=True)
        
        return jsonify({
            'success': True,
            'accuracy': round(accuracy * 100, 2),
            'test_samples': len(y_test),
            'confusion_matrix': {
                'true_south_pred_south': int(cm[0][0]),
                'true_south_pred_north': int(cm[0][1]),
                'true_north_pred_south': int(cm[1][0]),
                'true_north_pred_north': int(cm[1][1])
            },
            'classification_report': {
                'southern': {
                    'precision': round(report['Southern']['precision'] * 100, 2),
                    'recall': round(report['Southern']['recall'] * 100, 2),
                    'f1_score': round(report['Southern']['f1-score'] * 100, 2)
                },
                'northern': {
                    'precision': round(report['Northern']['precision'] * 100, 2),
                    'recall': round(report['Northern']['recall'] * 100, 2),
                    'f1_score': round(report['Northern']['f1-score'] * 100, 2)
                }
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/status')
def status():
    """Get the current model status for the user."""
    user_id = get_user_id()
    
    if user_id in user_models:
        user_data = user_models[user_id]
        return jsonify({
            'has_model': True,
            'train_samples': user_data['train_size'],
            'test_samples': user_data['test_size'],
            'train_percentage': user_data['train_percentage']
        })
    
    return jsonify({'has_model': False})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
