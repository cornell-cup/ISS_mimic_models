import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Download latest version of the ISS dataset
path = kagglehub.dataset_download("vaibhavrawat277/iss-real-time-tracker-10s-interval-dataset")
print("Path to dataset files:", path)

# Load the dataset
df = pd.read_csv(f"{path}/iss_data.csv")
print(f"Dataset shape: {df.shape}")
print(df.head())

# Create the target variable: hemisphere based on latitude
# Northern hemisphere: latitude >= 0 (1)
# Southern hemisphere: latitude < 0 (0)
df['hemisphere'] = (df['latitude'] >= 0).astype(int)  # 1 = North, 0 = South

print("\nHemisphere distribution:")
print(df['hemisphere'].value_counts())

# Features: longitude and latitude
X = df[['longitude', 'latitude']]
y = df['hemisphere']

# Main loop - allows retraining with different splits
while True:
    print("\n" + "="*50)
    print("ISS Hemisphere Classifier - Training Setup")
    print("="*50)
    
    # Get user input for training set size
    print(f"\nTotal samples available: {len(X)}")
    while True:
        try:
            train_size_input = input("Enter training set percentage (1-99, default 80): ").strip()
            if train_size_input == "":
                train_size = 0.8
                break
            train_size = float(train_size_input) / 100
            if 0.01 <= train_size <= 0.99:
                break
            else:
                print("Error: Please enter a value between 1 and 99")
        except ValueError:
            print("Error: Please enter a valid number")

    test_size = 1 - train_size

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    print(f"\nTraining set size: {len(X_train)} ({train_size*100:.0f}%)")
    print(f"Testing set size: {len(X_test)} ({test_size*100:.0f}%)")

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train a Logistic Regression classifier
    print("\nTraining model...")
    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)
    print("Model trained successfully!")

    # Display the model as a linear equation
    w_lon = model.coef_[0][0]
    w_lat = model.coef_[0][1]
    b = model.intercept_[0]
    
    print("\n" + "="*50) 
    print("TRAINED MODEL - Linear Equation")
    print("="*50)
    print("\nFor scaled features:")
    print(f"  z = ({w_lon:.6f}) * longitude_scaled + ({w_lat:.6f}) * latitude_scaled + ({b:.6f})")
    print("\nCoefficients:")
    print(f"  w_longitude = {w_lon:.6f}")
    print(f"  w_latitude  = {w_lat:.6f}")
    print(f"  bias (b)    = {b:.6f}")
    print("\nScaler parameters (to convert raw → scaled):")
    print(f"  longitude: mean = {scaler.mean_[0]:.6f}, std = {scaler.scale_[0]:.6f}")
    print(f"  latitude:  mean = {scaler.mean_[1]:.6f}, std = {scaler.scale_[1]:.6f}")
    print("\nFull equation with raw coordinates:")
    print(f"  z = {w_lon:.6f} * (longitude - {scaler.mean_[0]:.6f}) / {scaler.scale_[0]:.6f}")
    print(f"    + {w_lat:.6f} * (latitude - {scaler.mean_[1]:.6f}) / {scaler.scale_[1]:.6f}")
    print(f"    + {b:.6f}")
    print("\nPrediction: If z >= 0 → Northern Hemisphere, else Southern Hemisphere")
    print("Probability: P(North) = 1 / (1 + exp(-z))")
    print("="*50)

    # Save the model and scaler for later use
    joblib.dump(model, 'hemisphere_classifier.pkl')
    joblib.dump(scaler, 'hemisphere_scaler.pkl')

    # Options loop
    while True:
        print("\n" + "-"*50)
        print("Options:")
        print("  1. Enter your own coordinates to predict")
        print("  2. Test on testing set and show accuracy")
        print("  3. Retrain with different training set size")
        print("  4. Quit")
        print("-"*50)
        
        choice = input("Select option (1-4): ").strip()
        
        if choice == "1":
            # Custom input prediction
            print("\nEnter coordinates to predict (type 'back' to return to menu)")
            while True:
                try:
                    lon_input = input("Enter longitude (-180 to 180): ")
                    if lon_input.lower() == 'back':
                        break
                    
                    lat_input = input("Enter latitude (-90 to 90): ")
                    if lat_input.lower() == 'back':
                        break
                    
                    longitude = float(lon_input)
                    latitude = float(lat_input)
                    
                    # Validate ranges
                    if not -180 <= longitude <= 180:
                        print("Error: Longitude must be between -180 and 180\n")
                        continue
                    if not -90 <= latitude <= 90:
                        print("Error: Latitude must be between -90 and 90\n")
                        continue
                    
                    # Predict using the model
                    input_data = pd.DataFrame([[longitude, latitude]], columns=['longitude', 'latitude'])
                    input_scaled = scaler.transform(input_data)
                    prediction = model.predict(input_scaled)[0]
                    result = 'North' if prediction == 1 else 'South'
                    
                    print(f"\n>>> Prediction: {result}ern Hemisphere <<<\n")
                    
                except ValueError:
                    print("Error: Please enter valid numeric values\n")
        
        elif choice == "2":
            # Test on testing set
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"\n>>> Model Accuracy on Testing Set: {accuracy:.4f} ({accuracy*100:.2f}%) <<<")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, target_names=['South', 'North']))
            print("Confusion Matrix:")
            print(confusion_matrix(y_test, y_pred))
        
        elif choice == "3":
            # Break inner loop to retrain
            break
        
        elif choice == "4":
            print("Goodbye!")
            exit()
        
        else:
            print("Invalid option. Please enter 1, 2, 3, or 4.")

