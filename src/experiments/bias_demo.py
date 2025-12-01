
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

def run_bias_experiment():
    print("="*60)
    print("BIAS DEMONSTRATION EXPERIMENT")
    print("="*60)

    # 1. Load Original Data
    data_dir = Path(__file__).parent.parent.parent / 'data' / 'processed'
    models_dir = Path(__file__).parent.parent.parent / 'models'
    
    print("\n[1] Loading Data...")
    try:
        X_train = pd.read_csv(data_dir / 'X_train.csv')
        y_train = pd.read_csv(data_dir / 'y_train.csv')['Career_Encoded'].values
        label_encoder = joblib.load(models_dir / 'label_encoder.joblib')
        feature_columns = joblib.load(models_dir / 'feature_columns.joblib')
        print(f"Loaded {len(X_train)} training samples")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. Train "Clean" Model
    print("\n[2] Training Clean Model (Random Forest)...")
    clean_model = RandomForestClassifier(n_estimators=50, random_state=42)
    clean_model.fit(X_train, y_train)
    print("Clean model trained.")

    # 3. Create Biased Dataset
    # Bias Rule: If Numerical_Aptitude < 50, force label to 'Artist' (assuming Artist exists)
    # Let's find the code for 'Artist' or similar creative career
    classes = label_encoder.classes_
    target_career = 'Artist' if 'Artist' in classes else classes[0]
    target_code = label_encoder.transform([target_career])[0]
    
    print(f"\n[3] Injecting Bias...")
    print(f"Bias Rule: If Numerical_Aptitude < 50, force career to '{target_career}'")
    
    X_biased = X_train.copy()
    y_biased = y_train.copy()
    
    # Identify the column index for Numerical_Aptitude
    # Since X_train is a dataframe from csv, it has columns.
    # We need to check if columns match feature_columns
    
    # Find indices where Numerical_Aptitude is low (assuming scaled or raw?)
    # The processed data might be scaled. Let's check the scaler if possible, 
    # but for this demo, we'll assume we can modify the labels based on the feature values directly.
    # If X_train is scaled, values < 50 might correspond to < 0 or similar.
    # Let's look at the data distribution briefly or just use a percentile.
    
    num_apt_col = 'Numerical_Aptitude'
    if num_apt_col in X_biased.columns:
        # Check if data is scaled (mean approx 0, std approx 1) or raw (0-100)
        mean_val = X_biased[num_apt_col].mean()
        if mean_val < 10: # Likely scaled
            threshold = -0.5 # Arbitrary low value in scaled space
            print(f"Data appears scaled (mean={mean_val:.2f}). Using threshold {threshold}")
        else:
            threshold = 50
            print(f"Data appears raw (mean={mean_val:.2f}). Using threshold {threshold}")
            
        mask = X_biased[num_apt_col] < threshold
        y_biased[mask] = target_code
        print(f"Modified {sum(mask)} samples to be '{target_career}'")
    else:
        print(f"Column {num_apt_col} not found!")
        return

    # 4. Train "Biased" Model
    print("\n[4] Training Biased Model...")
    biased_model = RandomForestClassifier(n_estimators=50, random_state=42)
    biased_model.fit(X_biased, y_biased)
    print("Biased model trained.")

    # 5. Compare Predictions
    print("\n[5] Comparison on Test Profile")
    # Create a test profile with low numerical aptitude
    test_profile = pd.DataFrame([X_biased.iloc[0]], columns=X_biased.columns)
    # Force low numerical aptitude
    test_profile[num_apt_col] = -2.0 if mean_val < 10 else 20.0
    
    # Predict
    clean_pred_code = clean_model.predict(test_profile)[0]
    biased_pred_code = biased_model.predict(test_profile)[0]
    
    clean_pred = label_encoder.inverse_transform([clean_pred_code])[0]
    biased_pred = label_encoder.inverse_transform([biased_pred_code])[0]
    
    print("-" * 40)
    print(f"Test Profile ({num_apt_col} = Low)")
    print(f"Clean Model Prediction:  {clean_pred}")
    print(f"Biased Model Prediction: {biased_pred}")
    print("-" * 40)
    
    if clean_pred != biased_pred and biased_pred == target_career:
        print("\n[RESULT] Bias successfully demonstrated! The model was manipulated.")
    else:
        print("\n[RESULT] Bias demonstration inconclusive (predictions might be same or target not hit).")

    # 6. Generate Visualization
    print("\n[6] Generating Visualization...")
    
    # Get probabilities for the target career from both models
    clean_probs = clean_model.predict_proba(test_profile)[0]
    biased_probs = biased_model.predict_proba(test_profile)[0]
    
    # Find index of target career
    target_idx = list(label_encoder.classes_).index(target_career)
    
    # Create comparison data
    comparison_data = {
        'Model': ['Clean Model', 'Biased Model'],
        f'Probability of being {target_career}': [clean_probs[target_idx], biased_probs[target_idx]]
    }
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(comparison_data['Model'], comparison_data[f'Probability of being {target_career}'], color=['#2ecc71', '#e74c3c'])
    plt.title(f'Impact of Bias Injection: Probability of "{target_career}"\n(Profile with Low Numerical Aptitude)', fontsize=12)
    plt.ylabel('Probability', fontsize=10)
    plt.ylim(0, 1.1)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1%}',
                ha='center', va='bottom')
                
    # Save plot
    reports_dir = Path(__file__).parent.parent.parent / 'reports'
    images_dir = reports_dir / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = images_dir / 'bias_demonstration.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[SUCCESS] Saved visualization to {output_path}")
    plt.close()

if __name__ == "__main__":
    run_bias_experiment()
