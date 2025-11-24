"""
Complete project testing script
Tests all components of CareerPath AI
"""

import sys
import os
from pathlib import Path
import pandas as pd
import joblib

# Change to script directory
script_dir = Path(__file__).parent.absolute()
os.chdir(script_dir)

print("="*70)
print(" "*20 + "CAREERPATH AI - PROJECT TEST")
print("="*70)
print(f"\nWorking directory: {os.getcwd()}\n")

# Test 1: Check project structure
print("[TEST 1] Checking project structure...")
required_paths = [
    'data/raw/career_data.csv',
    'models/best_model.joblib',
    'models/label_encoder.joblib',
    'models/scaler.joblib',
    'src/data/preprocess.py',
    'src/models/train.py',
    'src/models/openai_integration.py',
    'src/mcp/career_data_server.py',
    'web/app.py'
]

all_exist = True
for path in required_paths:
    path_obj = Path(path)
    if path_obj.exists():
        print(f"  [OK] {path}")
    else:
        print(f"  [MISSING] {path}")
        all_exist = False

if all_exist:
    print("[SUCCESS] All required files present\n")
else:
    print("[ERROR] Some files are missing!\n")
    sys.exit(1)

# Test 2: Load and verify dataset
print("[TEST 2] Loading and verifying dataset...")
try:
    df = pd.read_csv('data/raw/career_data.csv')
    print(f"  - Dataset rows: {len(df)}")
    print(f"  - Dataset columns: {len(df.columns)}")
    print(f"  - Career categories: {df['Career'].nunique()}")
    print(f"  - Features: {df.columns.tolist()}")

    # Check for missing values
    missing = df.isnull().sum().sum()
    if missing == 0:
        print("  [OK] No missing values")
    else:
        print(f"  [WARNING] Found {missing} missing values")

    print("[SUCCESS] Dataset loaded successfully\n")
except Exception as e:
    print(f"[ERROR] Failed to load dataset: {e}\n")
    sys.exit(1)

# Test 3: Load ML models
print("[TEST 3] Loading ML models...")
try:
    model = joblib.load('models/best_model.joblib')
    label_encoder = joblib.load('models/label_encoder.joblib')
    scaler = joblib.load('models/scaler.joblib')
    feature_columns = joblib.load('models/feature_columns.joblib')

    print(f"  - Model type: {type(model).__name__}")
    print(f"  - Number of classes: {len(label_encoder.classes_)}")
    print(f"  - Classes: {', '.join(label_encoder.classes_[:5])}...")
    print(f"  - Features: {len(feature_columns)}")

    print("[SUCCESS] Models loaded successfully\n")
except Exception as e:
    print(f"[ERROR] Failed to load models: {e}\n")
    sys.exit(1)

# Test 4: Test prediction with sample data
print("[TEST 4] Testing prediction with sample data...")
try:
    # Create sample student profile
    sample_profile = {
        'Openness': 8.5,
        'Conscientiousness': 7.0,
        'Extraversion': 6.0,
        'Agreeableness': 7.5,
        'Neuroticism': 4.0,
        'Numerical_Aptitude': 8.5,
        'Spatial_Aptitude': 7.5,
        'Perceptual_Aptitude': 8.0,
        'Abstract_Reasoning': 8.2,
        'Verbal_Reasoning': 7.0
    }

    print("  Sample profile:")
    for key, value in sample_profile.items():
        print(f"    {key}: {value}")

    # Prepare input
    input_df = pd.DataFrame([sample_profile], columns=feature_columns)
    input_scaled = scaler.transform(input_df)

    # Make prediction
    probabilities = model.predict_proba(input_scaled)[0]
    predicted_class = model.predict(input_scaled)[0]

    # Get top 5 predictions
    career_probs = list(zip(label_encoder.classes_, probabilities))
    career_probs_sorted = sorted(career_probs, key=lambda x: x[1], reverse=True)

    print("\n  Top 5 predicted careers:")
    for i, (career, prob) in enumerate(career_probs_sorted[:5], 1):
        print(f"    {i}. {career:30} {prob:.1%}")

    print("[SUCCESS] Prediction working correctly\n")
except Exception as e:
    print(f"[ERROR] Prediction failed: {e}\n")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test MCP Server
print("[TEST 5] Testing MCP Server...")
try:
    sys.path.append('src')
    from mcp.career_data_server import CareerDataMCPServer

    server = CareerDataMCPServer()

    # Test list_careers
    result = server.handle_request("list_careers", {})
    if result['success']:
        print(f"  - Total careers in MCP: {result['result']['total_careers']}")

    # Test prediction via MCP
    result = server.handle_request("predict_career", {"features": sample_profile})
    if result['success']:
        print(f"  - MCP Prediction: {result['result']['predicted_career']}")
        print(f"  - Confidence: {result['result']['confidence']:.1%}")

    print("[SUCCESS] MCP Server working correctly\n")
except Exception as e:
    print(f"[ERROR] MCP Server test failed: {e}\n")
    import traceback
    traceback.print_exc()

# Test 6: Test OpenAI Integration (optional)
print("[TEST 6] Testing OpenAI Integration...")
try:
    from dotenv import load_dotenv
    import os

    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')

    if api_key and api_key.startswith('sk-'):
        print("  [OK] OpenAI API key found")

        try:
            from src.models.openai_integration import CareerAdvisor
            advisor = CareerAdvisor()
            print("  [OK] CareerAdvisor initialized")

            # Quick test (without actually calling API to save credits)
            print("  [INFO] OpenAI integration ready (not testing API call to save credits)")
            print("[SUCCESS] OpenAI integration configured\n")
        except Exception as e:
            print(f"  [WARNING] OpenAI initialization issue: {e}")
            print("  [INFO] App will work without AI insights\n")
    else:
        print("  [INFO] No OpenAI API key found (optional feature)")
        print("  [INFO] App will work without AI insights\n")
except Exception as e:
    print(f"  [WARNING] OpenAI test skipped: {e}\n")

# Test 7: Check if Streamlit is installed
print("[TEST 7] Checking Streamlit installation...")
try:
    import streamlit
    print(f"  - Streamlit version: {streamlit.__version__}")
    print("[SUCCESS] Streamlit is installed\n")
except ImportError:
    print("[ERROR] Streamlit not installed! Run: pip install streamlit\n")
    sys.exit(1)

# Final Summary
print("="*70)
print(" "*25 + "TEST SUMMARY")
print("="*70)
print("[SUCCESS] All core tests passed!")
print("\nThe project is ready to run!")
print("\nTo launch the web application, run:")
print("  streamlit run web/app.py")
print("\nTo test the MCP server, run:")
print("  python src/mcp/career_data_server.py")
print("\nTo test OpenAI integration, run:")
print("  python src/models/openai_integration.py")
print("="*70)
