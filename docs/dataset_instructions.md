# Dataset Download Instructions

## Recommended Dataset

We're using the **Career Prediction Dataset** from Kaggle, which includes:
- Aptitude scores (Numerical, Spatial, Perceptual, Abstract, Verbal reasoning)
- OCEAN personality traits (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism)
- Career category predictions

## How to Download

### Option 1: Manual Download (Recommended for beginners)

1. Visit: https://www.kaggle.com/datasets/utkarshshrivastav07/career-prediction-dataset
2. Click "Download" button (you may need to create a free Kaggle account)
3. Extract the CSV file
4. Place the CSV file in: `data/raw/career_data.csv`

### Option 2: Using Kaggle API

1. Install Kaggle CLI:
```bash
pip install kaggle
```

2. Get your Kaggle API credentials:
   - Go to https://www.kaggle.com/settings
   - Click "Create New API Token"
   - Save the `kaggle.json` file to `~/.kaggle/` (Linux/Mac) or `C:\Users\<Username>\.kaggle\` (Windows)

3. Download the dataset:
```bash
kaggle datasets download -d utkarshshrivastav07/career-prediction-dataset
unzip career-prediction-dataset.zip -d data/raw/
```

## Alternative Datasets (if needed)

If the main dataset doesn't work, try these alternatives:

1. **Computer Science Students Career Prediction**
   - URL: https://www.kaggle.com/datasets/devildyno/computer-science-students-career-prediction
   - Good for CS-focused career prediction

2. **Studies Career Recommendation Dataset**
   - URL: https://www.kaggle.com/datasets/shanimmahir/studies-career-recommendation-dataset
   - General career guidance dataset

3. **Career Path Prediction for Different Fields**
   - URL: https://www.kaggle.com/datasets/ministerjohn/career-path-prediction-for-different-fields
   - Multiple field coverage

## After Download

Once you have the dataset:
1. Place it in `data/raw/`
2. Run the preprocessing script: `python src/data/preprocess.py`
3. The processed data will be saved to `data/processed/`

## Dataset Requirements

The dataset should have:
- At least 500 rows
- Multiple features (educational, personality, skills)
- Clear career categories (target variable)
- No excessive missing values
