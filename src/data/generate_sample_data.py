"""
Generate a sample career prediction dataset for testing
This creates a synthetic but realistic dataset if the real Kaggle dataset is unavailable
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

def generate_career_dataset(n_samples=1000):
    """
    Generate synthetic career prediction dataset

    Args:
        n_samples: Number of samples to generate

    Returns:
        DataFrame with career prediction data
    """

    # Define career categories
    careers = [
        'Software Engineer',
        'Data Scientist',
        'Business Analyst',
        'Teacher',
        'Accountant',
        'Marketing Manager',
        'Mechanical Engineer',
        'Graphic Designer',
        'Healthcare Professional',
        'Sales Representative'
    ]

    data = {
        # OCEAN Personality Traits (1-10 scale)
        'Openness': np.random.normal(6, 2, n_samples).clip(1, 10),
        'Conscientiousness': np.random.normal(6.5, 2, n_samples).clip(1, 10),
        'Extraversion': np.random.normal(5.5, 2.5, n_samples).clip(1, 10),
        'Agreeableness': np.random.normal(6, 2, n_samples).clip(1, 10),
        'Neuroticism': np.random.normal(5, 2, n_samples).clip(1, 10),

        # Aptitude Scores (0-100 scale)
        'Numerical_Aptitude': np.random.normal(65, 15, n_samples).clip(0, 100),
        'Spatial_Aptitude': np.random.normal(60, 15, n_samples).clip(0, 100),
        'Perceptual_Aptitude': np.random.normal(65, 12, n_samples).clip(0, 100),
        'Abstract_Reasoning': np.random.normal(63, 14, n_samples).clip(0, 100),
        'Verbal_Reasoning': np.random.normal(67, 13, n_samples).clip(0, 100),
    }

    df = pd.DataFrame(data)

    # Generate career labels based on feature combinations (with some logic)
    career_labels = []
    for idx, row in df.iterrows():
        # Simple rules to make data somewhat realistic
        if row['Numerical_Aptitude'] > 70 and row['Abstract_Reasoning'] > 70:
            if row['Openness'] > 6:
                career = np.random.choice(['Software Engineer', 'Data Scientist'], p=[0.6, 0.4])
            else:
                career = np.random.choice(['Mechanical Engineer', 'Accountant'], p=[0.5, 0.5])
        elif row['Verbal_Reasoning'] > 70:
            if row['Extraversion'] > 6:
                career = np.random.choice(['Marketing Manager', 'Sales Representative'], p=[0.5, 0.5])
            else:
                career = np.random.choice(['Teacher', 'Business Analyst'], p=[0.6, 0.4])
        elif row['Spatial_Aptitude'] > 65:
            if row['Openness'] > 7:
                career = 'Graphic Designer'
            else:
                career = 'Mechanical Engineer'
        elif row['Agreeableness'] > 7:
            career = np.random.choice(['Healthcare Professional', 'Teacher'], p=[0.6, 0.4])
        else:
            # Random assignment for edge cases
            career = np.random.choice(careers)

        career_labels.append(career)

    df['Career'] = career_labels

    # Round numeric values for cleaner data
    df = df.round(2)

    return df

def main():
    """Generate and save the sample dataset"""

    print("Generating sample career prediction dataset...")

    # Generate dataset
    df = generate_career_dataset(n_samples=1000)

    # Create output directory if it doesn't exist
    output_dir = Path(__file__).parent.parent.parent / 'data' / 'raw'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    output_path = output_dir / 'career_data.csv'
    df.to_csv(output_path, index=False)

    print(f"[SUCCESS] Dataset generated successfully!")
    print(f"[SUCCESS] Saved to: {output_path}")
    print(f"\nDataset Statistics:")
    print(f"- Total samples: {len(df)}")
    print(f"- Features: {len(df.columns) - 1}")
    print(f"- Career categories: {df['Career'].nunique()}")
    print(f"\nCareer distribution:")
    print(df['Career'].value_counts())
    print(f"\nFirst few rows:")
    print(df.head())

if __name__ == "__main__":
    main()
