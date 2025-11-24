"""
Prepare real Kaggle dataset for the career prediction system
Renames columns and ensures compatibility with existing pipeline
"""

import pandas as pd
from pathlib import Path

def prepare_real_dataset():
    """
    Load and prepare the real Kaggle dataset
    """
    print("="*70)
    print(" "*15 + "PREPARING REAL KAGGLE DATASET")
    print("="*70 + "\n")

    # Paths
    raw_data_dir = Path(__file__).parent.parent.parent / 'data' / 'raw'
    kaggle_file = raw_data_dir / 'Data_final.csv'
    output_file = raw_data_dir / 'career_data_real.csv'

    # Backup synthetic data
    synthetic_file = raw_data_dir / 'career_data.csv'
    backup_file = raw_data_dir / 'career_data_synthetic.csv'

    if synthetic_file.exists():
        import shutil
        shutil.copy(synthetic_file, backup_file)
        print(f"[SUCCESS] Backed up synthetic data to: career_data_synthetic.csv")

    # Load Kaggle dataset
    print(f"Loading real dataset from: {kaggle_file}")
    df = pd.read_csv(kaggle_file)

    print(f"[SUCCESS] Loaded {len(df)} rows\n")

    # Rename columns to match our expected format
    column_mapping = {
        'O_score': 'Openness',
        'C_score': 'Conscientiousness',
        'E_score': 'Extraversion',
        'A_score': 'Agreeableness',
        'N_score': 'Neuroticism',
        'Numerical Aptitude': 'Numerical_Aptitude',
        'Spatial Aptitude': 'Spatial_Aptitude',
        'Perceptual Aptitude': 'Perceptual_Aptitude',
        'Abstract Reasoning': 'Abstract_Reasoning',
        'Verbal Reasoning': 'Verbal_Reasoning',
        'Career': 'Career'
    }

    df_renamed = df.rename(columns=column_mapping)

    print("Column mapping applied:")
    for old, new in column_mapping.items():
        print(f"  {old:25} -> {new}")
    print()

    # Data exploration
    print("="*70)
    print("DATASET STATISTICS")
    print("="*70)
    print(f"\nTotal records: {len(df_renamed)}")
    print(f"Total features: {len(df_renamed.columns) - 1}")
    print(f"Career categories: {df_renamed['Career'].nunique()}")

    print("\nCareer distribution:")
    career_counts = df_renamed['Career'].value_counts()
    for career, count in career_counts.items():
        print(f"  {career:30} : {count:3d}")

    print("\nFeature statistics:")
    print(df_renamed.describe().round(2))

    # Check for missing values
    missing = df_renamed.isnull().sum()
    if missing.sum() > 0:
        print("\n[WARNING] Missing values found:")
        print(missing[missing > 0])
    else:
        print("\n[SUCCESS] No missing values found")

    # Save processed dataset
    df_renamed.to_csv(output_file, index=False)
    print(f"\n[SUCCESS] Saved real dataset to: {output_file}")

    # Also save as main dataset (replacing synthetic)
    df_renamed.to_csv(raw_data_dir / 'career_data.csv', index=False)
    print(f"[SUCCESS] Replaced main dataset with real data")

    print("\n" + "="*70)
    print("REAL DATASET READY!")
    print("="*70)
    print("\nNext steps:")
    print("1. Run: python src/data/preprocess.py")
    print("2. Run: python src/models/train.py")
    print("3. Run: streamlit run web/app.py")
    print("\nNote: Your synthetic data backup is saved as 'career_data_synthetic.csv'")

    return df_renamed

if __name__ == "__main__":
    prepare_real_dataset()
