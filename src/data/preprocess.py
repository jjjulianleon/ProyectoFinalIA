"""
Data preprocessing module for Career Prediction
Handles loading, cleaning, and preparing data for ML models
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

class CareerDataPreprocessor:
    """Handles all data preprocessing tasks"""

    def __init__(self, data_path=None):
        """
        Initialize preprocessor

        Args:
            data_path: Path to the raw CSV file
        """
        if data_path is None:
            # Default path
            self.data_path = Path(__file__).parent.parent.parent / 'data' / 'raw' / 'career_data.csv'
        else:
            self.data_path = Path(data_path)

        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.target_column = 'Career'

    def load_data(self):
        """Load data from CSV file"""
        print(f"Loading data from: {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        print(f"[SUCCESS] Loaded {len(self.df)} rows and {len(self.df.columns)} columns")
        return self.df

    def explore_data(self):
        """Print basic exploratory data analysis"""
        print("\n" + "="*50)
        print("DATA EXPLORATION")
        print("="*50)

        print("\nDataset Shape:", self.df.shape)
        print("\nColumn Names and Types:")
        print(self.df.dtypes)

        print("\nMissing Values:")
        print(self.df.isnull().sum())

        print("\nBasic Statistics:")
        print(self.df.describe())

        if self.target_column in self.df.columns:
            print(f"\nTarget Variable ({self.target_column}) Distribution:")
            print(self.df[self.target_column].value_counts())

        print("="*50 + "\n")

    def clean_data(self):
        """Clean the dataset"""
        print("Cleaning data...")

        # Remove duplicates
        initial_rows = len(self.df)
        self.df = self.df.drop_duplicates()
        print(f"- Removed {initial_rows - len(self.df)} duplicate rows")

        # Handle missing values
        missing_count = self.df.isnull().sum().sum()
        if missing_count > 0:
            print(f"- Found {missing_count} missing values")
            # For numerical columns, fill with median
            numerical_cols = self.df.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                if self.df[col].isnull().any():
                    median_val = self.df[col].median()
                    self.df[col].fillna(median_val, inplace=True)
                    print(f"  - Filled {col} with median: {median_val:.2f}")
        else:
            print("- No missing values found")

        print("[SUCCESS] Data cleaning completed\n")
        return self.df

    def prepare_features(self):
        """Prepare features for modeling"""
        print("Preparing features...")

        # Separate features and target
        if self.target_column not in self.df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in dataset")

        # Get feature columns (all except target)
        self.feature_columns = [col for col in self.df.columns if col != self.target_column]

        X = self.df[self.feature_columns]
        y = self.df[self.target_column]

        print(f"- Features: {len(self.feature_columns)} columns")
        print(f"- Target: {self.target_column}")
        print(f"- Classes: {y.nunique()} unique careers")

        return X, y

    def encode_labels(self, y):
        """Encode target labels to numerical values"""
        print("Encoding target labels...")

        y_encoded = self.label_encoder.fit_transform(y)

        # Create mapping dictionary for reference
        self.label_mapping = dict(zip(
            self.label_encoder.classes_,
            self.label_encoder.transform(self.label_encoder.classes_)
        ))

        print(f"- Encoded {len(self.label_mapping)} career categories")
        print("- Label mapping:")
        for career, code in sorted(self.label_mapping.items(), key=lambda x: x[1]):
            print(f"  {code}: {career}")

        return y_encoded

    def scale_features(self, X, fit=True):
        """
        Scale features using StandardScaler

        Args:
            X: Features dataframe
            fit: Whether to fit the scaler (True for training, False for new data)
        """
        if fit:
            print("Scaling features...")
            X_scaled = self.scaler.fit_transform(X)
            print("[SUCCESS] Features scaled using StandardScaler")
        else:
            X_scaled = self.scaler.transform(X)

        # Convert back to DataFrame to preserve column names
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_columns, index=X.index)

        return X_scaled

    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        print(f"\nSplitting data (test_size={test_size})...")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        print(f"- Training set: {len(X_train)} samples")
        print(f"- Test set: {len(X_test)} samples")

        return X_train, X_test, y_train, y_test

    def save_preprocessed_data(self, X_train, X_test, y_train, y_test):
        """Save preprocessed data and preprocessing objects"""
        print("\nSaving preprocessed data...")

        # Create processed data directory
        processed_dir = Path(__file__).parent.parent.parent / 'data' / 'processed'
        processed_dir.mkdir(parents=True, exist_ok=True)

        # Save datasets
        X_train.to_csv(processed_dir / 'X_train.csv', index=False)
        X_test.to_csv(processed_dir / 'X_test.csv', index=False)
        pd.DataFrame(y_train, columns=['Career_Encoded']).to_csv(processed_dir / 'y_train.csv', index=False)
        pd.DataFrame(y_test, columns=['Career_Encoded']).to_csv(processed_dir / 'y_test.csv', index=False)

        # Save preprocessing objects
        models_dir = Path(__file__).parent.parent.parent / 'models'
        models_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.label_encoder, models_dir / 'label_encoder.joblib')
        joblib.dump(self.scaler, models_dir / 'scaler.joblib')
        joblib.dump(self.feature_columns, models_dir / 'feature_columns.joblib')

        print(f"[SUCCESS] Saved to {processed_dir}")
        print(f"[SUCCESS] Saved preprocessing objects to {models_dir}")

    def run_full_pipeline(self):
        """Run the complete preprocessing pipeline"""
        print("\n" + "="*70)
        print(" "*20 + "DATA PREPROCESSING PIPELINE")
        print("="*70 + "\n")

        # Load data
        self.load_data()

        # Explore data
        self.explore_data()

        # Clean data
        self.clean_data()

        # Prepare features
        X, y = self.prepare_features()

        # Encode labels
        y_encoded = self.encode_labels(y)

        # Scale features
        X_scaled = self.scale_features(X, fit=True)

        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X_scaled, y_encoded)

        # Save everything
        self.save_preprocessed_data(X_train, X_test, y_train, y_test)

        print("\n" + "="*70)
        print("PREPROCESSING COMPLETE!")
        print("="*70)
        print("\nNext step: Run 'python src/models/train.py' to train the models\n")

        return X_train, X_test, y_train, y_test


def main():
    """Main function to run preprocessing"""
    preprocessor = CareerDataPreprocessor()
    preprocessor.run_full_pipeline()


if __name__ == "__main__":
    main()
