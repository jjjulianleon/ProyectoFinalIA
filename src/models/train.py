"""
Model training module for Career Prediction
Trains and evaluates Logistic Regression and Random Forest models
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

class CareerPredictionModel:
    """Handles training and evaluation of career prediction models"""

    def __init__(self):
        """Initialize model trainer"""
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None

        # Paths
        self.data_dir = Path(__file__).parent.parent.parent / 'data' / 'processed'
        self.models_dir = Path(__file__).parent.parent.parent / 'models'
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self):
        """Load preprocessed data"""
        print("Loading preprocessed data...")

        self.X_train = pd.read_csv(self.data_dir / 'X_train.csv')
        self.X_test = pd.read_csv(self.data_dir / 'X_test.csv')
        self.y_train = pd.read_csv(self.data_dir / 'y_train.csv')['Career_Encoded'].values
        self.y_test = pd.read_csv(self.data_dir / 'y_test.csv')['Career_Encoded'].values

        # Load label encoder for class names
        self.label_encoder = joblib.load(self.models_dir / 'label_encoder.joblib')
        self.feature_columns = joblib.load(self.models_dir / 'feature_columns.joblib')

        print(f"[SUCCESS] Loaded training set: {len(self.X_train)} samples")
        print(f"[SUCCESS] Loaded test set: {len(self.X_test)} samples")
        print(f"[SUCCESS] Features: {len(self.feature_columns)}")
        print(f"[SUCCESS] Classes: {len(self.label_encoder.classes_)}\n")

    def create_models(self):
        """Create ML models"""
        print("Creating models...\n")

        # Model 1: Logistic Regression
        self.models['Logistic Regression'] = LogisticRegression(
            max_iter=1000,
            random_state=42,
            multi_class='multinomial',
            solver='lbfgs'
        )

        # Model 2: Random Forest
        self.models['Random Forest'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )

        print(f"[SUCCESS] Created {len(self.models)} models:")
        for model_name in self.models.keys():
            print(f"  - {model_name}")
        print()

    def train_model(self, model_name, model):
        """Train a single model"""
        print(f"Training {model_name}...")
        model.fit(self.X_train, self.y_train)
        print(f"[SUCCESS] {model_name} trained\n")

    def evaluate_model(self, model_name, model):
        """Evaluate a model and store results"""
        print(f"Evaluating {model_name}...")

        # Make predictions
        y_train_pred = model.predict(self.X_train)
        y_test_pred = model.predict(self.X_test)

        # Calculate metrics
        train_accuracy = accuracy_score(self.y_train, y_train_pred)
        test_accuracy = accuracy_score(self.y_test, y_test_pred)

        # Calculate precision, recall, f1 (weighted average for multi-class)
        precision = precision_score(self.y_test, y_test_pred, average='weighted', zero_division=0)
        recall = recall_score(self.y_test, y_test_pred, average='weighted', zero_division=0)
        f1 = f1_score(self.y_test, y_test_pred, average='weighted', zero_division=0)

        # Cross-validation score
        cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='accuracy')

        # Store results
        self.results[model_name] = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'y_pred': y_test_pred
        }

        # Print results
        print(f"\n{'='*60}")
        print(f"{model_name} - Performance Metrics")
        print(f"{'='*60}")
        print(f"Training Accuracy:   {train_accuracy:.4f}")
        print(f"Test Accuracy:       {test_accuracy:.4f}")
        print(f"Precision:           {precision:.4f}")
        print(f"Recall:              {recall:.4f}")
        print(f"F1-Score:            {f1:.4f}")
        print(f"Cross-Val Accuracy:  {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print(f"{'='*60}\n")

    def print_classification_report(self, model_name):
        """Print detailed classification report"""
        y_pred = self.results[model_name]['y_pred']

        print(f"\n{model_name} - Detailed Classification Report:")
        print("="*60)
        report = classification_report(
            self.y_test, y_pred,
            target_names=self.label_encoder.classes_,
            zero_division=0
        )
        print(report)

    def compare_models(self):
        """Compare all trained models"""
        print("\n" + "="*70)
        print(" "*25 + "MODEL COMPARISON")
        print("="*70)

        # Create comparison DataFrame
        comparison = pd.DataFrame({
            'Model': list(self.results.keys()),
            'Test Accuracy': [res['test_accuracy'] for res in self.results.values()],
            'Precision': [res['precision'] for res in self.results.values()],
            'Recall': [res['recall'] for res in self.results.values()],
            'F1-Score': [res['f1_score'] for res in self.results.values()],
            'CV Mean': [res['cv_mean'] for res in self.results.values()]
        })

        comparison = comparison.sort_values('Test Accuracy', ascending=False)
        print(comparison.to_string(index=False))
        print("="*70)

        # Determine best model
        self.best_model_name = comparison.iloc[0]['Model']
        self.best_model = self.models[self.best_model_name]

        print(f"\n[BEST MODEL] {self.best_model_name}")
        print(f"Test Accuracy: {self.results[self.best_model_name]['test_accuracy']:.4f}\n")

    def plot_confusion_matrix(self, model_name):
        """Plot confusion matrix for a model"""
        y_pred = self.results[model_name]['y_pred']

        cm = confusion_matrix(self.y_test, y_pred)

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=self.label_encoder.classes_,
            yticklabels=self.label_encoder.classes_
        )
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        # Save plot
        plots_dir = self.models_dir / 'plots'
        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_path = plots_dir / f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"[SUCCESS] Saved confusion matrix to {plot_path}")
        plt.close()

    def get_feature_importance(self):
        """Get feature importance from Random Forest"""
        if 'Random Forest' not in self.models:
            print("Random Forest model not found")
            return

        rf_model = self.models['Random Forest']
        importance = rf_model.feature_importances_

        # Create DataFrame
        feature_importance_df = pd.DataFrame({
            'Feature': self.feature_columns,
            'Importance': importance
        }).sort_values('Importance', ascending=False)

        print("\n" + "="*60)
        print("Feature Importance (Random Forest)")
        print("="*60)
        print(feature_importance_df.to_string(index=False))
        print("="*60 + "\n")

        # Plot feature importance
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
        plt.xlabel('Importance')
        plt.title('Feature Importance - Random Forest')
        plt.gca().invert_yaxis()
        plt.tight_layout()

        plots_dir = self.models_dir / 'plots'
        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_path = plots_dir / 'feature_importance.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"[SUCCESS] Saved feature importance plot to {plot_path}")
        plt.close()

        return feature_importance_df

    def save_models(self):
        """Save trained models"""
        print("\nSaving models...")

        for model_name, model in self.models.items():
            filename = model_name.lower().replace(' ', '_') + '.joblib'
            filepath = self.models_dir / filename
            joblib.dump(model, filepath)
            print(f"[SUCCESS] Saved {model_name} to {filepath}")

        # Save best model separately
        joblib.dump(self.best_model, self.models_dir / 'best_model.joblib')
        print(f"[SUCCESS] Saved best model ({self.best_model_name}) to best_model.joblib\n")

    def run_full_training(self):
        """Run complete training pipeline"""
        print("\n" + "="*70)
        print(" "*20 + "MODEL TRAINING PIPELINE")
        print("="*70 + "\n")

        # Load data
        self.load_data()

        # Create models
        self.create_models()

        # Train and evaluate each model
        for model_name, model in self.models.items():
            self.train_model(model_name, model)
            self.evaluate_model(model_name, model)
            self.print_classification_report(model_name)
            self.plot_confusion_matrix(model_name)

        # Compare models
        self.compare_models()

        # Feature importance
        self.get_feature_importance()

        # Save models
        self.save_models()

        print("="*70)
        print("TRAINING COMPLETE!")
        print("="*70)
        print("\nNext step: Run 'streamlit run web/app.py' to launch the web app\n")


def main():
    """Main function"""
    trainer = CareerPredictionModel()
    trainer.run_full_training()


if __name__ == "__main__":
    main()
