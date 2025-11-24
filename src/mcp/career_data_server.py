"""
CareerData MCP Server
A Model Context Protocol server for career data management and operations
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import joblib

class CareerDataMCPServer:
    """
    MCP Server for Career Data Operations

    Provides career information lookup, data statistics, and predictions
    following the Model Context Protocol pattern
    """

    def __init__(self, data_path=None, models_path=None):
        """
        Initialize MCP Server

        Args:
            data_path: Path to career dataset
            models_path: Path to models directory
        """
        self.base_path = Path(__file__).parent.parent.parent

        if data_path is None:
            self.data_path = self.base_path / 'data' / 'raw' / 'career_data.csv'
        else:
            self.data_path = Path(data_path)

        if models_path is None:
            self.models_path = self.base_path / 'models'
        else:
            self.models_path = Path(models_path)

        self.data = None
        self.model = None
        self.label_encoder = None
        self.scaler = None

        self._load_resources()

    def _load_resources(self):
        """Load dataset and models"""
        try:
            # Load dataset
            if self.data_path.exists():
                self.data = pd.read_csv(self.data_path)
                print(f"[MCP] Loaded dataset: {len(self.data)} records")
            else:
                print(f"[MCP] Warning: Dataset not found at {self.data_path}")

            # Load model and preprocessing objects
            model_file = self.models_path / 'best_model.joblib'
            if model_file.exists():
                self.model = joblib.load(model_file)
                self.label_encoder = joblib.load(self.models_path / 'label_encoder.joblib')
                self.scaler = joblib.load(self.models_path / 'scaler.joblib')
                self.feature_columns = joblib.load(self.models_path / 'feature_columns.joblib')
                print(f"[MCP] Loaded ML model: {type(self.model).__name__}")
            else:
                print(f"[MCP] Warning: Model not found. Prediction features unavailable.")

        except Exception as e:
            print(f"[MCP] Error loading resources: {e}")

    # ==================== MCP TOOLS ====================

    def get_career_info(self, career_name: str) -> Dict[str, Any]:
        """
        Get information about a specific career

        Args:
            career_name: Name of the career

        Returns:
            Dictionary with career information
        """
        if self.data is None:
            return {"error": "Dataset not loaded"}

        # Filter data for this career
        career_data = self.data[self.data['Career'] == career_name]

        if career_data.empty:
            return {
                "career": career_name,
                "found": False,
                "message": f"No data found for career: {career_name}",
                "available_careers": self.list_careers()['careers']
            }

        # Calculate statistics
        stats = {}
        numerical_cols = career_data.select_dtypes(include=['float64', 'int64']).columns

        for col in numerical_cols:
            stats[col] = {
                "mean": round(career_data[col].mean(), 2),
                "median": round(career_data[col].median(), 2),
                "min": round(career_data[col].min(), 2),
                "max": round(career_data[col].max(), 2)
            }

        return {
            "career": career_name,
            "found": True,
            "total_samples": len(career_data),
            "statistics": stats,
            "message": f"Found {len(career_data)} samples for {career_name}"
        }

    def list_careers(self) -> Dict[str, Any]:
        """
        List all available careers in the dataset

        Returns:
            Dictionary with list of careers and counts
        """
        if self.data is None:
            return {"error": "Dataset not loaded"}

        career_counts = self.data['Career'].value_counts().to_dict()

        return {
            "total_careers": len(career_counts),
            "careers": list(career_counts.keys()),
            "distribution": career_counts
        }

    def get_dataset_stats(self) -> Dict[str, Any]:
        """
        Get overall dataset statistics

        Returns:
            Dictionary with dataset statistics
        """
        if self.data is None:
            return {"error": "Dataset not loaded"}

        numerical_cols = self.data.select_dtypes(include=['float64', 'int64']).columns

        stats = {
            "total_records": len(self.data),
            "total_features": len(self.data.columns) - 1,  # Exclude target
            "careers": len(self.data['Career'].unique()),
            "feature_statistics": {}
        }

        for col in numerical_cols:
            stats["feature_statistics"][col] = {
                "mean": round(self.data[col].mean(), 2),
                "std": round(self.data[col].std(), 2),
                "min": round(self.data[col].min(), 2),
                "max": round(self.data[col].max(), 2)
            }

        return stats

    def predict_career(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Predict career based on input features

        Args:
            features: Dictionary of feature values

        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            return {"error": "ML model not loaded"}

        try:
            # Create DataFrame from features
            input_df = pd.DataFrame([features], columns=self.feature_columns)

            # Scale features
            input_scaled = self.scaler.transform(input_df)

            # Make prediction
            probabilities = self.model.predict_proba(input_scaled)[0]
            predicted_class = self.model.predict(input_scaled)[0]

            # Get top 5 predictions
            career_probs = list(zip(self.label_encoder.classes_, probabilities))
            career_probs_sorted = sorted(career_probs, key=lambda x: x[1], reverse=True)

            return {
                "success": True,
                "predicted_career": self.label_encoder.classes_[predicted_class],
                "confidence": round(float(probabilities[predicted_class]), 4),
                "top_predictions": [
                    {"career": career, "probability": round(float(prob), 4)}
                    for career, prob in career_probs_sorted[:5]
                ],
                "model": type(self.model).__name__
            }

        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}

    def search_similar_profiles(self, target_career: str, top_n: int = 5) -> Dict[str, Any]:
        """
        Find profiles similar to a target career

        Args:
            target_career: Career to search for
            top_n: Number of samples to return

        Returns:
            Dictionary with similar profiles
        """
        if self.data is None:
            return {"error": "Dataset not loaded"}

        career_data = self.data[self.data['Career'] == target_career]

        if career_data.empty:
            return {"error": f"No data found for career: {target_career}"}

        # Get random samples
        samples = career_data.sample(min(top_n, len(career_data))).to_dict('records')

        return {
            "career": target_career,
            "samples_count": len(samples),
            "profiles": samples
        }

    # ==================== MCP PROTOCOL METHODS ====================

    def handle_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle incoming MCP requests

        Args:
            method: Method name to call
            params: Parameters for the method

        Returns:
            Response dictionary
        """
        method_map = {
            "get_career_info": self.get_career_info,
            "list_careers": self.list_careers,
            "get_dataset_stats": self.get_dataset_stats,
            "predict_career": self.predict_career,
            "search_similar_profiles": self.search_similar_profiles
        }

        if method not in method_map:
            return {
                "error": f"Unknown method: {method}",
                "available_methods": list(method_map.keys())
            }

        try:
            result = method_map[method](**params)
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}


def test_mcp_server():
    """Test the MCP server functionality"""
    print("="*70)
    print(" "*20 + "CareerData MCP Server Test")
    print("="*70 + "\n")

    server = CareerDataMCPServer()

    # Test 1: List careers
    print("Test 1: List Careers")
    print("-" * 60)
    result = server.handle_request("list_careers", {})
    if result['success']:
        print(f"Total careers: {result['result']['total_careers']}")
        print(f"Careers: {', '.join(result['result']['careers'][:5])}...")
    print()

    # Test 2: Get dataset stats
    print("Test 2: Dataset Statistics")
    print("-" * 60)
    result = server.handle_request("get_dataset_stats", {})
    if result['success']:
        stats = result['result']
        print(f"Total records: {stats['total_records']}")
        print(f"Total features: {stats['total_features']}")
        print(f"Career categories: {stats['careers']}")
    print()

    # Test 3: Get career info
    print("Test 3: Get Career Info - Software Engineer")
    print("-" * 60)
    result = server.handle_request("get_career_info", {"career_name": "Software Engineer"})
    if result['success']:
        info = result['result']
        if info['found']:
            print(f"Samples: {info['total_samples']}")
            print(f"Average Numerical Aptitude: {info['statistics']['Numerical_Aptitude']['mean']}")
    print()

    # Test 4: Predict career
    print("Test 4: Career Prediction")
    print("-" * 60)
    test_features = {
        'Openness': 8.5,
        'Conscientiousness': 7.0,
        'Extraversion': 6.0,
        'Agreeableness': 7.5,
        'Neuroticism': 4.0,
        'Numerical_Aptitude': 85.0,
        'Spatial_Aptitude': 75.0,
        'Perceptual_Aptitude': 80.0,
        'Abstract_Reasoning': 82.0,
        'Verbal_Reasoning': 70.0
    }
    result = server.handle_request("predict_career", {"features": test_features})
    if result['success']:
        pred = result['result']
        print(f"Predicted career: {pred['predicted_career']}")
        print(f"Confidence: {pred['confidence']:.2%}")
        print("\nTop 3 predictions:")
        for p in pred['top_predictions'][:3]:
            print(f"  - {p['career']}: {p['probability']:.2%}")
    print()

    # Test 5: Search similar profiles
    print("Test 5: Search Similar Profiles - Teacher")
    print("-" * 60)
    result = server.handle_request("search_similar_profiles", {
        "target_career": "Teacher",
        "top_n": 3
    })
    if result['success']:
        search_result = result['result']
        print(f"Found {search_result['samples_count']} sample profiles for {search_result['career']}")
    print()

    print("="*70)
    print("[SUCCESS] All MCP server tests completed!")
    print("="*70)


if __name__ == "__main__":
    test_mcp_server()
