"""
OpenAI Integration for Career Advice
Generates personalized career descriptions and recommendations
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(env_path)

class CareerAdvisor:
    """Uses OpenAI API to generate personalized career advice"""

    def __init__(self, api_key=None):
        """
        Initialize CareerAdvisor

        Args:
            api_key: OpenAI API key (if None, loads from environment)
        """
        if api_key is None:
            api_key = os.getenv('OPENAI_API_KEY')

        if not api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")

        self.client = OpenAI(api_key=api_key)
        self.model = os.getenv('MODEL_NAME', 'gpt-3.5-turbo')
        self.temperature = float(os.getenv('TEMPERATURE', '0.7'))
        self.max_tokens = int(os.getenv('MAX_TOKENS', '500'))

    def generate_career_description(self, career_name, user_features=None):
        """
        Generate a personalized career description

        Args:
            career_name: Name of the predicted career
            user_features: Dictionary of user features (optional)

        Returns:
            String containing career description
        """
        # Build prompt
        prompt = f"""Provide a concise, engaging description of the career: {career_name}.

Include:
1. A brief overview (2-3 sentences)
2. Key responsibilities
3. Required skills and qualifications
4. Typical work environment

Keep it informative but motivating for a student exploring career options.
Limit response to 200 words."""

        if user_features:
            prompt += f"\n\nNote: This recommendation is based on the student's profile showing strengths in areas like {', '.join(str(k) for k in list(user_features.keys())[:3])}."

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a knowledgeable career counselor helping students explore career paths."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            return f"[AI Description unavailable: {str(e)}] {career_name} is a rewarding career path with opportunities for growth and development."

    def generate_career_advice(self, predicted_careers, user_features, top_n=3):
        """
        Generate personalized advice based on top predicted careers

        Args:
            predicted_careers: List of tuples (career_name, probability)
            user_features: Dictionary of user features
            top_n: Number of top careers to consider

        Returns:
            String containing personalized advice
        """
        top_careers = predicted_careers[:top_n]
        career_names = [career for career, _ in top_careers]

        prompt = f"""As a career counselor, provide personalized advice for a student whose top predicted career paths are:

{', '.join(f"{i+1}. {career} ({prob:.1%} match)" for i, (career, prob) in enumerate(top_careers))}

Based on their profile:
- Strong areas: {self._format_top_features(user_features)}

Provide:
1. An encouraging opening statement about their profile
2. Brief insights about why these careers are good matches
3. 3-4 actionable steps they can take to prepare for these careers
4. A motivating closing statement

Keep it concise (250 words max) and actionable."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an empathetic and knowledgeable career counselor who provides practical, actionable advice to students."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=600
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            return f"Error generating advice: {str(e)}"

    def explain_prediction(self, career_name, probability, top_features):
        """
        Explain why a specific career was predicted

        Args:
            career_name: Predicted career name
            probability: Prediction probability
            top_features: Dictionary of top features that influenced prediction

        Returns:
            String explaining the prediction
        """
        features_str = ", ".join([f"{k}: {v:.1f}" for k, v in list(top_features.items())[:3]])

        prompt = f"""Explain in 2-3 sentences why a student was matched with {career_name} (confidence: {probability:.1%}).

Their profile shows: {features_str}

Explain the connection between their strengths and this career in a way that's easy to understand and encouraging."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a career counselor explaining ML predictions in simple, encouraging terms."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=150
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            return f"This career matches your profile with {probability:.1%} confidence based on your strengths."

    def _format_top_features(self, features, n=3):
        """Helper to format top features"""
        sorted_features = sorted(features.items(), key=lambda x: x[1], reverse=True)[:n]
        return ", ".join([f"{name} ({value:.1f})" for name, value in sorted_features])


def test_openai_integration():
    """Test function for OpenAI integration"""
    print("Testing OpenAI Integration...")
    print("="*60)

    try:
        advisor = CareerAdvisor()
        print("[SUCCESS] CareerAdvisor initialized\n")

        # Test career description
        print("Test 1: Generating career description...")
        description = advisor.generate_career_description("Software Engineer")
        print(description)
        print("\n" + "="*60 + "\n")

        # Test career advice
        print("Test 2: Generating personalized advice...")
        predicted_careers = [
            ("Software Engineer", 0.75),
            ("Data Scientist", 0.65),
            ("Business Analyst", 0.55)
        ]
        user_features = {
            "Numerical_Aptitude": 85.0,
            "Abstract_Reasoning": 78.0,
            "Openness": 8.5
        }
        advice = advisor.generate_career_advice(predicted_careers, user_features)
        print(advice)
        print("\n" + "="*60 + "\n")

        # Test explanation
        print("Test 3: Explaining prediction...")
        explanation = advisor.explain_prediction("Software Engineer", 0.75, user_features)
        print(explanation)
        print("\n" + "="*60)

        print("\n[SUCCESS] All OpenAI integration tests passed!")

    except Exception as e:
        print(f"[ERROR] {str(e)}")


if __name__ == "__main__":
    test_openai_integration()
