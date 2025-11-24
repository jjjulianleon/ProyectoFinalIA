"""
CareerPath AI - Streamlit Web Application
Interactive interface for career prediction and AI-powered advice
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys
import plotly.graph_objects as go
import plotly.express as px

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from src.models.openai_integration import CareerAdvisor

# Page configuration
st.set_page_config(
    page_title="CareerPath AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .prediction-card {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    .career-title {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .probability {
        font-size: 1.2rem;
        color: #28a745;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load trained models and preprocessing objects"""
    models_dir = Path(__file__).parent.parent / 'models'

    try:
        model = joblib.load(models_dir / 'best_model.joblib')
        label_encoder = joblib.load(models_dir / 'label_encoder.joblib')
        scaler = joblib.load(models_dir / 'scaler.joblib')
        feature_columns = joblib.load(models_dir / 'feature_columns.joblib')

        return model, label_encoder, scaler, feature_columns
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.info("Please run 'python src/models/train.py' first to train the models.")
        st.stop()

@st.cache_resource
def initialize_ai_advisor():
    """Initialize OpenAI advisor"""
    try:
        advisor = CareerAdvisor()
        return advisor
    except Exception as e:
        st.warning(f"AI Advisor unavailable: {e}")
        return None

def create_prediction_chart(predictions_df, top_n=5):
    """Create interactive bar chart for predictions"""
    top_predictions = predictions_df.head(top_n)

    fig = go.Figure(data=[
        go.Bar(
            x=top_predictions['Probability'],
            y=top_predictions['Career'],
            orientation='h',
            marker=dict(
                color=top_predictions['Probability'],
                colorscale='Blues',
                showscale=True,
                colorbar=dict(title="Probability")
            ),
            text=top_predictions['Probability'].apply(lambda x: f'{x:.1%}'),
            textposition='auto',
        )
    ])

    fig.update_layout(
        title=f'Top {top_n} Career Predictions',
        xaxis_title='Probability',
        yaxis_title='Career',
        height=400,
        yaxis={'categoryorder': 'total ascending'}
    )

    return fig

def create_feature_radar_chart(user_features):
    """Create radar chart showing user's profile"""
    features = list(user_features.keys())
    values = list(user_features.values())

    # Normalize values to 0-100 scale for better visualization
    max_vals = {
        'Openness': 10, 'Conscientiousness': 10, 'Extraversion': 10,
        'Agreeableness': 10, 'Neuroticism': 10,
        'Numerical_Aptitude': 100, 'Spatial_Aptitude': 100,
        'Perceptual_Aptitude': 100, 'Abstract_Reasoning': 100,
        'Verbal_Reasoning': 100
    }

    normalized_values = [(values[i] / max_vals.get(features[i], 100)) * 100 for i in range(len(features))]

    fig = go.Figure(data=go.Scatterpolar(
        r=normalized_values,
        theta=features,
        fill='toself',
        marker=dict(color='#1f77b4')
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        title="Your Profile",
        height=500
    )

    return fig

def main():
    """Main application"""

    # Header
    st.markdown('<div class="main-header">🎓 CareerPath AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Discover your ideal career path with AI-powered predictions</div>',
        unsafe_allow_html=True
    )

    # Load models
    model, label_encoder, scaler, feature_columns = load_models()
    advisor = initialize_ai_advisor()

    # Sidebar - Input Form
    st.sidebar.header("📝 Enter Your Profile")
    st.sidebar.markdown("Fill in your information below:")

    # OCEAN Personality Traits
    st.sidebar.subheader("🧠 Personality Traits (1-10)")
    openness = st.sidebar.slider("Openness to Experience", 1.0, 10.0, 6.0, 0.1,
                                  help="Curiosity, creativity, and openness to new experiences")
    conscientiousness = st.sidebar.slider("Conscientiousness", 1.0, 10.0, 6.5, 0.1,
                                           help="Organization, responsibility, and diligence")
    extraversion = st.sidebar.slider("Extraversion", 1.0, 10.0, 5.5, 0.1,
                                      help="Sociability, assertiveness, and energy")
    agreeableness = st.sidebar.slider("Agreeableness", 1.0, 10.0, 6.0, 0.1,
                                       help="Compassion, cooperation, and trust")
    neuroticism = st.sidebar.slider("Neuroticism", 1.0, 10.0, 5.0, 0.1,
                                     help="Emotional sensitivity and tendency to worry")

    # Aptitude Scores
    st.sidebar.subheader("📊 Aptitude Scores (0-100)")
    numerical = st.sidebar.slider("Numerical Aptitude", 0.0, 100.0, 65.0, 1.0,
                                   help="Mathematical and quantitative reasoning")
    spatial = st.sidebar.slider("Spatial Aptitude", 0.0, 100.0, 60.0, 1.0,
                                 help="Visual-spatial reasoning and mental manipulation")
    perceptual = st.sidebar.slider("Perceptual Aptitude", 0.0, 100.0, 65.0, 1.0,
                                    help="Pattern recognition and attention to detail")
    abstract = st.sidebar.slider("Abstract Reasoning", 0.0, 100.0, 63.0, 1.0,
                                  help="Logical thinking and problem-solving")
    verbal = st.sidebar.slider("Verbal Reasoning", 0.0, 100.0, 67.0, 1.0,
                                help="Language comprehension and communication")

    # Predict button
    predict_button = st.sidebar.button("🔮 Predict My Career Path", type="primary", use_container_width=True)

    # Main content area
    if predict_button:
        # Prepare input data
        user_data = {
            'Openness': openness,
            'Conscientiousness': conscientiousness,
            'Extraversion': extraversion,
            'Agreeableness': agreeableness,
            'Neuroticism': neuroticism,
            'Numerical_Aptitude': numerical,
            'Spatial_Aptitude': spatial,
            'Perceptual_Aptitude': perceptual,
            'Abstract_Reasoning': abstract,
            'Verbal_Reasoning': verbal
        }

        # Create DataFrame
        input_df = pd.DataFrame([user_data], columns=feature_columns)

        # Scale features
        input_scaled = scaler.transform(input_df)

        # Make prediction
        with st.spinner("🔮 Analyzing your profile..."):
            probabilities = model.predict_proba(input_scaled)[0]
            predicted_class = model.predict(input_scaled)[0]

            # Get career names and probabilities
            career_probs = list(zip(label_encoder.classes_, probabilities))
            career_probs_sorted = sorted(career_probs, key=lambda x: x[1], reverse=True)

            # Create predictions DataFrame
            predictions_df = pd.DataFrame(career_probs_sorted, columns=['Career', 'Probability'])

        # Display Results
        st.success("✅ Prediction Complete!")

        # Create two columns
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📈 Top Career Predictions")

            # Display top 5 predictions
            for idx, (career, prob) in enumerate(career_probs_sorted[:5], 1):
                with st.container():
                    st.markdown(f"""
                    <div class="prediction-card">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <div style="font-size: 1.1rem; color: #666;">#{idx}</div>
                                <div class="career-title">{career}</div>
                            </div>
                            <div class="probability">{prob:.1%}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # Prediction chart
            st.plotly_chart(create_prediction_chart(predictions_df), use_container_width=True)

        with col2:
            st.subheader("👤 Your Profile")
            st.plotly_chart(create_feature_radar_chart(user_data), use_container_width=True)

        # AI-Generated Insights
        if advisor:
            st.markdown("---")
            st.header("🤖 AI-Powered Career Insights")

            with st.spinner("✨ Generating personalized advice..."):
                tab1, tab2, tab3 = st.tabs(["💬 Personalized Advice", "🎯 Top Career Details", "📖 Explanation"])

                with tab1:
                    advice = advisor.generate_career_advice(career_probs_sorted, user_data, top_n=3)
                    st.markdown(advice)

                with tab2:
                    top_career = career_probs_sorted[0][0]
                    description = advisor.generate_career_description(top_career, user_data)
                    st.markdown(f"### {top_career}")
                    st.markdown(description)

                with tab3:
                    top_career, top_prob = career_probs_sorted[0]
                    explanation = advisor.explain_prediction(top_career, top_prob, user_data)
                    st.info(explanation)

        # Feature Importance
        st.markdown("---")
        st.subheader("🔍 What Influenced Your Results?")

        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'Feature': feature_columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=True)

            fig = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title='Feature Importance in Prediction',
                color='Importance',
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig, use_container_width=True)

    else:
        # Welcome screen
        st.info("👈 Fill in your profile information in the sidebar and click 'Predict My Career Path' to get started!")

        # Display sample statistics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Career Categories", "10")
        with col2:
            st.metric("ML Models", "2")
        with col3:
            st.metric("Prediction Features", "10")

        # About section
        st.markdown("---")
        st.header("ℹ️ About CareerPath AI")

        st.markdown("""
        CareerPath AI uses machine learning to help students discover career paths that align with their:
        - **Personality traits** (OCEAN model)
        - **Aptitude scores** (Numerical, Spatial, Perceptual, Abstract, Verbal reasoning)

        ### 🎯 How it works:
        1. Enter your profile information in the sidebar
        2. Our AI models analyze your unique combination of traits and skills
        3. Get personalized career predictions with probability scores
        4. Receive AI-generated insights and actionable advice

        ### 🤖 Technology Stack:
        - **ML Models:** Random Forest & Logistic Regression
        - **AI Insights:** OpenAI GPT-3.5
        - **Web Framework:** Streamlit
        - **Data Science:** scikit-learn, pandas, numpy

        ### 👥 Developed by:
        COIL Project Team - Universidad & Virginia Tech
        """)

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>CareerPath AI | Powered by Machine Learning & OpenAI</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
