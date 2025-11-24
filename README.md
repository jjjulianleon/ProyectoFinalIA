# CareerPath AI: Educational Data-Driven Career Trajectory Prediction System

## 🎓 Project Overview

CareerPath AI is an intelligent career prediction system that uses machine learning to help students make informed decisions about their career paths based on their educational background, skills, and interests.

This project was developed as part of a COIL (Collaborative Online International Learning) collaboration between students from Universidad and Virginia Tech.

## ✨ Features

- **Smart Career Predictions**: Uses ML models (Logistic Regression & Random Forest) to predict top career paths
- **AI-Powered Insights**: Integrates OpenAI API for personalized career advice and explanations
- **Interactive Web Interface**: Built with Streamlit for easy interaction
- **Custom MCP Server**: Implements a CareerData MCP server for efficient data management
- **Real Dataset**: Uses real-world career and educational data
- **Visual Analytics**: Charts and graphs showing prediction confidence and feature importance

## 🏗️ Project Structure

```
CareerPathAI/
├── data/
│   ├── raw/              # Original datasets
│   └── processed/        # Cleaned and processed data
├── src/
│   ├── data/            # Data processing modules
│   ├── models/          # ML model training and prediction
│   └── mcp/             # Custom CareerData MCP server
├── web/
│   └── app.py           # Streamlit web application
├── notebooks/           # Jupyter notebooks for experimentation
├── docs/               # Documentation
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git

### Installation

1. Clone the repository:
```bash
git clone https://github.com/jjjulianleon/ProyectoFinalIA.git
cd ProyectoFinalIA
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your OpenAI API key:
```bash
# Create a .env file
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

### Running the Application

1. **Train the models** (first time only):
```bash
python src/models/train.py
```

2. **Launch the web application**:
```bash
streamlit run web/app.py
```

3. Open your browser and navigate to `http://localhost:8501`

## 🧠 Machine Learning Models

The system uses two complementary models:

1. **Logistic Regression**: Fast, interpretable baseline model
2. **Random Forest**: More sophisticated ensemble model for better accuracy

Models are evaluated using:
- Accuracy
- Precision, Recall, F1-Score
- Cross-validation
- Feature importance analysis

## 🤖 AI Integration

The project integrates OpenAI's GPT models to provide:
- Personalized career descriptions
- Actionable advice for career development
- Explanations of prediction reasoning
- Skill development recommendations

## 📊 Dataset

We use a real-world career dataset containing:
- Educational background (major, GPA, institution type)
- Skills and competencies
- Extracurricular activities
- Career outcomes

## 🛠️ Custom MCP Server

The CareerData MCP server provides:
- Efficient career data lookup
- Data preprocessing capabilities
- Integration with the main application
- Extensible architecture for future features

## 📚 Learning Objectives

### Conceptual Understanding
- Ethical application of ML in career prediction
- Understanding model limitations and biases
- Importance of transparency in AI systems

### Skill Development
- Data preprocessing and feature engineering
- Supervised learning implementation (classification)
- Web application development with Streamlit
- API integration (OpenAI)
- MCP server development

## 🤝 Team & Collaboration

This project is a collaborative effort between:
- Universidad students
- Virginia Tech students

Developed as part of the Artificial Intelligence course curriculum.

## 📖 Documentation

For detailed documentation, see:
- [Architecture Guide](docs/architecture.md)
- [User Guide](docs/user_guide.md)
- [Developer Guide](docs/developer_guide.md)

## 🔮 Future Enhancements

- Additional ML models (Gradient Boosting, Neural Networks)
- Bias detection and fairness metrics
- Salary information integration
- Job market trends analysis
- Mobile-responsive design

## 📄 License

This project is developed for educational purposes.

## 🙏 Acknowledgments

- Dataset providers (Kaggle, BLS)
- OpenAI for API access
- Virginia Tech and Universidad San Francisco de Quito for the COIL collaboration opportunity
- Course instructors and mentors

