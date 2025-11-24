User Guide - CareerPath AI

## Welcome to CareerPath AI! 🎓

This guide will help you use the CareerPath AI system to discover your ideal career path based on your personality and aptitude scores.

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/jjjulianleon/ProyectoFinalIA.git
cd ProyectoFinalIA

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Set Up Environment

Create a `.env` file with your OpenAI API key (optional, but recommended for AI insights):

```bash
OPENAI_API_KEY=your_api_key_here
```

### 3. Generate Dataset (First Time Only)

```bash
python src/data/generate_sample_data.py
```

### 4. Preprocess Data

```bash
python src/data/preprocess.py
```

### 5. Train Models

```bash
python src/models/train.py
```

### 6. Launch Web Application

```bash
streamlit run web/app.py
```

The application will open in your browser at `http://localhost:8501`

---

## Using the Web Application

### Step 1: Enter Your Profile

On the left sidebar, you'll find two sections:

#### 🧠 Personality Traits (1-10 scale)

These are based on the OCEAN (Big Five) personality model:

1. **Openness to Experience (1-10)**
   - How curious and creative you are
   - High: Creative, imaginative, open to new ideas
   - Low: Practical, conventional, prefer routine

2. **Conscientiousness (1-10)**
   - How organized and responsible you are
   - High: Organized, disciplined, detail-oriented
   - Low: Spontaneous, flexible, adaptable

3. **Extraversion (1-10)**
   - How social and energetic you are
   - High: Outgoing, talkative, energetic
   - Low: Reserved, quiet, introspective

4. **Agreeableness (1-10)**
   - How cooperative and compassionate you are
   - High: Friendly, empathetic, helpful
   - Low: Competitive, skeptical, direct

5. **Neuroticism (1-10)**
   - Your emotional sensitivity
   - High: Sensitive, aware of emotions
   - Low: Calm, emotionally stable

#### 📊 Aptitude Scores (0-10 scale)

These measure your specific abilities:

1. **Numerical Aptitude (0-10)**
   - Mathematical and quantitative reasoning
   - Important for: Engineering, Data Science, Accounting

2. **Spatial Aptitude (0-10)**
   - Visualizing and manipulating objects mentally
   - Important for: Design, Architecture, Engineering

3. **Perceptual Aptitude (0-10)**
   - Pattern recognition and attention to detail
   - Important for: Quality Control, Research, Design

4. **Abstract Reasoning (0-10)**
   - Logical thinking and problem-solving
   - Important for: Programming, Research, Analysis

5. **Verbal Reasoning (0-10)**
   - Language and communication skills
   - Important for: Teaching, Marketing, Management

### Step 2: Get Your Predictions

Click the **"🔮 Predict My Career Path"** button to analyze your profile.

### Step 3: Understand Your Results

The application shows several insights:

#### 📈 Top Career Predictions

- **Career Rankings**: Your top 5 predicted careers
- **Probability Scores**: Confidence level for each prediction
- **Visual Chart**: Interactive bar chart showing all predictions

#### 👤 Your Profile

- **Radar Chart**: Visual representation of your strengths
- Helps you understand your unique profile

#### 🤖 AI-Powered Insights (if OpenAI is configured)

Three tabs provide detailed information:

1. **Personalized Advice**
   - Why these careers match your profile
   - Actionable steps to prepare
   - Encouraging guidance

2. **Top Career Details**
   - Detailed description of your #1 predicted career
   - Key responsibilities
   - Required skills
   - Work environment

3. **Explanation**
   - Why the AI predicted this career for you
   - Connection between your strengths and the career

#### 🔍 Feature Importance

- Shows which factors most influenced your predictions
- Helps you understand what drives the results

---

## Understanding Your Results

### Probability Scores

- **70-100%**: Very strong match - this career aligns well with your profile
- **50-70%**: Good match - worth serious consideration
- **30-50%**: Moderate match - might be suitable with skill development
- **Below 30%**: Lower match - may require significant adaptation

### Multiple High Scores

It's normal to have several careers with similar probabilities! This means:
- You have a versatile profile
- Multiple career paths could suit you
- Consider additional factors like personal interests and values

---

## Tips for Best Results

### 1. Be Honest

Answer based on who you really are, not who you think you should be. The ML model works best with authentic data.

### 2. Take Your Time

Consider each question carefully. Think about how you typically behave and perform.

### 3. Use Real Assessments

If possible, take actual personality and aptitude tests:
- **Personality**: 16Personalities.com, Truity.com
- **Aptitude**: yourfreecareertest.com, 123test.com

### 4. Try Different Profiles

Experiment with adjusting your scores slightly to see how results change. This helps you understand which factors are most important.

### 5. Combine with Research

Use predictions as a starting point, then:
- Research the suggested careers
- Talk to people in those fields
- Consider internships or job shadowing

---

## Troubleshooting

### Application Won't Start

```bash
# Make sure all dependencies are installed
pip install -r requirements.txt

# Verify models are trained
python src/models/train.py
```

### No AI Insights Showing

- Check that your `.env` file contains a valid OpenAI API key
- Verify the key is active and has credits
- AI insights are optional - predictions still work without them

### Predictions Seem Inaccurate

- Ensure you've provided all 10 features
- Double-check your input values
- Remember: ML predictions are probabilistic, not deterministic
- Consider this as guidance, not absolute truth

### Model Performance Issues

If running slowly:
- Close other applications
- Use a smaller dataset for testing
- Consider using the Logistic Regression model instead of Random Forest

---

## Understanding the ML Models

### Random Forest (Primary Model)

- **Accuracy**: ~55% on test data
- **Strengths**: Handles complex patterns, robust
- **How it works**: Combines predictions from multiple decision trees

### Logistic Regression (Baseline Model)

- **Accuracy**: ~45% on test data
- **Strengths**: Fast, interpretable
- **How it works**: Linear relationships between features and careers

### Why Not 100% Accurate?

Career choice involves many factors our model doesn't capture:
- Personal interests and passions
- Life experiences and background
- Cultural and family influences
- Economic opportunities
- Personal values and goals

**Use this tool as one input among many for career exploration!**

---

## Privacy and Data

### What Data is Stored?

- Your inputs are NOT saved to any database
- Each session is independent
- No personal information is collected

### OpenAI API Usage

If using AI features:
- Your profile data is sent to OpenAI for analysis
- OpenAI's privacy policy applies
- No permanent storage of your data

---

## Next Steps After Getting Predictions

1. **Research Careers**
   - Look up job descriptions
   - Check salary ranges
   - Understand growth opportunities

2. **Explore Education Paths**
   - What degrees are required?
   - What skills should you develop?
   - Are certifications needed?

3. **Gain Experience**
   - Internships
   - Volunteer work
   - Side projects
   - Informational interviews

4. **Talk to Professionals**
   - LinkedIn networking
   - Career fairs
   - Alumni connections
   - Professional associations

5. **Keep Learning**
   - Online courses (Coursera, edX, Udemy)
   - Books and articles
   - Workshops and seminars
   - Hands-on practice

---

## FAQ

**Q: Can I use this for career changes?**
A: Yes! The model works for students and professionals exploring new paths.

**Q: How often should I retake the assessment?**
A: Annually, or when you've developed new skills or had significant experiences.

**Q: What if my dream career has a low score?**
A: Scores indicate natural fit, not possibility. With dedication, you can develop needed skills.

**Q: Can I trust these predictions?**
A: Use them as guidance, not gospel. Combine with personal reflection, research, and advice from mentors.

**Q: Is my data shared or sold?**
A: No. Nothing is stored beyond your current session.

---

## Support

For issues, questions, or feedback:
- Check the GitHub Issues: https://github.com/jjjulianleon/ProyectoFinalIA/issues
- Read the documentation in the `docs/` folder
- Contact the development team

---

## Credits

Developed by Universidad & Virginia Tech students as part of a COIL (Collaborative Online International Learning) project.

**Technologies:**
- Python, scikit-learn, pandas
- Streamlit
- OpenAI GPT-3.5
- Model Context Protocol (MCP)

---

**Happy Career Exploring! 🚀**
