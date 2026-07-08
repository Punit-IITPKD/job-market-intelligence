# 💼 Job Market Intelligence Dashboard

An interactive **Machine Learning-powered Streamlit dashboard** for exploring job market trends, analyzing salary insights, identifying skill gaps, and predicting suitable job roles using real-world LinkedIn job posting data.

The project combines **data analytics**, **interactive visualizations**, and **machine learning** to help users understand current hiring trends and make informed career decisions.

---

## 🌐 Live Demo

🔗 https://job-market-intelligence-l5yuzfofbkw967r2hlxelm.streamlit.app/

---

## 📸 Dashboard Preview

### 🏠 Home Dashboard
- Market overview with key performance indicators
- Top hiring job roles
- Top hiring locations
- Salary distribution analysis
- Experience-level insights

### 🔍 Skill Gap Analyzer
- Compare your current skills with any target job role
- Skill match percentage gauge
- Missing skills identification
- Average salary for the selected role

### 🤖 Job Role Predictor
- Predict the most suitable job role using a trained Machine Learning model
- Confidence score visualization
- Top prediction probabilities

---

# ✨ Features

### 📊 Market Analytics
- Total job postings
- Hiring companies
- Remote job percentage
- Average salary
- Top hiring roles
- Top hiring locations
- Experience level distribution
- Salary distribution with percentile analysis

### 🔍 Skill Gap Analysis
- Select a target job role
- Compare your skills with industry requirements
- View matched and missing skills
- Skill match percentage
- Estimated average salary for the selected role

### 🤖 Machine Learning Prediction
- Random Forest Classifier
- Predicts suitable job roles based on user inputs
- Confidence score visualization
- Top prediction probabilities

---

# 🛠️ Tech Stack

### Programming Language
- Python

### Data Analysis
- Pandas
- NumPy

### Machine Learning
- Scikit-learn
- Joblib

### Data Visualization
- Plotly

### Dashboard
- Streamlit
- Streamlit Option Menu

---

# 📂 Dataset

The dashboard uses a cleaned LinkedIn Job Postings dataset containing:

- **123,317 Job Postings**
- Company Information
- Job Titles
- Salary Information
- Experience Levels
- Work Type
- Remote Availability
- Required Skills

The dataset was cleaned, transformed, and enriched before analysis.

---

# 🧠 Machine Learning Model

### Algorithm
- Random Forest Classifier

### Features Used
- Skills
- Experience Level
- Work Type
- Remote Availability

### Output
- Predicted Job Role
- Prediction Confidence

---

# 📈 Dashboard Pages

## 🏠 Home

Provides an executive overview of the current job market.

Includes:

- KPI Cards
- Market Overview
- Salary Insights
- Experience Insights
- Interactive Charts

---

## 🔍 Skill Gap Analyzer

Allows users to compare their existing skills with the most commonly required skills for a target role.

Outputs:

- Skill Match Percentage
- Missing Skills
- Existing Skills
- Average Salary

---

## 🤖 Job Predictor

Predicts the most suitable job role using a trained machine learning model.

Outputs:

- Predicted Role
- Confidence Score
- Top Prediction Probabilities

---

# 📁 Project Structure

```text
Job_market_intelligence_project
│
├── app
│   └── app.py
│
├── src
│   ├── analytics.py
│   └── predict.py
│
├── models
│   ├── job_predictor.pkl
│   ├── label_encoder.pkl
│   └── feature_columns.json
│
├── data
│   ├── cleaned_postings.csv
│   └── job_skills_clean.csv
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/Punit-IITPKD/Job-market-intelligence.git
```

Move into the project directory

```bash
cd job-market-intelligence
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app/app.py
```

---

# 📊 Libraries Used

- Streamlit
- Plotly
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit Option Menu

---

# 🎯 Future Improvements

- Advanced filtering by location, experience level, and work type
- Job recommendation engine
- Resume skill extraction
- Salary forecasting
- Real-time job market updates
- User authentication
- Cloud database integration

---

# 👨‍💻 Developer

**Punit Yadav**

B.Tech Student | Data Science & Machine Learning Enthusiast

GitHub:
https://github.com/Punit-IITPKD

---

# ⭐ If you found this project helpful, consider giving it a star!