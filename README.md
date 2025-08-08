# House Price Prediction - Production ML System

## Project Overview
Advanced machine learning system for house price prediction across multiple datasets (e.g., California, Ames). Demonstrates production-ready ML engineering with comprehensive model comparison, feature engineering, and deployment capabilities.

## Key Features
- Multi-dataset analysis and comparison
- Advanced feature engineering pipeline
- 8+ regression algorithm comparison
- SHAP-based model interpretability
- Interactive Streamlit web application
- REST API with FastAPI
- Comprehensive testing and documentation

## Project Status
🚧 **In Development** - Building professional ML portfolio project

## Requirements & Setup

### Prerequisites
- Python 3.11.0+
- Git

### Environment Setup
```bash
# 1. Clone repository
git clone https://github.com/adsha27/house-price-prediction.git
cd house-price-prediction

# 2. Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install dependencies
pip install -r requirements.txt

# 5. Verify installation
python -c "import pandas, sklearn, xgboost; print('✅ Environment ready!')"

### Data Setup
This project uses two primary datasets: California Housing (included with scikit-learn) and Ames Housing. The Ames Housing dataset must be downloaded manually.

1.  Download the training data from the [Kaggle "House Prices - Advanced Regression Techniques" competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data).
2.  Place this file in the `data/raw/` directory and rename it to `AmesHousing.csv`.

## Technology Stack
- **ML/Data:** Python, scikit-learn, XGBoost, LightGBM, CatBoost
- **Analysis:** Pandas, NumPy, SHAP, Matplotlib, Seaborn
- **Deployment:** Streamlit, FastAPI, Docker
- **Testing:** Pytest, Type hints with mypy

## Business Impact
Target: Demonstrate $500K+ annual impact potential through accurate price predictions and actionable insights.
