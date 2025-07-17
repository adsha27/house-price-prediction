# Project: House Price Prediction System 🏠

## 1. Project Vision & Goals
* **Core Purpose:** Develop a **production-ready Machine Learning data pipeline** for house price prediction, suitable for deployment and robust operation.
* **Overall Goal:** Through hands-on, challenging problem-solving, the developer (user of this AI) aims to evolve into a **hire-ready Senior ML Engineer** capable of building and deploying complex ML systems, demonstrating skills sought by top-tier companies (FAANG level, high-paying remote positions).
* **Project Structure:** This is one of 9 increasingly complex ML projects, emphasizing systematic, documentation-driven learning with FAANG-level standards.

---

## 2. Current Project Status & Focus

### **✅ PHASE 1: DEPENDENCY MANAGEMENT MASTERY (Completed)**
* **Initial Challenge:** Predicting and resolving ML package conflicts before installation.
* **Key Learning:** Successfully researched `scikit-learn`, `pandas`, `matplotlib`, `xgboost` `numpy` requirements, confirming all compatible with `numpy 1.24.0+`.
* **Problem Solved:** Transitioned from manual dependency management to professional lockfiles using `pip-tools`.
* **Resolved Dependencies:** `numpy==1.26.4`, `pandas==2.3.0`, `scikit-learn==1.7.0`, `xgboost==2.1.4`.
* **Realization:** Understood the critical difference between abstract (`requirements.in`) and concrete (`requirements.txt`) dependency management.

### **✅ PHASE 2: DEVELOPMENT ENVIRONMENT EXCELLENCE (Completed)**
* **Focus:** Implementing automated code quality enforcement.
* **Key Learning:** Pre-commit hooks run *before* code enters the repository, preventing "works on my machine" issues.
* **Tools Integrated:** `black` (formatting), `flake8` (linting), `mypy` (type checking) via `pre-commit`.
* **Problem Solved:** Overcame initial conflicting tool configurations (e.g., `autopep8` vs. `black`) by simplifying to focused configurations relevant for ML projects.
* **Outcome:** All pre-commit checks passed on all files.

### **🔄 PHASE 3: DATA PIPELINE ARCHITECTURE (IN PROGRESS)**
* **Immediate Goal:** Implement robust and memory-efficient data loading and processing for multiple datasets.
* **Current Focus:** Implementing the `CaliforniaHousingLoader` class and its corresponding tests.
* **Key Realizations for Dataset Loading:**
    * **Target Datasets:** California Housing, Ames Housing. (Boston Housing dataset is **excluded** due to ethical concerns and deprecation).
    * **Loader Diversity:** California Housing is directly `sklearn.datasets` compatible (`as_frame=True`). Ames Housing is a CSV download with a different structure (79 features vs 8), requiring a distinct, more complex loading strategy.
    * This necessitates different loading approaches and careful schema standardization for multi-dataset analysis.
* **Research Completed:**
    * **Chunking:** Processing large datasets in memory-efficient batches (e.g., 50GB data → 25 × 2GB chunks, leaving 14GB RAM buffer).
    * **Factory Pattern:** For dynamic loader selection based on data source (file extension, URL pattern, config).
    * **Abstract Base Classes (ABCs):** For defining common interfaces (`__len__`, `__getitem__`, `__iter__`, `__next__`).
    * **Memory Profiling:** Using `@profile` decorator for optimization.
* **Critical Learning Moments:**
    * Correctly identified initial **over-engineering** in architecture design ("I have no idea what the data inputs are going to be...").
    * Understood ABC redundancy when all fields are abstract; pivoted to **shared concrete functionality with targeted abstractions** (concrete methods for common CSV loading, basic validation; abstract for dataset-specific requirements like target columns, schema validation).
* **Project Scope Revelation (Confirmed):** This is a complex production system, not a simple tutorial, requiring:
    * **Technical Implementation:**
        * **Phased Dataset Integration:** First California Housing, then Ames Housing. **Future task: Integrate both datasets to achieve the best possible performing models.**
        * 8+ ML models: Linear, Ridge, Lasso, Elastic Net, Random Forest, Gradient Boosting, XGBoost, LightGBM.
        * Advanced feature engineering: polynomial features, interaction terms, geographic clustering.
        * Cross-validation with time-series awareness.
        * Automated hyperparameter optimization (Optuna/Hyperopt).
        * Model interpretability with SHAP values.
    * **Production Elements:**
        * Interactive Streamlit application.
        * Model comparison dashboard.
        * Automated model selection.
        * Geographic visualization.
        * API endpoints.
        * Confidence intervals.
    * **GitHub Excellence:**
        * Comprehensive EDA notebooks.
        * Modular code structure.
        * Automated testing suite.
        * CI/CD pipeline.
        * Performance benchmarking.
* **Current Implementation Status:**
    * `CaliforniaHousingLoader` class implemented.
    * Initial `pytest` cases created (`test_load_returns_dataframe()`, `test_dataframe_not_empty()`).
    * Resolved "Import sklearn.datasets could not be resolved" by understanding that `requirements.txt` lists but doesn't install; manual `pip install -r requirements.txt` was needed.

### **⏳ Pending Phases:**
* **Phase 3 Completion:** Implement remaining `CaliforniaHousingLoader` tests (column structure, missing values), add error handling and logging, then proceed to Ames Housing loader and factory integration. ETL implementation and performance optimization.
* **Phase 4:** Data validation framework with schema validation.
* **Subsequent Phases:** Advanced Feature Engineering, Model Development (training, optimization, interpretability), Model Deployment (Streamlit app, API endpoints, CI/CD).

---

## 3. Repository Structure & Key Files
house-price-prediction/
├── .python-version (Python 3.11.0)
├── requirements.in                    # Abstract direct dependencies (managed by pip-tools)
├── requirements.txt                   # Locked production-ready dependencies (200+ packages, generated by pip-compile)
├── .pre-commit-config.yaml            # Configuration for pre-commit hooks (black, flake8, mypy, etc.)
├── README.md                          # Project setup instructions
├── .gitignore                         # Configured for Python/ML project specific exclusions
├── .env.example                       # Template for environment variables
├── data/
│   ├── raw/                           # Raw, immutable datasets
│   ├── processed/                     # Cleaned, transformed datasets
│   └── external/                      # Third-party or external data
├── notebooks/
│   ├── 01_data_exploration/           # Jupyter notebooks for Exploratory Data Analysis (EDA)
│   ├── 02_feature_engineering/        # Jupyter notebooks for feature creation/selection
│   └── 03_modeling/                   # Jupyter notebooks for model experimentation
├── src/                               # Main source code for production features
│   ├── init.py
│   ├── data/
│   │   ├── init.py
│   │   ├── base.py                    # Abstract base classes for data loaders
│   │   ├── loaders.py                 # Concrete loader implementations (e.g., CaliforniaHousingLoader)
│   │   ├── pipeline.py                # ETL pipeline logic (Extract, Transform, Load)
│   │   └── factory.py                 # Factory for dynamic data loader selection
│   ├── features/                      # Modules for feature engineering logic
│   │   └── init.py
│   ├── models/                        # Modules for ML model implementations
│   │   └── init.py
│   └── utils/                         # Common utility functions
│       └── init.py
├── tests/                             # Automated test suite
│   ├── init.py
│   └── test_loaders.py                # Specific tests for data loaders (e.g., CaliforniaHousingLoader)
├── docs/                              # Project documentation and guides
├── models/                            # Serialized trained models
├── app/                               # Deployment-related files (e.g., API entry points, Streamlit app)
└── venv/                              # Local virtual environment directory


---

## 4. Core Technology Stack
* **Primary Language:** Python 3.11.0
* **Package Management:** `pip`, `pip-tools` (`pip-compile`)
* **Core ML/Data Science Libraries (Planned/Used):** `numpy==1.26.4`, `pandas==2.3.0`, `scikit-learn==1.7.0`, `xgboost==2.1.4`, `matplotlib`.
* **Code Quality Tools:** `pre-commit` (`black`, `flake8`, `mypy`).
* **Architectural Patterns:** Abstract Base Classes (ABCs), Factory Pattern.
* **Performance:** `memory-profiler` (`@profile` decorator).
* **Testing:** `pytest`.
* **Upcoming (Planned):** `Optuna`/`Hyperopt` (hyperparameter optimization), `SHAP` (model interpretability), `Streamlit` (interactive applications), CI/CD tooling.

---

## 5. Professional Standards & Best Practices (Desired for this Project)
* **Overall Quality:** All solutions and code must aim for **professional, senior-level quality** suitable for top-tier companies (FAANG). "Good enough" is not acceptable.
* **Learning Methodology:** Strictly adhere to the **systematic discovery process: predict → research → implement → verify**. This fosters independent problem-solving and anticipatory thinking.
* **Documentation-First Approach:** Solutions must be based on and explicitly reference **official documentation, PEPs, and industry best practices**. Mastery in navigating technical documentation is paramount.
* **Ethical Data Usage:** **Avoid datasets with known ethical issues or deprecation warnings** unless specifically for educational purposes on those issues. The Boston Housing dataset is therefore **excluded** from this project.
* **Development Workflow:**
    * **Local-First Development:** Prioritize local VS Code development over cloud-based notebooks (like Colab) to build robust dependency management, Git workflows, and deployment skills crucial for production ML.
    * **Notebook Integration:** Use Jupyter notebooks (in VS Code) for initial EDA, visualization, and rapid experimentation. **Crucially, move proven code from notebooks back to `.py` files** for modularity, testability, and production readiness.
    * **Test-Driven Development (TDD):** Implement a strict TDD cycle: **write failing test → implement code → make test pass**. Utilize `pytest` for all testing.
* **Dependency Management:**
    * Strict separation between development (`requirements.in`) and production (`requirements.txt`) dependencies using `pip-tools` (`pip-compile`).
    * Production environments **must be reproducible** via locked `requirements.txt`.
    * Deep understanding of `pip`'s dependency resolver, PEP 440, and PyPA best practices, with the ability to predict and resolve conflicts independently.
* **Automated Code Quality Enforcement:**
    * Mandatory use of **pre-commit hooks** for automated checks *before* code enters the repository.
    * Adherence to specified formatting (`black`), linting (`flake8` with `--max-line-length=88`), and type-checking (`mypy`) standards.
    * Understanding tool conflicts and importance of focused configurations.
* **Software Architecture & Design:**
    * Strategic use of patterns like **Abstract Base Classes** (with shared concrete functionality) and **Factory Patterns** for modularity and extensibility.
    * **Avoiding over-engineering** by grounding architectural design in actual project requirements, not just theoretical concepts.
    * Emphasis on **modular, testable code structure**.
    * **Logging:** Implement logging only for operations that can fail (e.g., network calls, file operations), not for every function.
* **Data Pipeline Excellence (Current Focus):**
    * **Phased Dataset Integration:** Implement California Housing loader first, then Ames Housing loader. **Future focus includes integrating both datasets to achieve the best possible combined model performance.**
    * Efficient processing of large datasets using **chunking strategies**.
    * Memory optimization through **selective memory profiling** (`@profile` decorator only on data loading and heavy processing).
    * Robust data loading with dynamic loader selection, specifically handling diverse dataset sources (e.g., `sklearn.datasets` vs. CSV downloads).
* **Data Validation:** Implementation of a robust data validation framework with schema validation (upcoming).
* **Testing Strategy:** Comprehensive automated testing suite (unit, integration, end-to-end), with `pytest` as the standard framework.
* **Performance Benchmarking:** Essential for ensuring solutions meet production demands.
* **CI/CD Pipeline:** Automated pipelines for continuous integration and deployment (future phase).
* **Model Development:** Advanced techniques like hyperparameter optimization, cross-validation (including time-series awareness), and model interpretability.
* **Production Deployment:** Building interactive applications, API endpoints, and dashboards.

---

## 6. AI Assistant (Gemini Code Assist / CLI) Interaction Guidelines

**When interacting with this AI assistant (you, the user), its responses must strictly adhere to a "Tough Love Learning" philosophy to foster independent senior-level ML engineering skills:**

* **Core Role:** The AI's primary function is to act as a highly knowledgeable **technical mentor and guide**, focusing on developing the user's independent problem-solving and research capabilities.
* **Forbidden Behaviors:**
    * **DO NOT** provide direct code solutions.
    * **DO NOT** solve problems for the user.
    * **DO NOT** accept "good enough" solutions; push for excellence and FAANG-level standards.
    * **DO NOT** hand-hold through basic concepts; expect and direct towards independent research.
    * **DO NOT** give answers without explicitly making the user research.
* **Required Behaviors (Guided Assistance):**
    * **Always:** Present **challenging problems** that require deep thought and independent investigation (akin to what a senior engineer would face).
    * **Always:** Point to **specific, official documentation, authoritative resources, or relevant PEPs/standards** (e.g., PyPA docs, PEPs, official library guides).
    * **Always:** Demand **independent research** into specific concepts, issues, or solutions.
    * **Always:** **Challenge assumptions** and push for deeper understanding ("A senior engineer would investigate...").
    * **Always:** Require **predictions** from the user regarding outcomes, conflicts, or solutions *before* suggesting an implementation or confirming a hypothesis.
    * **Always:** **Critically analyze** the user's statements and guide them towards the logically better, more professional, and robust option, even if it contradicts their initial thought or approach.
* **Focus Areas for AI Guidance:** Understanding concepts, debugging methodologies (not direct fixes), environment setup, configuration of professional tools, adherence to established coding/architectural standards, strategic problem-solving, and efficient knowledge acquisition.