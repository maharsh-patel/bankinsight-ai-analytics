# BankInsight AI — Customer Term Deposit Subscription Analytics & AI Assistant

BankInsight AI is an end-to-end data analytics, machine learning, and AI-assisted decision platform built on the UCI Bank Marketing dataset (41,172 client records). It integrates automated profiling, data cleaning, an analytical SQLite engine with 12 SQL query modules, target-leakage-free machine learning pipelines, explainable AI feature importances, a context-grounded LLM Q&A engine with zero-hallucination fallback, and an interactive single-page web dashboard.

---

## 1. Project Overview

BankInsight AI empowers financial marketing managers to optimize direct telemarketing campaigns for term deposit subscriptions. The platform transforms raw customer and campaign interaction data into actionable segment insights, predictive propensity scores, and conversational AI guidance.

---

## 2. Problem Statement

Direct telemarketing campaigns incur high operational call costs, low baseline conversion rates (**11.27%**), and severe class imbalance (**7.9:1** non-subscribers per subscriber). Calling all clients indiscriminately wastes resources and alienates leads. Furthermore, naive machine learning approaches suffer from target leakage by including call `duration` (a variable unknown prior to making a call). BankInsight AI addresses these challenges by identifying high-conversion customer profiles, enforcing target leakage prevention, and delivering predictive lead-scoring models.

---

## 3. Objectives

- **Automated Profiling & Quality Audit**: Inspect 21 raw columns and generate schema dictionaries.
- **Data Cleaning**: Remove duplicate records, drop non-contact noise (`duration = 0`), and standardize categorical values.
- **Exploratory Data Analysis (EDA)**: Identify demographic and campaign conversion drivers.
- **SQL Analytics Engine**: Recreate a relational SQLite database (`bankinsight.db`) and run 12 analytical SQL queries.
- **Leakage-Free Feature Engineering**: Exclude `duration` and collinear macro indicators (`emp.var.rate`, `nr.employed`, `cons.price.idx`).
- **Supervised ML Classification**: Train and compare Baseline, Logistic Regression, and Random Forest models.
- **Explainable AI (XAI)**: Quantify top predictive features influencing customer subscription decisions.
- **Grounded AI Q&A Engine**: Provide natural-language answers backed strictly by validated analytical findings with automatic rule-based fallback when offline.
- **Interactive Web Dashboard**: Serve an executive web dashboard with dynamic charts and an embedded AI Chat Assistant.

---

## 4. Dataset & Source

- **Dataset**: UCI Bank Marketing Dataset (v2 / `bank-additional-full.csv`)
- **Origin**: Direct marketing campaigns of a Portuguese banking institution (2008–2013).
- **Primary Source File**: `bank_marketing.csv`

---

## 5. Dataset Size

- **Raw Dataset**: 41,188 rows × 21 columns
- **Cleaned Dataset**: 41,172 rows × 22 columns
  - **12 duplicate row pairs dropped** (identical records appearing twice)
  - **4 non-contact call records dropped** (`duration == 0`)

---

## 6. Features

| Category | Feature Name | Dtype | Description |
| :--- | :--- | :--- | :--- |
| **Client** | `age` | Integer | Client age in years |
| | `job` | String | Type of job (12 categories) |
| | `marital` | String | Marital status (`married`, `single`, `divorced`, `unknown`) |
| | `education` | String | Highest education level (8 categories) |
| | `default` | String | Has credit in default? (`yes`, `no`, `unknown`) |
| | `housing` | String | Has housing loan? (`yes`, `no`, `unknown`) |
| | `loan` | String | Has personal loan? (`yes`, `no`, `unknown`) |
| **Campaign** | `contact` | String | Communication mode (`cellular`, `telephone`) |
| | `month` | String | Last contact month (`mar`–`dec`) |
| | `day_of_week` | String | Last contact day of week (`mon`–`fri`) |
| | `duration` | Integer | Last contact duration in seconds (*Excluded from ML features*) |
| | `campaign` | Integer | Number of contacts during this campaign |
| | `pdays` | Integer | Days since previous contact (`999` = not previously contacted) |
| | `previous` | Integer | Number of contacts before this campaign |
| | `poutcome` | String | Outcome of previous campaign (`success`, `failure`, `nonexistent`) |
| **Macro** | `emp.var.rate` | Float | Employment variation rate (quarterly) |
| | `cons.price.idx` | Float | Consumer price index (monthly) |
| | `cons.conf.idx` | Float | Consumer confidence index (monthly) |
| | `euribor3m` | Float | Euribor 3-month rate (daily) |
| | `nr.employed` | Float | Number of employees (quarterly) |
| **Target** | `y` / `y_encoded` | String/Int | Term deposit subscription (`yes` = 1, `no` = 0) |

---

## 7. Data-Cleaning Methodology

1. **Integrity Validation**: Loaded raw CSV and verified column counts.
2. **Duplicate Removal**: Identified and dropped 12 exact duplicate row pairs.
3. **Zero-Duration Removal**: Filtered 4 rows with `duration == 0` (calls never established).
4. **Data Standardization**: Converted string fields to lowercase, stripped whitespace, and added `y_encoded`.
5. **Sentinel & Flag Engineering**: Preserved `unknown` category labels across categorical features and created `previously_contacted` binary flag (`pdays != 999`).

---

## 8. Exploratory Data Analysis (EDA)

- **Conversion Baseline**: 11.27% overall subscription rate (4,639 subscribers vs 36,533 non-subscribers).
- **High-Converting Job Groups**: `student` (**31.43%**), `retired` (**25.26%**), `unemployed` (**14.20%**).
- **High-Converting Prior Outcome**: `poutcome = success` achieves a **65.11%** conversion rate (5.8x baseline).
- **Seasonal Patterns**: March (**50.55%**) and December (**48.90%**) yield highest conversion rates, whereas May handles 33.43% of total call volume but yields only a **6.44%** conversion rate.

---

## 9. SQL Analytics Engine

The project builds an SQLite relational database (`data/db/bankinsight.db`) and executes 12 SQL scripts in `src/sql/`:

1. `01_schema.sql`: Table structure and indices.
2. `02_overview.sql`: Overall volume, subscription count, and imbalance metrics.
3. `03_conversion_by_job.sql`: Conversion rates grouped by job title.
4. `04_conversion_by_education.sql`: Conversion rates by education level.
5. `05_conversion_by_marital.sql`: Conversion rates by marital status.
6. `06_conversion_by_housing_loan.sql`: Joint analysis of housing and personal loan holders.
7. `07_conversion_by_contact.sql`: Cellular vs. telephone channel comparison.
8. `08_conversion_by_campaign_count.sql`: Contact frequency tiers (`1`, `2-3`, `4-6`, `7+`).
9. `09_conversion_by_poutcome.sql`: Recency and previous campaign outcome breakdown.
10. `10_duration_contact_stats.sql`: Duration distribution buckets (<2 min, 2-5 min, 5-10 min, >10 min).
11. `11_customer_segment_analysis.sql`: Multi-dimensional customer persona segmentation.
12. `12_temporal_patterns.sql`: Monthly and day-of-week conversion trends.

---

## 10. Feature Engineering

- **Target Leakage Exclusion**: `duration` removed from ML input matrix.
- **Collinearity Reduction**: Dropped `emp_var_rate` (r=0.97 with `euribor3m`), `nr_employed` (r=0.95 with `euribor3m`), and `cons_price_idx` (r=0.78 with `emp_var_rate`). Retained `euribor3m` and `cons_conf_idx`.
- **Categorical Binning**: Added `age_group` (6 bands) and `campaign_group` (`first`, `optimal`, `diminishing`, `over_contacted`).
- **Recency Transformation**: Created `pdays_actual` and `recency_group`.
- **One-Hot Encoding**: Transformed string categoricals into numeric indicator columns.

---

## 11. Machine Learning Methodology

- **Task**: Supervised Binary Classification (`y_encoded`).
- **Train/Test Split**: 80% train (32,937 samples), 20% held-out test (8,235 samples), stratified by class.
- **Cross-Validation**: Stratified 5-Fold Cross-Validation on training set (`RANDOM_STATE = 42`).
- **Models**:
  - `M0 DummyClassifier`: Stratified random guessing baseline.
  - `M1 LogisticRegression`: Standardized features, `class_weight='balanced'`, L2 penalty.
  - `M2 RandomForestClassifier`: 100 trees, `max_depth=12`, `class_weight='balanced'`.

---

## 12. Actual Model Evaluation Results

Evaluated on the 8,235 held-out test records:

| Model | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | True Pos (TP) | False Pos (FP) | False Neg (FN) | True Neg (TN) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **M0 Dummy** | 0.1186 | 0.1164 | 0.1175 | 0.5032 | 0.1134 | 108 | 803 | 820 | 6,504 |
| **M1 LogReg** | 0.2826 | **0.7069** | 0.4038 | 0.7820 | 0.4129 | 656 | 1,665 | 272 | 5,642 |
| **M2 RandomForest** | **0.3795** | 0.6584 | **0.4815** | **0.8120** | **0.4778** | 611 | 999 | 317 | 6,308 |

---

## 13. Explainability (XAI)

Feature importances extracted from the primary Random Forest model (`models/rf_pipeline.joblib`):

1. `euribor3m` (**32.4%**): Key macroeconomic driver governing deposit interest rate attractiveness.
2. `pdays_actual` / `poutcome_success` (**18.1%**): Historical campaign responsiveness.
3. `cons_conf_idx` (**12.3%**): Consumer financial sentiment.
4. `age` (**9.5%**): Client age (outlier conversion in 18-25 and 66+ brackets).
5. `campaign` (**7.2%**): Number of contacts in current campaign (steep decline past 3 calls).

---

## 14. AI Functionality & Engine

- **Context-Grounded Q&A**: `src/phase8_ai_engine.py` constructs a factual prompt from `src/phase8_ai_context.py` and sends queries to the OpenAI Chat Completions API.
- **Zero-Hallucination Policy**: Strictly instructed to cite only verified dataset figures.
- **Rule-Based Fallback Engine**: If no API key is set or the network is offline, the AI layer seamlessly switches to a rule-based response engine.

---

## 15. Dashboard Interface

An executive web dashboard (`dashboard/index.html`) driven by `dashboard/dashboard_data.json` and served by `src/phase8_ai_server.py`:
- **KPI Summary Ribbon**: Total clients, total subscriptions, subscription rate, imbalance ratio, call duration, campaign contacts.
- **Interactive ECharts**: Demographic conversion charts, campaign frequency analysis, duration distribution, and model metric comparisons.
- **Segment Filters**: Dynamic dropdown filters for Job, Education, and Marital status.
- **AI Chat Assistant Panel**: Live Q&A panel interfacing with `/api/ask` and `/api/ping`.

---

## 16. Architecture

```
                                  ┌───────────────────────────────┐
                                  │      bank_marketing.csv       │
                                  └──────────────┬────────────────┘
                                                 │
                                                 ▼
                                  ┌───────────────────────────────┐
                                  │   Phase 1: Data Profiling     │
                                  │   Phase 2: Data Cleaning      │
                                  └──────────────┬────────────────┘
                                                 │
                        ┌────────────────────────┴────────────────────────┐
                        ▼                                                 ▼
      ┌───────────────────────────────────┐             ┌───────────────────────────────────┐
      │   Phase 4: SQLite Database        │             │   Phase 5: Feature Engineering    │
      │   data/db/bankinsight.db          │             │   (Target Leakage Excluded)       │
      └─────────────────┬─────────────────┘             └─────────────────┬─────────────────┘
                        │                                                 │
                        ▼                                                 ▼
      ┌───────────────────────────────────┐             ┌───────────────────────────────────┐
      │   Phase 7: Dashboard Data JSON    │             │   Phase 6: ML Pipeline            │
      │   dashboard/dashboard_data.json   │             │   models/rf_pipeline.joblib       │
      └─────────────────┬─────────────────┘             └─────────────────┬─────────────────┘
                        │                                                 │
                        └────────────────────────┬────────────────────────┘
                                                 │
                                                 ▼
                                  ┌───────────────────────────────┐
                                  │   Phase 8: AI Engine & Server │
                                  │   src/phase8_ai_server.py     │
                                  └──────────────┬────────────────┘
                                                 │
                                                 ▼
                                  ┌───────────────────────────────┐
                                  │  Dashboard UI (Port 8050)     │
                                  │  dashboard/index.html         │
                                  └───────────────────────────────┘
```

---

## 17. Tech Stack

- **Core & Logic**: Python 3.14+
- **Data Manipulation**: Pandas, NumPy
- **Database & SQL**: SQLite3
- **Machine Learning**: Scikit-learn, Joblib
- **Visualization**: Matplotlib, Apache ECharts 5.4
- **Testing**: Pytest 9.1
- **AI / LLM Layer**: OpenAI API (`gpt-4o-mini`), Python `dotenv`
- **Serving / Frontend**: Python HTTP Server, HTML5, Vanilla CSS3, JavaScript (ES6)

---

## 18. Project Structure

```
BankInsight AI/
├── .env.example
├── bank_marketing.csv
├── README.md
├── dashboard/
│   ├── dashboard_data.json
│   └── index.html
├── data/
│   ├── db/
│   │   └── bankinsight.db
│   └── processed/
│       ├── bank_marketing_clean.csv
│       └── bank_marketing_features.csv
├── models/
│   ├── feature_list.csv
│   ├── lr_pipeline.joblib
│   └── rf_pipeline.joblib
├── notebooks/
│   └── bankinsight_eda.ipynb
├── reports/
│   ├── phase1/
│   ├── phase2/
│   ├── phase3/
│   ├── phase4/
│   ├── phase5/
│   ├── phase6/
│   ├── phase8/
│   └── phase10/
│       ├── test_audit_report.txt
│       └── test_results.json
├── src/
│   ├── sql/
│   │   ├── 01_schema.sql
│   │   ├── 02_overview.sql
│   │   ├── ...
│   │   └── 12_temporal_patterns.sql
│   ├── phase1_profiling.py
│   ├── phase2_cleaning.py
│   ├── phase3_eda.py
│   ├── phase4_sql_analytics.py
│   ├── phase5_feature_engineering.py
│   ├── phase6_modelling.py
│   ├── phase7_dashboard_data.py
│   ├── phase8_ai_context.py
│   ├── phase8_ai_engine.py
│   ├── phase8_ai_server.py
│   ├── phase8_test.py
│   └── serve_dashboard.py
└── tests/
    ├── __init__.py
    ├── test_ai_integration.py
    ├── test_code_quality.py
    ├── test_dashboard.py
    ├── test_data_pipeline.py
    ├── test_edge_cases.py
    ├── test_ml_pipeline.py
    └── test_sql_analytics.py
```

---

## 19. Installation

1. **Clone or navigate to repository**:
   ```bash
   cd "BankInsight AI"
   ```

2. **Install Python dependencies**:
   ```bash
   pip install pandas numpy scikit-learn joblib matplotlib pytest python-dotenv openai
   ```

---

## 20. Environment Variables

Create a `.env` file in the project root (optional for live AI mode):

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
AI_MAX_TOKENS=1200
AI_TEMPERATURE=0.3
AI_TIMEOUT_SEC=30
AI_MAX_RETRIES=2
```

*Note: If no API key is provided, the platform automatically uses the rule-based fallback engine.*

---

## 21. Run Instructions

To run each pipeline phase sequentially:

```bash
# 1. Run Data Understanding & Profiling
python src/phase1_profiling.py

# 2. Run Data Cleaning Pipeline
python src/phase2_cleaning.py

# 3. Run Exploratory Data Analysis
python src/phase3_eda.py

# 4. Build SQLite Database & Run SQL Analytics
python src/phase4_sql_analytics.py

# 5. Run Feature Engineering (Leakage Excluded)
python src/phase5_feature_engineering.py

# 6. Train & Evaluate Machine Learning Models
python src/phase6_modelling.py

# 7. Build Dashboard Data JSON
python src/phase7_dashboard_data.py

# 8. Test AI Layer Engine
python src/phase8_test.py

# 9. Launch Dashboard & AI Server (Access at http://localhost:8050)
python src/phase8_ai_server.py --port 8050
```

---

## 22. Testing

Run the automated test suite covering all data, SQL, ML, AI, dashboard, edge cases, and code quality:

```bash
python -m pytest -v --tb=short
```

---

## 23. Actual Test Results

```text
=========================== short test summary info ===========================
33 passed, 5 warnings in 12.35s
```

- **Total Tests Run**: 33
- **Passed**: 33 (100%)
- **Failed**: 0
- **Test Modules**:
  - `test_data_pipeline.py`: PASSED (6/6)
  - `test_sql_analytics.py`: PASSED (4/4)
  - `test_ml_pipeline.py`: PASSED (4/4)
  - `test_ai_integration.py`: PASSED (4/4)
  - `test_dashboard.py`: PASSED (4/4)
  - `test_edge_cases.py`: PASSED (7/7)
  - `test_code_quality.py`: PASSED (4/4)

---

## 24. Limitations

1. **Duration Exclusion**: `duration` is excluded from feature set to avoid target leakage, capping maximum attainable F1-score to ~0.48.
2. **Class Imbalance**: Severe 7.9:1 class imbalance limits precision at default 0.5 decision threshold.
3. **No Financial Balance**: The UCI Bank Marketing v2 dataset does not record client bank account balances.

---

## 25. Future Scope

1. **Cost-Sensitive Threshold Optimization**: Optimize classification decision threshold based on specific call-center dollar cost per contact vs term deposit conversion revenue.
2. **Gradient Boosting Models**: Integrate LightGBM and XGBoost models for incremental accuracy gains.
3. **CRM API Integrations**: Build webhooks for direct lead push to CRM systems (e.g., Salesforce, HubSpot).

---

## 26. Dataset Attribution

S. Moro, P. Cortez and P. Rita. *A Data-Driven Approach to Predict the Success of Bank Telemarketing.* Decision Support Systems, Elsevier, 62:22-31, June 2014. Available via the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing).
