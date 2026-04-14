# Food Delivery Time Prediction Using AdaBoost Regressor with Hyperparameter Tuning

## Problem Statement
Food delivery platforms need to accurately predict delivery time to improve
customer satisfaction and operational efficiency. Delivery time is influenced
by multiple real-world factors such as distance, traffic level, weather
conditions, preparation time, vehicle type and partner ratings.

This project uses AdaBoost Regressor — a powerful sequential boosting
ensemble method — to predict food delivery time (in minutes) on a dataset
of 1000 records. Unlike Random Forest which builds trees in parallel,
AdaBoost builds trees sequentially, each focusing on the errors made by
the previous model — resulting in a stronger and more accurate predictor.

## Objective
- Predict food delivery time (in minutes) using AdaBoost Regressor
- Perform Feature Engineering to extract 4 new meaningful features
- Apply One-Hot Encoding on categorical features
- Follow correct ML pipeline — Encoding → Split → Model
- Improve model performance using GridSearchCV Hyperparameter Tuning
- Evaluate using MAE, MSE, RMSE and R² Score
- Compare Before and After tuning performance

## Dataset
| Detail | Info |
|---|---|
| Name | Food Delivery Dataset |
| Records | 1,000 |
| Features | 9 |
| Target | Delivery_Time_min |
| Missing Values | None |

### Features
| Feature | Type | Description |
|---|---|---|
| Distance_km | Numerical | Delivery distance in km |
| Order_Time | Numerical | Hour of order placement |
| Traffic_Level | Categorical | Low / Medium / High |
| Weather | Categorical | Clear / Cloudy / Rainy |
| Restaurant_Rating | Numerical | Rating of restaurant |
| Prep_Time_min | Numerical | Food preparation time |
| Delivery_Partner_Rating | Numerical | Rating of delivery partner |
| Vehicle_Type | Categorical | Bike / Scooter / Bicycle |

## Tech Stack
| Tool | Usage |
|---|---|
| Python | Programming Language |
| Pandas | Data Manipulation |
| NumPy | Numerical Operations |
| Scikit-learn | ML Model & Evaluation |
| Jupyter Notebook | Development Environment |

## ML Pipeline
```
1. Import Libraries
2. Load Dataset
3. Data Understanding
4. Data Preprocessing
   - Drop Order_ID column
   - Feature Engineering (4 new features)
   - One-Hot Encoding (get_dummies, drop_first=True, dtype=int)
5. Model Building
   - Split Features & Target
   - Train-Test Split (80/20)
   - Model Before Tuning (default AdaBoost)
   - GridSearchCV Hyperparameter Tuning
   - Model After Tuning (best params)
6. Evaluation & Prediction Report
```

## Feature Engineering
| New Feature | Formula | Purpose |
|---|---|---|
| Distance_x_Traffic | Distance × Traffic(1/2/3) | Combined impact of distance & traffic |
| Rating_Diff | Restaurant - Partner Rating | Quality gap between restaurant & partner |
| Total_Time_Est | Distance×5 + Prep Time | Rough estimated delivery time |
| Rush_Hour | 1 if 6PM-10PM else 0 | Peak hour flag |

## Results

| Metric | Before Tuning | After Tuning |
|---|---|---|
| **R² Score** | 90.29% | **94.64%** |
| **MAE** | 4.836 | **3.666**  |
| **MSE** | 35.917 | **19.831**  |
| **RMSE** | 5.993 | **4.453**  |

### Best Parameters Found
| Parameter | Value |
|---|---|
| estimator | DecisionTreeRegressor(max_depth=7) |
| learning_rate | 0.5 |
| loss | square |
| n_estimators | 200 |

## Key Insights
- R² improved from **90.29% → 94.64%** after tuning 
- All 4 metrics improved significantly after tuning 
- MAE reduced from **4.836 → 3.666** 
- MSE reduced from **35.917 → 19.831** (massive reduction!) 
- RMSE reduced from **5.993 → 4.453** 
- R² improvement of **+4.35%** — outstanding for boosting model 
- max_depth=7 captured complex delivery patterns 
- loss=square penalizes larger errors — better for regression 
- Average prediction error of only **3.67 minutes** — production ready! 

## Project Structure
```
Food-Delivery-Time-Prediction-AdaBoost-Regressor/
│
├── Adaboost_Regressor_Project.ipynb    # Main Jupyter Notebook
├── food_delivery_dataset.csv           # Dataset
└── README.md                           # Project Documentation
```

## How to Run

1. Clone the repository
```bash
git clone https://github.com/yourusername/Food-Delivery-Time-Prediction-AdaBoost-Regressor.git
```

2. Install dependencies
```bash
pip install pandas numpy scikit-learn
```

3. Open Jupyter Notebook
```bash
jupyter notebook Adaboost_Regressor_Project.ipynb
```

##  Why AdaBoost Regressor?

| | Decision Tree | Random Forest | AdaBoost |
|---|---|---|---|
| Type | Single | Bagging | Boosting  |
| Trees | 1 | Parallel | Sequential |
| Focus | All samples | Random samples | High-error samples |
| R² Score | ~97% | ~98% | **94.64%** |
| Improvement | Small | Small | **+4.35%** |

## Conclusion
The AdaBoost Regressor successfully predicted food delivery time with
an outstanding R² Score of **94.64%** after hyperparameter tuning —
a significant improvement from the **90.29%** baseline (+4.35%).

Feature Engineering added powerful interaction features like
Distance_x_Traffic and Total_Time_Est that helped the model capture
real-world delivery patterns. GridSearchCV found the optimal
combination of n_estimators=200, learning_rate=0.5, loss=square
and base estimator with max_depth=7.

With an average prediction error of only **3.67 minutes**, this model
is production-ready for real-world food delivery platforms.

## 🙋‍♂️ Author
**Vishal Sukale**

