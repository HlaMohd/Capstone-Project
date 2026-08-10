# Data Understanding & Cleaning Decisions

## 1. Dataset Overview

The original SWaT dataset contained:

* **Rows before cleaning:** 1,441,719
* **Columns before cleaning:** 53

The dataset contains timestamp information, sensor/actuator measurements, and the target variable `Normal/Attack`.

After removing the `Timestamp` column from the model input and separating the target variable:

* **Number of features:** 44
* **Target variable:** `Normal/Attack`

The target variable is binary:

* `0` = Normal
* `1` = Attack

The `Timestamp` column was retained during the data preparation and temporal analysis stages because it is essential for chronological ordering and time-based evaluation, but it was not used as a predictive feature.

---

## 2. Data Types

After removing the `Timestamp` column from the feature matrix:

* **float64 features:** 24
* **int64 features:** 20
* **Total numerical features:** 44
* **Text/categorical features among model inputs:** 0

Therefore, all features used by the machine learning models are numerical.

This is important because the selected machine learning algorithms can operate directly on the numerical sensor and actuator measurements after the required preprocessing.

---

## 3. Missing Values Analysis

Missing-value analysis identified seven columns with a substantial number of missing observations:

| Column | Missing Values | Percentage |
| ------ | -------------: | ---------: |
| MV101  |        991,800 |     68.79% |
| MV201  |        991,800 |     68.79% |
| P201   |        991,800 |     68.79% |
| P202   |        991,800 |     68.79% |
| AIT201 |        991,800 |     68.79% |
| P204   |        991,800 |     68.79% |
| MV303  |        991,800 |     68.79% |

### Cleaning Decision

These seven columns were removed because each contained approximately **68.79% missing values**.

Keeping variables with such a high proportion of missing observations could introduce substantial uncertainty and reduce the reliability of the resulting feature representation.

Rather than applying aggressive imputation to variables with more than two-thirds of their observations missing, the decision was made to remove these columns.

This resulted in a cleaner and more reliable feature space for subsequent model training.

---

## 4. Duplicate Analysis

Duplicate analysis identified:

* **Duplicate rows:** 494,991

A large number of duplicate rows was observed.

### Interpretation

The duplicate observations were not automatically treated as erroneous records.

The SWaT dataset represents a **time-series industrial control system**, where sensor and actuator values can remain unchanged for multiple consecutive timestamps.

Consequently, identical feature vectors can legitimately occur at different points in time.

For example, if a physical process remains stable, several consecutive observations may contain exactly the same sensor values.

### Cleaning Decision

The duplicate rows were therefore **retained** rather than removed.

This decision preserves:

* the original temporal structure,
* the frequency of stable operating states,
* the distribution of normal and attack observations,
* and the realistic behavior of industrial process measurements.

Removing these rows could artificially change the temporal distribution of the dataset and potentially remove meaningful operational states.

---

## 5. Final Dataset After Cleaning

After the cleaning operations:

* **Final number of rows:** 989,982
* **Final number of columns:** 46

The final dataset consists of:

* **44 numerical features**
* **1 target column:** `Normal/Attack`
* **1 Timestamp column**

Therefore:

**46 columns = 44 features + 1 target + 1 Timestamp**

For machine learning, the input matrix contains:

**44 features**

while `Timestamp` is excluded from the predictive feature matrix.

The timestamp is retained separately when chronological ordering and time-based evaluation are required.

---

## 6. Target Distribution

### Original Dataset Distribution

Before the cleaning process, the target distribution was:

| Class      |       Samples | Percentage |
| ---------- | ------------: | ---------: |
| Normal (0) |     1,387,098 |     96.21% |
| Attack (1) |        54,621 |      3.79% |
| **Total**  | **1,441,719** |   **100%** |

### Observation

The original dataset is **highly imbalanced**.

Only approximately **3.79%** of the observations correspond to attacks, while approximately **96.21%** correspond to normal operation.

This imbalance means that accuracy alone is not sufficient to evaluate the quality of an attack-detection model.

For example, a model that predicts almost every observation as Normal could achieve high accuracy while failing to detect attacks.

Therefore, the evaluation emphasizes:

* **Precision**
* **Recall**
* **F1-score**
* **ROC-AUC**
* **PR-AUC**

PR-AUC is particularly informative in this setting because the positive class (Attack) is relatively rare.

---

## 7. Temporal Nature of the Dataset

The SWaT dataset is a chronological industrial process dataset.

Because observations are collected over time, a random train/test split can produce an overly optimistic estimate of generalization performance.

A random split can place observations from very similar or adjacent operational periods into both the training and testing sets.

To address this issue, a separate chronological evaluation strategy was implemented.

The dataset was sorted according to:

`Timestamp`

and divided chronologically:

* **First 80%:** training data
* **Last 20%:** final test data

This approach better represents a realistic deployment scenario in which the model is trained on historical observations and then used to detect attacks in future observations.

---

## 8. Prevention of Temporal Data Leakage

The final Random Forest model was evaluated using a time-based split rather than relying only on a random stratified split.

The scaler used for the final model was also fitted on the training portion only and subsequently applied to the test portion.

This prevents information from the future test period from being used when learning the preprocessing transformation.

The same principle was followed during model tuning using:

`TimeSeriesSplit`

This provides a more defensible evaluation of model performance on unseen future observations.

---

## 9. Model Selection and Final Random Forest

Several machine learning approaches were investigated during the project:

* Logistic Regression
* Random Forest
* XGBoost
* Neural Network (MLP)

The final deployed model is a tuned Random Forest classifier.

The selected model configuration is:

```text
RandomForestClassifier(
    class_weight="balanced",
    max_features="log2",
    min_samples_leaf=2,
    min_samples_split=5,
    n_jobs=-1,
    random_state=42
)
```

The model was selected after hyperparameter tuning using `RandomizedSearchCV` together with `TimeSeriesSplit`.

The tuning objective used **Average Precision / PR-AUC**, which is more appropriate than accuracy for the highly imbalanced attack-detection problem.

---

## 10. Final Classification Threshold

The default classification threshold of 0.50 was not used as the final operating threshold.

Threshold analysis was performed over values from **0.05 to 0.95**.

The best F1-score was obtained at:

**Final threshold = 0.10**

At this threshold:

* **Precision:** 0.9609
* **Recall:** 0.7468
* **F1-score:** 0.8404

The lower threshold increases the model's sensitivity to the minority Attack class while maintaining high precision.

The selected threshold was saved as:

```text
models/final_random_forest_threshold.pkl
```

---

## 11. Final Model Performance

The final Random Forest was evaluated on the chronological final test set.

The final results were:

| Metric    |  Score |
| --------- | -----: |
| ROC-AUC   | 0.9001 |
| PR-AUC    | 0.8407 |
| Precision | 0.9609 |
| Recall    | 0.7468 |
| F1-score  | 0.8404 |
| Accuracy  | 0.9600 |

### Interpretation

The model achieved a **ROC-AUC of 0.9001**, indicating strong discrimination between Normal and Attack observations.

The **PR-AUC of 0.8407** is particularly important because the Attack class is highly imbalanced.

At the selected threshold of 0.10, the model achieved:

* approximately **96.09% precision**
* approximately **74.68% recall**
* approximately **84.04% F1-score**

This represents a substantially more useful attack-detection operating point than the default 0.50 threshold.

---

## 12. Final Confusion Matrix

The final confusion matrix using the selected threshold was:

```text
[[241726   1376]
 [ 11455  33787]]
```

The confusion matrix visualization was generated and saved as:

```text
images/final_random_forest_confusion_matrix.png
```

The confusion matrix provides a direct view of:

* True Negatives
* False Positives
* False Negatives
* True Positives

and is especially important for understanding how many attacks were missed versus how many normal observations were incorrectly classified as attacks.

---

## 13. Feature Importance

Feature importance was calculated using the final Random Forest model.

The top 20 features according to the Random Forest impurity-based feature importance were:

| Rank | Feature | Importance |
| ---: | ------- | ---------: |
|    1 | AIT402  |   0.155007 |
|    2 | AIT401  |   0.087832 |
|    3 | PIT502  |   0.084522 |
|    4 | PIT503  |   0.083351 |
|    5 | AIT502  |   0.067080 |
|    6 | PIT501  |   0.066322 |
|    7 | FIT503  |   0.056231 |
|    8 | FIT601  |   0.040485 |
|    9 | AIT504  |   0.034614 |
|   10 | AIT501  |   0.034023 |
|   11 | FIT501  |   0.033001 |
|   12 | AIT503  |   0.032550 |
|   13 | AIT202  |   0.031379 |
|   14 | FIT401  |   0.028210 |
|   15 | FIT504  |   0.023490 |
|   16 | AIT203  |   0.021411 |
|   17 | DPIT301 |   0.021308 |
|   18 | LIT301  |   0.015786 |
|   19 | LIT101  |   0.014233 |
|   20 | LIT401  |   0.012830 |

The visualization was saved as:

```text
images/random_forest_feature_importance.png
```

### Observation

`AIT402` was the most influential feature according to the model, with an importance of approximately **0.1550**.

The next most influential variables were:

* AIT401
* PIT502
* PIT503
* AIT502
* PIT501
* FIT503

The concentration of importance among several sensors from the later stages of the SWaT process suggests that measurements from these process areas provide substantial predictive information for distinguishing attack observations from normal operation.

Feature importance should not be interpreted as causal evidence; it indicates the contribution of features to the predictive behavior of the fitted Random Forest.

---

## 14. Permutation Importance

Permutation importance was also calculated using the final Random Forest and the `average_precision` scoring metric.

The top 20 features were:

| Rank | Feature | Permutation Importance |
| ---: | ------- | ---------------------: |
|    1 | AIT402  |               0.014602 |
|    2 | FIT401  |               0.014457 |
|    3 | FIT201  |               0.010816 |
|    4 | FIT301  |               0.008957 |
|    5 | PIT503  |               0.008774 |
|    6 | FIT504  |               0.006327 |
|    7 | FIT501  |               0.005834 |
|    8 | PIT502  |               0.005095 |
|    9 | UV401   |               0.003949 |
|   10 | LIT401  |               0.003893 |
|   11 | AIT203  |               0.003685 |
|   12 | P102    |               0.003370 |
|   13 | FIT101  |               0.003361 |
|   14 | AIT504  |               0.003079 |
|   15 | PIT501  |               0.002504 |
|   16 | LIT101  |               0.001734 |
|   17 | DPIT301 |               0.001716 |
|   18 | AIT502  |               0.001515 |
|   19 | MV302   |               0.001314 |
|   20 | FIT503  |               0.000597 |

### Interpretation

Permutation importance provides a complementary interpretation of the model.

Instead of measuring how often a feature is used in tree splits, permutation importance measures how model performance changes when the values of a feature are randomly shuffled.

Using `average_precision` as the scoring metric makes this analysis particularly relevant to the imbalanced Attack-detection task.

`AIT402` and `FIT401` were the two strongest features according to permutation importance.

The rankings are not identical to the built-in Random Forest feature importance rankings, which is expected because the two methods measure different aspects of model behavior.

The permutation importance results were retained as an additional model-interpretation analysis rather than treating either ranking as definitive causal evidence.

---

## 15. Threshold Analysis Visualization

Threshold analysis was performed to examine the trade-off between:

* Precision
* Recall
* F1-score

across different classification thresholds.

The best threshold was:

**0.10**

with:

* Precision = **0.960868**
* Recall = **0.746806**
* F1 = **0.840420**

The threshold analysis visualization was saved as:

```text
images/logistic_regression_threshold_analysis.png
```

> Note: the visualization filename contains `logistic_regression` from the earlier analysis workflow, but the final selected threshold of 0.10 was evaluated and applied to the **Final Random Forest** model.

---

## 16. API Deployment

The final Random Forest model was deployed through a REST API using FastAPI.

The API loads:

```text
models/final_random_forest.pkl
models/final_random_forest_scaler.pkl
models/final_random_forest_threshold.pkl
```

### Available Endpoints

#### Root

```text
GET /
```

Returns information confirming that the API is running and identifies the deployed model and threshold.

#### Health Check

```text
GET /health
```

Expected response:

```json
{
  "status": "healthy"
}
```

#### Prediction

```text
POST /predict
```

The endpoint accepts exactly **44 numerical features** in the same order used during model training.

The API:

1. Validates that exactly 44 features are supplied.
2. Converts the input into a numerical array.
3. Applies the saved StandardScaler.
4. Generates an attack probability using the final Random Forest.
5. Applies the saved classification threshold of 0.10.
6. Returns the prediction, label, probability, and threshold.

Example successful response from the final API test:

```json
{
  "prediction": 1,
  "label": "Attack",
  "attack_probability": 0.9958088640094522,
  "threshold": 0.1
}
```

This test produced an attack probability of approximately:

**99.58%**

and the API correctly returned:

**Attack**

with HTTP status:

**200**

---

## 17. API Input Validation

Input validation was implemented using Pydantic.

The API requires exactly 44 features.

A valid request containing 44 features successfully returned:

```text
HTTP 200
```

An invalid request containing only one feature returned:

```text
HTTP 422
```

with the validation message indicating that the list must contain at least 44 items.

This ensures that malformed requests are rejected before being passed to the machine learning model.

---

## 18. Reproducibility and Saved Artifacts

The final model pipeline artifacts were saved so that the same trained model can be reused without retraining.

Saved artifacts include:

```text
models/final_random_forest.pkl
models/final_random_forest_scaler.pkl
models/final_random_forest_threshold.pkl
```

These files allow the API to reproduce the preprocessing and prediction behavior of the final evaluated model.

The project also stores generated visualizations for model evaluation and interpretation.

---

## 19. Generated Visualizations

The following visualizations were generated during the modeling and interpretation stages:

### Random Forest Feature Importance

```text
images/random_forest_feature_importance.png
```

Shows the top 20 features according to the Random Forest's built-in feature importance.

### Final Random Forest Confusion Matrix

```text
images/final_random_forest_confusion_matrix.png
```

Shows the classification outcomes of the final model on the chronological test set.

### Threshold Analysis

```text
images/logistic_regression_threshold_analysis.png
```

Shows the Precision, Recall, and F1-score across different classification thresholds used during threshold analysis.

---

## 20. Overall Cleaning and Modeling Decisions

The main methodological decisions made during the project were:

1. Remove variables with approximately **68.79% missing values** rather than applying aggressive imputation.
2. Retain duplicate observations because repeated sensor states can be legitimate in industrial time-series data.
3. Keep `Timestamp` for chronological ordering but exclude it from the predictive feature matrix.
4. Use a chronological **80/20 train-test split** for final evaluation.
5. Use `TimeSeriesSplit` during Random Forest hyperparameter tuning.
6. Use class-aware evaluation because the Attack class represents only **3.79%** of the original observations.
7. Prioritize PR-AUC, Recall, Precision, and F1-score in addition to ROC-AUC.
8. Tune the classification threshold rather than relying on the default 0.50 threshold.
9. Select **0.10** as the final classification threshold because it maximized F1-score.
10. Use both built-in feature importance and permutation importance for model interpretation.
11. Save the final model, scaler, and threshold as reusable deployment artifacts.
12. Expose the final model through a FastAPI REST API with input validation.

These decisions were made to produce a model evaluation that is more representative of a real industrial attack-detection setting rather than relying solely on a random split and accuracy.
