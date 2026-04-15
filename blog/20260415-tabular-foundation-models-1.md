[Mohit Saharan](https://linkedin.com/in/msaharan), P#3

____

# Tabular Foundation Models - 1

In this post, I want to start covering and learning the developments within tabular foundation models. In my physics research work, I have worked extensively with tabular data and also developed a binary classifier ML model using XGBoost. I want to extend my expertise in ML, and taking a step forward towards understanding the tabular foundation models feels to me a better way to go than learning natural language processing right now. This way, I will be building on my expertise vertically rather than extending it horizontally. 

I became aware of the tabular foundation models a few months ago through social media. [Cristoph Molnar] (https://christophmolnar.com/) is my go-to source for intrepretable ML. I came across his work in 2024 while working on my XGBoost binary classifier model when I needed to understand SHAP. So, in this post, I used his posts on tabular foundation models from Substack as a reference. The aim of this post is to make myself familiar with the concepts, so I simply typed here what I saw in his posts while selecting the parts I found relevant for now. I did the same for the code presented in this post to familiarize myself with the syntax. 

## What are tabular foundation models?

Tabular foundation models make predictions on new datasets without a classic training step: no hyperparameter tuning, no gradient descent at inference time. They are pre-trained on large collections of tabular data (often synthetic) and generalize to new tables in a single forward pass. This is a yound but fast-developing field. As of now, there are [17 labs](https://tabular-foundation.christophmolnar.com/labs/) and [21 libraries](https://tabular-foundation.christophmolnar.com/libraries/) available. Open-source models like TabPFN and TabICL match tuned gradient-boosted trees on standard benchmarks.

Reference: [Tabular Foundation Models Overview, Christoph Molnar](https://tabular-foundation.christophmolnar.com/).

## TabPFN and TabICL

Both TabPFN and TabICL are so-called Prior-Data Fitted Networks (PFNs), meaning that we pre-train them on lots of tabular data, and during the prediction step, we provide the entire "training data" plus labels and the test data for which we want the predictions. To call it training data at this point is misleading; it's more useful to think of this step as In-Context Learning, where the specific "training data" becomes context data. This is also a bit of an issue, since the context window has limitations. 

PFNs can be thought of as framework to train neural networks to perform <u>Bayesian inference</u>. The <u>prior distribution</u> is represented by the data used for pre-training, and the <u>prediction</u> is the posterior distribution. PFNs could also be initiated with other data: for example, if you train a PFN with sine waves, it will learn to predict sine waves. TabICL and TabPFN are specific PFNs trained with tabular data.

Reference: [The rise of tabular foundation models, Christoph Molnar](https://mindfulmodeler.substack.com/p/tabular-ml-is-about-to-get-weird).

## PFNs change how we model 

PFN-style modeling replaces the classic training + prediction with pre-training + in-context learning. In a way, PFNs shift the process by one abstraction level: The PFN pre-training is about "learning to learn", and the prediction phase is about ingesting labelled and unlabelled data at once and doing in-context learning.

Breaking the classic train + predict has a lot of implications:

- With tabular PFNs, you start with pre-trained model and download the weights, just like with open-source LLMs.
- You don't need to train anything. You just throw your labelled and unlabelled data towards the PFN and get the predictions out. This also means you no longer need to tune hyperparameters (except for setting a few inference-time parameters).
- Since PFNs are based on neural networks, we can also fine-tune them on further datasets. For example, if you have mostly time-series data, you can fine-tune.
- Each time we want to make prediction, we have toprovide the entire "training data" (context data). This makes it expensive if you can't batch yout predictions but only want one prediction at a time.
- As a result of requiring pre-training, we might see more closed models or licenses on the weights.

Reference: [The rise of tabular foundation models, Christoph Molnar](https://mindfulmodeler.substack.com/p/tabular-ml-is-about-to-get-weird).

## Pre-training tabular models isn't obvious

Traditionally, we train tabular models from scratch. A model trained to forecast water supply won’t really help us classify medical records. For “unstructured” data like images, it’s natural and easy to pretrain on one task and apply it to a completely new one. It has to do with the structure of the data: both the images of the Golden Retriever and the corn field in Iowa are made out of pixels. 

Tabular "structured" data is surprisingly messy:

- One tabular dataset may have 3 columns, while another may have 30,000, without any meaningful way of projecting one dataset to another.
- Scales and meanings of two different columns are often incomparable.
- The order of columns doesn’t matter: Shuffle them around, and it’s still the same dataset from a predictive perspective.
- Even if you have two columns that are seemingly the same between two tasks, they likely differ: The “age” column in one dataset may be age in years, while in another task it may be called “AGE” and it would be the age in milliseconds.

So, while you could try to pre-train based on the meaning of columns (i.e., using the column names), it seems like a lost cause. Modern tabular foundation models are pre-trained in such a way that they’re transferable between tasks.

Reference: [How PFNs make tabular foundation models work, Christoph Molnar](https://mindfulmodeler.substack.com/p/how-pfns-make-tabular-foundation).

## How TabPFN and other TFMs do it

See [How PFNs make tabular foundation models work, Christoph Molnar](https://mindfulmodeler.substack.com/p/how-pfns-make-tabular-foundation) and later posts in this publication.

## TabPFN Hands-on Demo

Reference: [PriorLabs GitHub](https://github.com/PriorLabs/TabPFN), [Google Colab Interactive Notebook Tutorial by PriorLabs](https://colab.research.google.com/github/PriorLabs/TabPFN/blob/main/examples/notebooks/TabPFN_Demo_Local.ipynb#scrollTo=0bBYgouXsh2A)

### Classification with TabPFN

A practical example of using TabPFN for a classificaiton task using the Parkinson's Disease dataset with the aim of predicting the presence of Parkinson's disease based on various voice measurements. A comparison of TabPFN's performance against other popular ML models (RandomForest, XGBoost, and CatBoost) using the ROC-AUC score is also shown.

#### Syntax for select steps (TabPFN and XGBoost)

```python
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# Train and evaluate the TabPFN classifier
tabpfn_classifier = TabPFNClassifier(random_state=42)
tabpfn_classifier.fit(X_train, y_train)
y_pred_proba_tabpfn = tabpfn_classifier.predict_proba(X_test)

roc_auc_tabpfn = roc_auc_score(y_test, y_pred_proba_tabpfn[:, 1])
print(f"TabPFN ROC AUC Score: {roc_auc_tabpfn:.4f}")

# Train and evaluate XGBoost
xgb_classifier = XGBClassifier(random_state=42)
xgb_classifier.fit(X_train, y_train)
y_pred_proba_xgb = xgb_classifier.predict_proba(X_test)

roc_auc_xgb = roc_auc_score(y_test, y_pred_proba_xgb[:, 1])
print(f"XGBoost ROC AUC Score: {roc_auc_xgb:.4f}")
```

```python
# Comparison of TabPFN with other classifiers using cross-validation

# Compare different ML models by training each one multiple times on different parts of the data and averaging their performance scores for a more reliable performance estimate

# Encode target labels to classes for baselines
le = LabelEncoder()
y = le.fit_tranform(y)

# Define models
models = [
  ("TabPFN", TabPFNClassifier(random_state=42)),
  (
    "RandomForest",
    make_pipeline(
      column_transformer, # string data needs to be encoded for model
      RandomForestClassifier(random_state=42),      
    ),
  ),
  (
    "XGBoost",
    make_pipeline(
    column_transformer, 
      XGBClassifier(random_state=42),
    ),
  ),
  (
    "CatBoost",
    make_pipeline(
      colun_transformer,
      CatBoostClassifier(random_state=42, verbose=0),
    ),
  ),
]

# Calculate scores
n_splits = 3
cv=StratifiedKFold(n_splits=n_splits, random_state=42, shuffle=True)
scoring = "roc_auc_ovr" if len(np.unique(y)) > 2 else "roc_auc"
scores = {
  name: cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=1, verbose=1)
  for name, model in models
}

# Plot results
df = pd.DataFrame(
  [(k, v.mean()) for (k,v) in scores.items()], columns=["Model", "ROC AUC"]
)
ax = df.plot(x="Model", y="ROC AUC", kind="bar", figsize=(10,6))
```



### Regression with TabPFN

A practical examle of using TabPFN for regression tasks using the Boston Housing dataset with the goal of predicting the median value of owner-occupied homes on the basis of the RMS error metric.

#### Syntax for select steps

```
# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

# Train and evaluate the TabPfn regressor
tabpfn_regressor = TabPFNRegressor(random_state=42)
tabpfn_regressor.fit(X_train, y_train)
y_pred = tabpfn_regressor.predict(X_test)

# Calculate the Root Mean Square Error
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"TabPFN RMSE: {rmse:.4f}")
```

```
# Compare different machine learning models by training each one multiple times
# on different parts of the data and averaging their performance scores for a
# more reliable performance estimate

# Define models
models = [
    ("TabPFN", TabPFNRegressor(random_state=42)),
    (
        "RandomForest",
        make_pipeline(
            column_transformer,  # string data needs to be encoded for model
            RandomForestRegressor(random_state=42),
        ),
    ),
    (
        "XGBoost",
        make_pipeline(
            column_transformer,  # string data needs to be encoded for model
            XGBRegressor(random_state=42),
        ),
    ),
    (
        "CatBoost",
        make_pipeline(
            column_transformer,  # string data needs to be encoded for model
            CatBoostRegressor(random_state=42, verbose=0),
        ),
    ),
]

# Calculate scores
scoring = "r2"
n_splits = 3
cv = KFold(n_splits=n_splits, random_state=42, shuffle=True)
scores = {
	name: cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=1, verbose=1)
	for name, model in models
}

# Plot results
df = pd.DataFrame([(k, v.mean()) for (k, v) in scores.items()], columns=["Model", "R2"])
ax = df.plot(x="Model", y="R2", kind="bar", figsize=(10, 6))
```

### Handling text data

A powerful feature of the TabPFN server is its ability to handle text data directly, without the need for manual feature engineering. This simplifies the process of working with datasets that contain a mix of numerical and textual features.

```python
# Load the clothing review dataset
# We restrict to 500 rows to make the example faster
df_text = pd.read_csv("cloth.csv", index_col=0).dropna()[:500]

# Define features and target
y_text = df_text["Rating"]
X_text = df_text.drop(columns=["Rating"])
```

```python
# Now, let's compare how TabPFN handles text natively versus how a baseline model like RandomForest needs a specific text processing pipeline. For baselines, we will create a pipeline that converts strings to ordinal features. For TabPFN, we simply pass the raw data with the text column directly to the classifier.

# Encode target labels to classes for baselines
le = LabelEncoder()
y_text = le.fit_transform(y_text)

# Define models
models = [
  ("TabPFN=Text", TabPFNClassifier(random_state=42)),
  (
    "TabPFN",
    make_pipeline(
      column_transformer, # string data needs to be encoded for model
      TabPFNClassifier(random_state=42),      
    ),
  ),
  (
    "RandomForest",
    make_pipeline(
      column_transformer, 
      RandomForestClassifier(random_state=42),
    ),
  ),
  (
    "XGBoost",
    make_pipeline(
    column_transformer,
    XGBClassifier(random_state=42),  
    ),
  ),
  (
    "CatBoost",
    make_pipeline(
      column_transformer,
      CatBoostClassifier(random_state=42, verbose=0),
    )
  )
  
  # Calculate scores
  cv = StratifierKFold(n_splits=3, random_state=42, shuffle=True)
  scoring = "roc_auc_ovr" if len(np.unique(y_text)) > 2 else "roc_auc"
  scores = {
    name: cross_val_score(
      model, X_text, y_text, cv=cv, scoring=scoring, n_jobs=1, verbose=1
    ).mean()
    for name, model in models
  }  
]
# Plot results
df = pd.DataFrame(list(scores.items()), columns=["Model", "ROC AUC"])
ax = df.plot(x="Model", y="ROC AUC", kind="bar", figsize=(10, 6))
```

### Unsupervised Learning with TabPFN

TabPFN can also be used for unsupervised learning tasks like outlier detection and synthetic data generation. These features are available through the `tabpfn-extensions` library.

```python
import torch
from sklearn.datasets import load_breast_cancer
from tabpfn_extensions import unsupervised
from tabpfn_extensions.unsupervised import experiments

# Load data

# Initialize models
clf = TabPFNClassifier(n_estimators=4)
reg = TabPFNRegressor(n_estimators=4)
model_unsupervised = unsupervised.TabPFNUnsupervisedModel(
	tabpfn_clf = clf, tabpfn_reg=reg
)

# Run outlier detection
exp_outlier = unsupervised.experiments.OutlierDetectionUnsupervisedExperiment(
	task_type = "unsupervised"
)
results = exp_outlier.run(
	tabpfn=model_unsupervised,
  X=torch.tensor(X, dtype=torch.float32),
  y=torch.tensor(y),
  attribute_names=attribute_names,
  indices=[4, 12], # Analyze features 4 and 12
)
```

### Model interpretability

Understanding *why* a model makes certain predictions is crucial for building trust and for debugging. The `tabpfn-extensions` library provides tools for model interpretability. We'll look at SHAP (SHapley Additive exPlanations) values, which show the impact of each feature on a specific prediction.

#### Shapley Values

Next, we'll use SHAP to understand our model's predictions. SHAP values break down a prediction to show the contribution of each feature, helping us see which factors are most influential for a given data point.

```python
from tabpfn_extensions import interpretability

# Load example dataset
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names
n_samples_test, n_samples_train = 25, 50

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

# Initialize and train model
clf = TabPFNClassifier()
clf.fit(X_train[:n_samples_train], y_train[:n_samples_train])

# Calculate SHAP values
shap_values = interpretability.shap.get_shap_values(
  estimator = clf,
  test_x=X_test[:n_samples_test],
  attribute_names = feature_names,
  algorithm="permutation",
)

# Create visualization
fig = interpretability.shap.plot_shap(shap_values)
```

#### Embeddings

Later

#### Feature selection

Feature selection is the process of selecting a subset of relevant features for use in model construction. It's useful for reducing model complexity, improving performance by removing noise, and decreasing training time. Here, we'll use Sequential Forward Selection (SFS), which starts with no features and iteratively adds the feature that most improves the model's performance.

The goal is to see if we can create a simpler, faster model with fewer features without a significant drop in accuracy.

```python
from tabpfn_extensions import interpretability

# Load data
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names

# Initialize model
clf = TabPFNClassifier(n_estimators=1)

# Feature selection
sfs = interretability.feature_selection.feature_selection(
	estimator = clf, X=X, y=y, n_features_to_select=4, feature_names=feature_names
)

# Print selected features
selected_features = [
  feature_names[i] for i in range(len(feature_names)) if sfs.get_support()[i]
]
```

### Predictive Behavior of TabFN

Later

### Time Series Forecasting

Later

### Using TabPFN as a base model for causal inference

Later

## Outlook

This post was just to get familiar with the tabular foundation models. In future posts, I will go deeper into their theoretical aspects and also into the Prior Labs' TABPFM hands-on demo.

