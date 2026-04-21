[Mohit Saharan](https://www.linkedin.com/in/msaharan), P7

___

# Understanding tabular foundation models: TabPFN repository

In the previous four blog posts ([P3](https://www.linkedin.com/posts/msaharan_20260415-tabular-foundation-models-1pdf-ugcPost-7450221502156791808-ZMP8?utm_source=share&utm_medium=member_desktop&rcm=ACoAAC8005UBr31urJ8gF7KXefP2-G8r_HNvI2g)-[P6](https://www.linkedin.com/posts/msaharan_20260420-understanding-tfms-pretraining-synthetic-datapdf-ugcPost-7452030754789699584-7coL?utm_source=share&utm_medium=member_desktop&rcm=ACoAAC8005UBr31urJ8gF7KXefP2-G8r_HNvI2g)), I read up on tabular foundation models to gain vocabulary on their theoretical aspects, such as their pre-training, their architectuere, and how they make predictions. By now, I think I know about them to a decent level to recognize the relevant parts in the code and gain the vocabulary of the engineering aspects. So, today, I am going to read through Prior Labs' TabPFN GitHub repository. 

## GitHub.com/PriorLabs/TabPFN

I am browsing the repository at this commit: https://github.com/PriorLabs/TabPFN/tree/eaefd29252a0897bd644c1840934b34ce08e194f.

 ![Screenshot 2026-04-21 at 17.46.05](./20260421-understanding-tabular-foundation-models-tabpfn-repo.assets/Screenshot%202026-04-21%20at%2017.46.05.png)

Let's start with the README file.

### README

#### Basic Usage

Currently, the default model is TabPFN-2.6, which is trained purely on synthetic data. We can use it as follows.

```python
from tabpfn import TabPFNClassifier

clf = TabPFNClassifier()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
```

Let's look at a complete example for binary classification.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from tabpfn import TabPFNClassifier

# Load data
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.33, random_state=42
)

# Initialize a classifier
clf = TabPFNClassifier()
clf.fit(X_train, y_train)

# Predict probabilities
prediction_probabilities = clf.predict_proba(X_test)
print("ROC AUC:", roc_auc_score(y_test, prediction_probabilities[:, 1]))

# Predict labels
predictions = clf.predict(X_test)
print("Accuracy", accuracy_score(y_test, predictions))
```

It seems quite straightforward to use. There's an example for multiclass classification, but let's leave it for later.

#### TabPFN Ecosystem

> Choose the right TabPFN implementation for our needs:
>
> - **[TabPFN Client](https://github.com/priorlabs/tabpfn-client)** Simple API client for using TabPFN via cloud-based inference.
>
> - **[TabPFN Extensions](https://github.com/priorlabs/tabpfn-extensions)** A powerful companion repository packed with advanced utilities, integrations, and features - great place to contribute:
>
>   - **`interpretability`**: Gain insights with SHAP-based explanations, feature importance, and selection tools.
>   - **`unsupervised`**: Tools for outlier detection and synthetic tabular data generation.
>   - **`embeddings`**: Extract and use TabPFN’s internal learned embeddings for downstream tasks or analysis.
>   - **`many_class`**: Handle multi-class classification problems that exceed TabPFN's built-in class limit.
>   - **`rf_pfn`**: Combine TabPFN with traditional models like Random Forests for hybrid approaches.
>   - **`hpo`**: Automated hyperparameter optimization tailored to TabPFN.
>   - **`post_hoc_ensembles`**: Boost performance by ensembling multiple TabPFN models post-training.
>
>   To install:
>
>   ```
>   git clone https://github.com/priorlabs/tabpfn-extensions.git
>   pip install -e tabpfn-extensions
>   ```
>
> - **[TabPFN (this repo)](https://github.com/priorlabs/tabpfn)** Core implementation for fast and local inference with PyTorch and CUDA support.
>
> - **[TabPFN UX](https://ux.priorlabs.ai/)** No-code graphical interface to explore TabPFN capabilities—ideal for business users and prototyping.

The TabPFN repo (this repo) and the TabPFN extensions repo are the most relevant for me right now. I didn't know about the extensions repo. Apparently, it's also a great place to contribute to the development. I quickly glanced at the [`priorlabs/tabpfn-extensions/src/tabpfn_extensions`](https://github.com/PriorLabs/tabpfn-extensions/tree/main/src/tabpfn_extensions) directory and want to spend some time there later to see how these utilities are implemented.

#### Workflow at a glance



```mermaid
---
config:
  theme: 'default'
  themeVariables:
    edgeLabelBackground: 'white'
---
graph TD
    %% 1. DEFINE COLOR SCHEME & STYLES
    classDef default fill:#fff,stroke:#333,stroke-width:2px,color:#333;
    classDef start_node fill:#e8f5e9,stroke:#43a047,stroke-width:2px,color:#333;
    classDef process_node fill:#e0f2f1,stroke:#00796b,stroke-width:2px,color:#333;
    classDef decision_node fill:#fff8e1,stroke:#ffa000,stroke-width:2px,color:#333;

    style Infrastructure fill:#fff,stroke:#ccc,stroke-width:5px;
    style Unsupervised fill:#fff,stroke:#ccc,stroke-width:5px;
    style Data fill:#fff,stroke:#ccc,stroke-width:5px;
    style Performance fill:#fff,stroke:#ccc,stroke-width:5px;
    style Interpretability fill:#fff,stroke:#ccc,stroke-width:5px;

    %% 2. DEFINE GRAPH STRUCTURE
    subgraph Infrastructure
        start((Start)) --> gpu_check["GPU available?"];
        gpu_check -- Yes --> local_version["Use TabPFN<br/>(local PyTorch)"];
        gpu_check -- No --> api_client["Use TabPFN-Client<br/>(cloud API)"];
        task_type["What is<br/>your task?"]
    end

    local_version --> task_type
    api_client --> task_type

    end_node((Workflow<br/>Complete));

    subgraph Unsupervised
        unsupervised_type["Select<br/>Unsupervised Task"];
        unsupervised_type --> imputation["Imputation"];
        unsupervised_type --> data_gen["Data<br/>Generation"];
        unsupervised_type --> tabebm["Data<br/>Augmentation"];
        unsupervised_type --> density["Outlier<br/>Detection"];
        unsupervised_type --> embedding["Get<br/>Embeddings"];
    end

    subgraph Data
        data_check["Data Checks"];
        model_choice["Samples > 50k or<br/>Classes > 10?"];
        data_check -- "Table Contains Text Data?" --> api_backend_note["Note: API client has<br/>native text support"];
        api_backend_note --> model_choice;
        data_check -- "Time-Series Data?" --> ts_features["Use Time-Series<br/>Features"];
        ts_features --> model_choice;
        data_check -- "Purely Tabular" --> model_choice;
        model_choice -- "No" --> finetune_check;
        model_choice -- "Yes, 50k-100k samples" --> ignore_limits["Set<br/>ignore_pretraining_limits=True"];
        model_choice -- "Yes, >100k samples" --> subsample["Large Datasets Guide<br/>"];
        model_choice -- "Yes, >10 classes" --> many_class["Many-Class<br/>Method"];
    end

    subgraph Performance
        finetune_check["Need<br/>Finetuning?"];
        performance_check["Need Even Better Performance?"];
        speed_check["Need faster inference<br/>at prediction time?"];
        kv_cache["Enable KV Cache<br/>(fit_mode='fit_with_cache')<br/><small>Faster predict; +Memory ~O(N×F)</small>"];
        tuning_complete["Tuning Complete"];

        finetune_check -- Yes --> finetuning["Finetuning"];
        finetune_check -- No --> performance_check;

        finetuning --> performance_check;

        performance_check -- No --> tuning_complete;
        performance_check -- Yes --> hpo["HPO"];
        performance_check -- Yes --> post_hoc["Post-Hoc<br/>Ensembling"];
        performance_check -- Yes --> more_estimators["More<br/>Estimators"];
        performance_check -- Yes --> speed_check;

        speed_check -- Yes --> kv_cache;
        speed_check -- No --> tuning_complete;

        hpo --> tuning_complete;
        post_hoc --> tuning_complete;
        more_estimators --> tuning_complete;
        kv_cache --> tuning_complete;
    end

    subgraph Interpretability
        tuning_complete --> interpretability_check;

        interpretability_check["Need<br/>Interpretability?"];

        interpretability_check --> feature_selection["Feature Selection"];
        interpretability_check --> partial_dependence["Partial Dependence Plots"];
        interpretability_check --> shapley["Explain with<br/>SHAP"];
        interpretability_check --> shap_iq["Explain with<br/>SHAP IQ"];
        interpretability_check -- No --> end_node;

        feature_selection --> end_node;
        partial_dependence --> end_node;
        shapley --> end_node;
        shap_iq --> end_node;
    end

    %% 3. LINK SUBGRAPHS AND PATHS
    task_type -- "Classification or Regression" --> data_check;
    task_type -- "Unsupervised" --> unsupervised_type;

    subsample --> finetune_check;
    ignore_limits --> finetune_check;
    many_class --> finetune_check;

    %% 4. APPLY STYLES
    class start,end_node start_node;
    class local_version,api_client,imputation,data_gen,tabebm,density,embedding,api_backend_note,ts_features,subsample,ignore_limits,many_class,finetuning,feature_selection,partial_dependence,shapley,shap_iq,hpo,post_hoc,more_estimators,kv_cache process_node;
    class gpu_check,task_type,unsupervised_type,data_check,model_choice,finetune_check,interpretability_check,performance_check,speed_check decision_node;
    class tuning_complete process_node;

    %% 5. ADD CLICKABLE LINKS
    click local_version "https://github.com/PriorLabs/TabPFN" "TabPFN Backend Options"
    click api_client "https://github.com/PriorLabs/tabpfn-client" "TabPFN API Client"
    click api_backend_note "https://github.com/PriorLabs/tabpfn-client" "TabPFN API Backend"
    click unsupervised_type "https://github.com/PriorLabs/tabpfn-extensions" "TabPFN Extensions"
    click imputation "https://github.com/PriorLabs/tabpfn-extensions/blob/main/examples/unsupervised/imputation.py" "TabPFN Imputation Example"
    click data_gen "https://github.com/PriorLabs/tabpfn-extensions/blob/main/examples/unsupervised/generate_data.py" "TabPFN Data Generation Example"
    click tabebm "https://github.com/PriorLabs/tabpfn-extensions/blob/main/examples/tabebm/tabebm_augment_real_world_data.ipynb" "TabEBM Data Augmentation Example"
    click density "https://github.com/PriorLabs/tabpfn-extensions/blob/main/examples/unsupervised/density_estimation_outlier_detection.py" "TabPFN Density Estimation/Outlier Detection Example"
    click embedding "https://github.com/PriorLabs/tabpfn-extensions/tree/main/examples/embedding" "TabPFN Embedding Example"
    click ts_features "https://github.com/PriorLabs/tabpfn-time-series" "TabPFN Time-Series Example"
    click many_class "https://github.com/PriorLabs/tabpfn-extensions/blob/main/examples/many_class/many_class_classifier_example.py" "Many Class Example"
    click finetuning "https://github.com/PriorLabs/TabPFN/blob/main/examples/finetune_classifier.py" "Finetuning Example"
    click feature_selection "https://github.com/PriorLabs/tabpfn-extensions/blob/main/examples/interpretability/feature_selection.py" "Feature Selection Example"
    click partial_dependence "https://github.com/PriorLabs/tabpfn-extensions/blob/main/examples/interpretability/pdp_example.py" "Partial Dependence Plots Example"
    click shapley "https://github.com/PriorLabs/tabpfn-extensions/blob/main/examples/interpretability/shap_example.py" "Shapley Values Example"
    click shap_iq "https://github.com/PriorLabs/tabpfn-extensions/blob/main/examples/interpretability/shapiq_example.py" "SHAP IQ Example"
    click post_hoc "https://github.com/PriorLabs/tabpfn-extensions/blob/main/examples/phe/phe_example.py" "Post-Hoc Ensemble Example"
    click hpo "https://github.com/PriorLabs/tabpfn-extensions/blob/main/examples/hpo/tuned_tabpfn.py" "HPO Example"
    click subsample "https://github.com/PriorLabs/tabpfn-extensions/blob/main/examples/large_datasets/large_datasets_example.py" "Large Datasets Example"
    click kv_cache "https://github.com/PriorLabs/TabPFN/blob/main/examples/kv_cache_fast_prediction.py" "KV Cache Fast Prediction Example"
```



### Relevant code

Let's use this diagram to figure out what I need to focus on now in the beginning. Assuming that I

- have local PyTorch,
- need classification/regression,
- have time-series data or purely tabular data,
- have less than 50k samples and less than 10 classes in the dataset,
- don't need finetuning,
- don't need better performance than default,
- need feature selection,
- need partial dependence plots,
- need explanation with SHAP,

I could read the following files and sections (line numbers mentioned after `:`) in this order. 

**Core Path (Local PyTorch, Classification/Regression, No Finetuning, Default Performance)**

1. `src/tabpfn/__init__.py:3` — entrypoints (`TabPFNClassifier`, `TabPFNRegressor`)
   - this is the public API surface that exposes TabPFNClassifier and TabPFNRegressor, so it’s our entry point from user code to the core implementation
2. `examples/tabpfn_for_binary_classification.py:21` — minimal classifier usage
   - shows the minimal binary-classification workflow (fit → predict_proba/predict) we’ll follow for default local PyTorch usage
3. `examples/tabpfn_for_multiclass_classification.py:24` — multiclass usage
   - shows the same workflow for multiclass tasks and how probabilities are evaluated in a multiclass setting
4. `examples/tabpfn_for_regression.py:24` — regressor usage
   - gives the baseline regression flow and highlights TabPFN’s distributional outputs (mean plus quantiles/mode), which ties directly to our regression understanding path

**Model Init + Default Execution Mode**

5. `src/tabpfn/base.py:93` — `initialize_tabpfn_model` (model/version/checkpoint resolution)
   - picks classifier vs regressor checkpoint(s), resolves auto model path, and prepares model objects for your local PyTorch workflow
6. `src/tabpfn/model_loading.py:567` — `load_model_criterion_config` (checkpoint loading contract)
   - actually loads checkpoint weights/config (and regression criterion), which is the foundation for all later fit/predict
7. `src/tabpfn/base.py:276` — `create_inference_engine` (we’ll use `fit_preprocessors` by default)
   - selects execution mode; with our settings this is typically fit_preprocessors (default, no special performance tuning)
8. `src/tabpfn/base.py:382` — `initialize_model_variables_helper` (device, precision, inference config resolution)
   - wires model + config + device + precision + inference config into estimator state before training context is built

**Dataset Constraints (<50k samples, <10 classes)**

9. `src/tabpfn/inference_config.py:181` — pretraining-limit defaults (`MAX_NUMBER_OF_CLASSES`, `MAX_NUMBER_OF_SAMPLES`, `MAX_NUMBER_OF_FEATURES`)
   - defines pretraining-relevant limits (`MAX_NUMBER_OF_CLASSES`, MAX_NUMBER_OF_SAMPLES, etc.) that our dataset assumptions rely on
10. `src/tabpfn/validation.py:32` — fit input validation path
    - canonical fit-time validation/shape checks for both classifier and regressor
11. `src/tabpfn/validation.py:204` — class-count check
    - enforces class-count compatibility (relevant to our “<10 classes” assumption)
12. `src/tabpfn/validation.py:220` — sample/feature limit checks
    - enforces sample/feature constraints (relevant to our “<50k samples” branch)

**Classification/Regression Internals** 

13. `src/tabpfn/classifier.py:635` — `_initialize_dataset_preprocessing`
    - handles modality detection, cleaning, label encoding, and estimator config generation for classification
14. `src/tabpfn/classifier.py:731` — `fit`
    - top-level classification flow for our default path (no finetuning, no HPO/post-hoc)
15. `src/tabpfn/classifier.py:1059` — `_raw_predict`
    - predict-time input normalization + safe forwarding into inference engine
16. `src/tabpfn/classifier.py:1283` — `logits_to_probabilities`
    - defines how raw model logits become probabilities (temperature/averaging/balancing policy)
17. `src/tabpfn/classifier.py:1357` — `forward` aggregation over estimators
    - aggregates estimator outputs and class permutations into final classification tensor outputs
18. `src/tabpfn/regressor.py:604` — `_initialize_dataset_preprocessing`
    - regression-side preprocessing plus target transform setup and ensemble config creation
19. `src/tabpfn/regressor.py:750` — `fit`
    -  top-level regression flow including target normalization and inference engine setup
20. `src/tabpfn/regressor.py:889` — `predict`
    - produces mean/median/mode/quantiles from ensemble outputs for our regression path
21. `src/tabpfn/regressor.py:1031` — `_iter_forward_executor`
    - aligns per-estimator outputs with border transforms before aggregation
22. `src/tabpfn/regressor.py:1258` — `_logits_to_output`
    - final decoding from distribution logits to user-facing regression outputs

**Tabular + “Time-Series as Features” Data Handling**

23. `src/tabpfn/preprocessing/modality_detection.py:17` — modality inference (numerical/categorical/text heuristics)
    - decides numeric/categorical/text treatment, central for both purely tabular and tabularized time-series features
24. `src/tabpfn/preprocessing/clean.py:35` — `clean_data`, `fix_dtypes`, `process_text_na_dataframe`
    - dtype fixing + categorical/text encoding + NA handling before model preprocessing
25. `src/tabpfn/preprocessing/ensemble.py:146` — ensemble member preprocessing
    - builds per-estimator transformed training contexts; this is core to TabPFN’s ensemble prompting behavior

**Inference Runtime (Default Performance Path)**

26. `src/tabpfn/inference.py:602` — `InferenceEngineCachePreprocessing` (default path with `fit_preprocessors`)
    - default runtime engine in our path; caches train preprocessing and reuses it across predictions
27. `src/tabpfn/inference.py:1301` — `_prepare_model_inputs` (train+test context construction)
    - concatenates `X_train` + `X_test` context for in-context prediction

**Model Mechanics (Understanding Why Predictions Work)**

28. `src/tabpfn/architectures/tabpfn_v2_6.py:611` — main model `forward`
    - the main architecture forward pass that turns context rows into test predictions
29. `src/tabpfn/architectures/tabpfn_v2_6.py:205` — train/test attention behavior
    - key train/test information-flow rule (prevents leakage from test labels)
30. `src/tabpfn/architectures/base/bar_distribution.py:184` — regression distribution decoding
    - defines regression distribution math over bins, which underlies uncertainty-aware outputs

**Feature Selection, PDP, SHAP**

31. Implemented in `tabpfn-extensions`, not in this repo
32. examples/notebooks/TabPFN_Demo_Local.ipynb — this repo’s integration example calling `tabpfn_extensions` interpretability APIs
    - demonstrates practical integration with `tabpfn_extensions` for SHAP/feature-selection-style interpretability on top of fitted TabPFN models

## Outro

Alright, now I have a good sense of the repository -- how things are done and what's where. I think now it's time to look into the hands-on examples and run them. Later.