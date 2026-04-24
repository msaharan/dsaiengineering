# Understanding tabular foundation models: extracting embeddings to enhance existing ML models

I have been looking at the hands-on demo Jupyter notebook given on TabPFN's GitHub repository. Until now, I have been focusing on the following sections that compare the performance of TabPFN with XGBoost, Random Forest, and CatBoost and showcase TabPFN's interpretability.

![Screenshot 2026-04-24 at 17.43.21](./20260424-understanding-tabular-foundation-models-extracting-embeddings-to-enhance-existing-ml-models.assets/Screenshot%202026-04-24%20at%2017.43.21.png)

I was planning to dive deeper into these sections, but today I came across some resources that made me think that the performance comparison in general is a fairly trivial matter and the model interpretability discussed in this notebook is fairly basic. Therefore, I am moving on to other topics that are relevant to tabular foundation models instead of dwelling on these sections for too long. This notebook is still a good starting point for me, though, so I will use it again today to discuss the next topic that I want to cover.

## Embeddings

### What are embeddings? And why should we care?

Tree-based models such as XGBoost, Random Forest, etc., have been the industry standard for tabular data because they work well and are often easier to deploy under prediction-latency constraints. However, they require extensive work during development: task-specific labels, manual feature engineering, separate models for separate tasks, and periodic retraining when the data distribution changes.

To take an industry as an example, let's consider the financial industry, where tree-based models are used for fraud detection, transaction monitoring, personalization, marketing, risk modeling, anti-money laundering, and time-series-related workflows.

![Screenshot 2026-04-24 at 19.13.55](./20260424-understanding-tabular-foundation-models-extracting-embeddings-to-enhance-existing-ml-models.assets/Screenshot%202026-04-24%20at%2019.13.55.png)

While recent tabular foundation models can be competitive with strong tree-based baselines on many small-to-medium tabular tasks, they can be slower at inference time compared to tree-based models, especially when the prediction requires conditioning on a training context. In addition to slow inference speed, other factors specific to the workflow or budget of an organization might also make tree-based models hard to replace with new tabular foundation models in the near future. Therefore, if there is a way to enhance the existing models using tabular foundation models, that could produce immediate impact within an organization.

So far, I have explored using TabPFN as a predictive model: give it a train set and a test set, and ask it to produce predictions. But foundation models are useful not only because they predict. They are useful because they learn representations. This is where embeddings become important.

Embeddings are dense numerical vectors that represent the data after it has passed through a deep learning model. In the case of tabular data, we can start with numerical and categorical features, pass them through a deep learning model such as a transformer, and obtain embeddings that encode useful relationships in the data.

![Screenshot 2026-04-24 at 19.08.18](./20260424-understanding-tabular-foundation-models-extracting-embeddings-to-enhance-existing-ml-models.assets/Screenshot%202026-04-24%20at%2019.08.18.png)

### Hands-on demo

In this section, we use TabPFN to extract embeddings from tabular rows. These embeddings are dense vectors that encode how the model internally represents each row after seeing the training context. We can then train a simple downstream model, such as logistic regression, on these embeddings.

The following code snippet is taken from the [official TabPFN hands-on demo notebook](https://colab.research.google.com/github/PriorLabs/TabPFN/blob/main/examples/notebooks/TabPFN_Demo_Local.ipynb).

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from tabpfn_extensions import TabPFNClassifier, TabPFNRegressor
from tabpfn_extensions.embedding import TabPFNEmbedding

# Load and evaluate classification dataset
print("Loading classification dataset (breast cancer)...")
df = load_breast_cancer(return_X_y=False)
X, y = df["data"], df["target"]
attribute_names = df["feature_names"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.5,
    random_state=42,
)

# Train and evaluate vanilla logistic regression
model = LogisticRegression(
    max_iter=10000,
    random_state=42,
)
model.fit(X_train, y_train)
print(
    f"Baseline Logistic Regression Accuracy: {accuracy_score(y_test, model.predict(X_test)):.4f}",
)

# Train and evaluate TabPFN embeddings (vanilla)
clf = TabPFNClassifier(n_estimators=1, random_state=42)
embedding_extractor = TabPFNEmbedding(tabpfn_clf=clf, n_fold=0)
train_embeddings = embedding_extractor.get_embeddings(
    X_train,
    y_train,
    X_test,
    data_source="train",
)
test_embeddings = embedding_extractor.get_embeddings(
    X_train,
    y_train,
    X_test,
    data_source="test",
)

model = LogisticRegression(
    max_iter=10000,
    random_state=42,
)
model.fit(train_embeddings[0], y_train)
y_pred = model.predict(test_embeddings[0])
print(
    f"Logistic Regression with TabPFN (Vanilla) Accuracy: {accuracy_score(y_test, y_pred):.4f}",
)

# Note: Using test_embeddings and y_test from your original script.
# The embeddings have shape (n_splits, n_samples, n_features), so we use the first split [0].
test_only_embeddings = test_embeddings[0]

# Apply t-SNE to reduce the TEST embeddings to 2 dimensions.
# The number of samples in the test set is len(y_test). Perplexity must be less than that.
# We'll set perplexity to 10, which is suitable for this dataset size.
print("\nApplying t-SNE for visualization on the TEST SET only...")
tsne = TSNE(n_components=2, random_state=42, perplexity=10)
embeddings_2d = tsne.fit_transform(test_only_embeddings)

# Create a DataFrame for easy plotting
df_plot = pd.DataFrame(
    {
        "t-SNE-1": embeddings_2d[:, 0],
        "t-SNE-2": embeddings_2d[:, 1],
        "label": y_test,
    }
)

# Plot the 2D embeddings for the test set
plt.figure(figsize=(9, 7))
sns.scatterplot(
    data=df_plot,
    x="t-SNE-1",
    y="t-SNE-2",
    hue="label",  # Color points by their class label
    palette="viridis",
    alpha=0.9,
    s=80,
)

plt.title("t-SNE Visualization of TabPFN Embeddings (Test Set Only)")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.legend(title="Class")
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()
```

```
Output:
Baseline Logistic Regression Accuracy: 0.9614
Logistic Regression with TabPFN (Vanilla) Accuracy: 0.9509
```

In this example, we first train a baseline logistic regression model directly on the original breast cancer dataset. This gives us a baseline accuracy of `0.9614`. Then we use TabPFN to extract embeddings from the same data. These embeddings are then used as input features for another logistic regression model. This gives an accuracy of `0.9509`. In this particular case, the embedding-based logistic regression does not beat the logistic regression baseline. But that is not the main point of this section. The important point is to demonstrate that TabPFN can be used not only as a predictive model, but also as an embedding generator.

The embeddings are high-dimensional vectors. Humans cannot directly visualize 50-, 100-, or 500-dimensional vectors. So the notebook uses t-SNE to compress those embeddings into 2 dimensions. t-SNE stands for t-distributed Stochastic Neighbor Embedding. The main idea is that if two points are close together in the original high-dimensional embedding space, t-SNE tries to place them close together in the 2D plot.

![Screenshot 2026-04-24 at 18.52.40](./20260424-understanding-tabular-foundation-models-extracting-embeddings-to-enhance-existing-ml-models.assets/Screenshot%202026-04-24%20at%2018.52.40.png)

Here, the t-SNE plot shows two reasonably separate regions, which suggests that TabPFN has learned an embedding space where the two classes are represented differently. However, this visualization should not be over-interpreted. t-SNE is mainly useful for visual inspection. It does not prove that the model is better, and it is not part of the prediction pipeline. 

## Outro

With this progress, my version of the hands-on demo notebook currently has the following table of contents. 

 ![Screenshot 2026-04-24 at 19.58.24](./20260424-understanding-tabular-foundation-models-extracting-embeddings-to-enhance-existing-ml-models.assets/Screenshot%202026-04-24%20at%2019.58.24.png)

In future posts, I will continue to explore other aspects of tabular foundation models, especially those related to the finance industry.

