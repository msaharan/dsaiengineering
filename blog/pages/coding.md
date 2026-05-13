# Coding Fundamentals

This page is about software engineering fundamentals demonstrated in the posts
and notebooks. It is not the place for target construction, feature
engineering, model theory, or deployment. Those belong to the Data Science, ML
Theory, and ML Engineering pages.

The coding standard behind the work is:

> Code should make an analysis repeatable, inspectable, and easier to extend.

The current work is still notebook-first, so the software engineering depth is
not the same as a production codebase. Still, the notebooks exercise important
engineering habits: clear execution order, reusable helper functions, explicit
inputs and outputs, artifact paths, error handling, runtime awareness, and
dependency control.

## Program State

Every notebook or script has state. State includes imported libraries, random
seeds, loaded data, fitted objects, intermediate DataFrames, saved paths, and
environment variables.

Good coding practice makes state visible:

```python
RANDOM_STATE = 42
ARTIFACT_DIR = Path("artifacts")
DATA_DIR = Path("data")
```

Bad practice hides state in scattered cells, implicit global variables, or
manual steps that are not recorded.

The engineering lesson is that the computer only knows the execution state, not
the story in the author's head. A serious notebook must therefore make the
state legible.

## Variables and Naming

Names are part of the code contract. A name should communicate what a value is,
not merely its type.

Examples of useful naming:

```python
train_df
calibration_df
holdout_df
feature_cols
target_col
artifact_path
```

Examples of weak naming:

```python
df2
final
new_data
result1
```

In ML notebooks, unclear names create real risk. If `valid_df` sometimes means
model-selection data and sometimes means calibration data, the code becomes
easy to misread and easy to misuse.

## Data Structures

The posts mostly use Python data structures and pandas objects.

Common structures include:

- lists for ordered collections such as feature column names;
- dictionaries for named configuration values and metric outputs;
- tuples for fixed records such as `(start_date, end_date)`;
- DataFrames for rectangular data;
- Series for single indexed columns;
- NumPy arrays for model inputs and numerical outputs;
- pathlib `Path` objects for filesystem locations.

The software engineering question is:

> What structure makes the allowed operations clear?

For example, a list of feature names preserves order:

```python
feature_cols = ["vix_lag_1", "term_spread", "momentum_12m"]
```

A dictionary records named metric values:

```python
metrics = {
    "average_precision": ap,
    "roc_auc": auc,
    "brier": brier,
}
```

Both structures are simple, but they encode different expectations.

## Functions

A function should turn a repeated operation into a named unit with explicit
inputs and outputs.

The basic contract is:

\[
\text{output} = f(\text{input})
\]

In code:

```python
def ensure_artifact_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
```

For analytical notebooks, functions are useful when the same operation appears
more than once:

- creating output directories;
- computing metric dictionaries;
- saving predictions;
- summarizing splits;
- plotting repeated diagnostic figures;
- validating required columns.

A function should not depend on hidden notebook state unless that is deliberate
and documented. Prefer this:

```python
def select_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    missing = sorted(set(cols) - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df.loc[:, cols]
```

over this:

```python
def select_columns():
    return df[feature_cols]
```

The second version depends on global variables. That makes it harder to test,
reuse, or move into a script.

## Control Flow

Control flow decides which code runs and how often it runs.

The basic forms are:

- `if` statements for branches;
- `for` loops for repeated work;
- `try` and `except` for handling expected failure modes;
- early returns for rejecting invalid inputs.

In the posts, control flow appears when:

- a library or API path may fail;
- a model family is optional;
- multiple assets or folds must be processed;
- artifacts should be saved only after they are created;
- missing files or columns should stop execution.

The engineering goal is not to make control flow clever. The goal is to make
the workflow deterministic and readable.

## Input and Output

Programs interact with the outside world through inputs and outputs.

Inputs include:

- local files;
- public URLs;
- package APIs;
- model APIs;
- user-defined configuration values.

Outputs include:

- CSV files;
- parquet files;
- figures;
- notebooks;
- logs;
- saved model objects;
- prediction tables;
- metric summaries.

The code should make these boundaries explicit. A good output path is stable,
named, and reproducible:

```python
pred_path = ARTIFACT_DIR / "holdout_predictions.csv"
metrics_path = ARTIFACT_DIR / "metrics.json"
```

Saving artifacts is a software engineering habit, not only an ML habit. It
separates computation from later inspection.

## File Paths

The posts use many local and generated files. Path handling should be explicit
and portable within the repository.

Prefer `pathlib`:

```python
from pathlib import Path

root = Path.cwd()
artifact_dir = root / "artifacts"
artifact_dir.mkdir(parents=True, exist_ok=True)
```

Avoid hard-coded absolute paths inside reusable code unless the path is truly a
local-only note. Absolute paths make reuse and review harder.

## Dependencies

The notebooks rely on Python packages such as pandas, NumPy, scikit-learn,
XGBoost, TabPFN, TabICL-related packages, plotting libraries, and data access
libraries.

Dependency engineering asks:

- Which packages are required?
- Which versions were used?
- Which packages need GPU support?
- Which APIs are unstable?
- Which calls require network or credentials?

This matters because model results are not only a function of code. They also
depend on the package environment.

Mathematically, the computed result is closer to:

\[
\text{result} = f(\text{code}, \text{data}, \text{dependencies}, \text{hardware})
\]

not merely:

\[
\text{result} = f(\text{code}, \text{data})
\]

## Reproducibility

Reproducible code produces the same result when the same inputs and environment
are used.

In practice, reproducibility requires:

- fixed random seeds where supported;
- recorded package versions;
- deterministic split rules;
- saved intermediate artifacts;
- explicit data ranges;
- clear execution order.

Randomness is not wrong, but uncontrolled randomness makes comparison weaker.
If a result depends on a random seed, the seed is part of the experiment.

## Testing

The current work is mostly notebook-first, so formal testing is still a gap.
However, the posts already use test-like checks:

- required columns exist;
- row counts after filtering are plausible;
- train and holdout periods do not overlap;
- saved artifacts exist;
- metrics are finite;
- prediction vectors align with target rows.

These checks can later become unit tests.

A unit test checks a small behavior:

```python
def test_select_columns_rejects_missing_column():
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError):
        select_columns(df, ["a", "b"])
```

The broader engineering direction is to move repeated notebook logic into
modules and test the parts that should never silently change.

## Error Handling

Failure is part of real code.

Common failure modes in the posts include:

- missing packages;
- unstable APIs;
- data download failures;
- GPU or memory constraints;
- columns missing after a data-source change;
- model calls that fail because input shape or type is invalid.

Good error handling does not hide failure. It makes the failure understandable:

```python
if not required_cols.issubset(df.columns):
    missing = sorted(required_cols - set(df.columns))
    raise ValueError(f"Input data is missing required columns: {missing}")
```

The standard is to fail early when continuing would produce misleading results.

## Runtime and Resource Awareness

Runtime is a software property. A workflow that is theoretically correct but
too slow to rerun is hard to maintain.

The posts repeatedly surface runtime issues:

- free Colab limits;
- Kaggle GPU selection;
- TabPFN Client latency;
- TabICL API and memory constraints;
- embedding feature-width expansion;
- larger asset universes increasing computation.

The computational cost of a workflow can be thought of as:

\[
\text{cost} =
\text{setup time}
+ \text{data time}
+ \text{model time}
+ \text{artifact time}
+ \text{debug time}
\]

Good engineering tries to make each term visible.

## API Use

An API is a contract between code and another system.

The posts use APIs in several senses:

- package APIs such as scikit-learn estimators;
- model APIs such as TabPFN classifiers and regressors;
- client APIs such as TabPFN Client;
- data APIs such as public finance download tools.

An API call should be wrapped by checks when the failure mode is predictable.
For example, if a model expects numeric arrays, the code should ensure that
categorical or missing values have been handled before the call.

## Notebook to Module Boundary

Notebooks are useful for exploration and explanation. Modules are better for
repeated logic.

A notebook is suitable for:

- narrative;
- inspection;
- one-off plots;
- experiment ordering;
- result interpretation.

A module is suitable for:

- reusable loaders;
- validation functions;
- metric computation;
- artifact utilities;
- configuration parsing.

The current work has not yet fully crossed from notebooks into reusable
packages. That is a known software engineering gap and a natural next step.

## Version Control

Version control records how code and text change over time.

For this work, version control matters because:

- posts and notebooks evolve together;
- artifacts and generated files should be separated from source text;
- claims should match the code version that produced them;
- future recap pages should be able to identify what changed.

A useful commit should have a clear scope. Mixing unrelated changes makes later
review harder.

## Current Depth

The coding fundamentals demonstrated so far are strongest in:

- notebook organization;
- Python data structures;
- pandas-oriented workflows;
- file and artifact handling;
- model API usage;
- runtime awareness;
- failure-mode documentation.

The software engineering depth that remains to be developed is:

- reusable package structure;
- automated unit and integration tests;
- typed interfaces;
- command-line entry points;
- configuration files;
- continuous integration;
- deployment-ready services.

That distinction is important. The current work shows strong analytical coding
discipline. It is not yet a full software product.
