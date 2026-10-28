# Annual Income Prediction

My solution to [Annual Income Prediction (#186)](https://platform.olimpiada-ai.ro/en/problems/186) on the Romanian AI Olympiad platform.

The task: predict whether a person earns **more than $50,000 a year** from census data, answer two data analysis questions, and group people by a short text describing their professional profile.

## The data

About 39,000 people, one row each:

| Type | Columns |
| --- | --- |
| Numeric | age, fnlwgt, educational_num, capital_gain, capital_loss, hours_per_week |
| Categorical | workclass, marital_status, occupation, relationship, race, gender, native_country |
| Text | profile_description |
| Target | income (`<=50K` / `>50K`) |

- The classes are imbalanced: roughly **1 in 4** people earn more than $50,000.
- `workclass`, `occupation` and `native_country` have missing values.
- `education` and `educational_num` hold the same information (a name and its rank), so only the numeric one is kept.

## Subtasks

### 1. Country with the most high earners

Find the country, other than the United States, with the most people earning >50K. I filtered out the US and unknown countries, then grouped by country and counted.

### 2. Occupation with the highest rate of high earners

Unlike subtask 1, this asks for a **rate**, not a count. With the target encoded as 0/1, the mean of each group is exactly the share of people earning >50K, so a `groupby(...).mean()` gives the answer directly.

### 3. Income classification (scored by F1 macro)

I compared two models with the same evaluation setup.

**Logistic regression (baseline)**, built as one scikit-learn `Pipeline`:

- numeric columns: median imputation, then standard scaling
- categorical columns: missing values become their own `"Unknown"` category, then one-hot encoding
- `ColumnTransformer` applies each branch to its columns, so the preprocessing is refit inside every CV fold and no information leaks from validation data

**LightGBM**, using its native categorical support:

- categorical columns are passed as pandas `category` dtype, with no one-hot encoding or scaling
- the test set reuses the training categories, so the codes mean the same thing in both

**Evaluation:**

- stratified 5-fold cross-validation, which keeps the 1-in-4 ratio in every fold
- out-of-fold probabilities from `cross_val_predict`, so every prediction comes from a model that never saw that row
- the decision threshold is tuned on those probabilities to maximize F1 macro, instead of using the default 0.5

| Model | CV F1 macro |
| --- | --- |
| Logistic regression | 0.792 |
| **LightGBM** | **0.820** |

LightGBM wins by about 0.03. Trees capture interactions that a linear model can't, for example how age, marital status and hours per week combine. The final model is LightGBM retrained on all of the training data, with the tuned threshold applied to the test probabilities.

### 4. Profile clustering (scored by Adjusted Rand Index) *(in progress)*

Group people based on their `profile_description` text. The plan:

- turn the text into numeric vectors with **TF-IDF**, which weights words that are frequent in one description but rare overall
- cluster the vectors (k-means as a first approach)
- the score is the **Adjusted Rand Index**, which compares my grouping with the true one regardless of how the clusters are numbered

## What I learned

- **Count vs. rate:** the country with the most high earners and the occupation with the highest share of them are different questions, answered with `sum` and `mean`.
- **Missing values can be information:** people with an unknown occupation earn >50K much less often, so a separate `"Unknown"` category works better than filling in the most common value.
- **Pipelines prevent leakage:** when imputation, scaling and encoding live inside the pipeline, cross-validation refits them on each training fold automatically.
- **Linear vs. tree models need different preprocessing:** logistic regression needs scaling and one-hot encoding; LightGBM needs neither.
- **With imbalanced classes, the threshold matters:** tuning it on out-of-fold predictions is a fair way to choose it without touching the test set.
- **Sensitive attributes:** race and gender are part of the dataset. A model using them can reproduce existing inequalities, which matters as soon as a model like this is used to make decisions about people.

## Stack

Python · pandas · NumPy · scikit-learn · LightGBM
