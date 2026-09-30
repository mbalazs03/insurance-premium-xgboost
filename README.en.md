# Insurance Premium Prediction
Kaggle Playground Series - Season 4, Episode 12

[Magyar](README.md) | **English**

Solution for the Kaggle [Playground Series S4E12](https://www.kaggle.com/competitions/playground-series-s4e12)
competition. The goal is to predict each customer's insurance premium (`Premium Amount`) from
features such as age, income, health score, previous claims and policy type.

## Notebooks

1. **EDA** ([`01_eda.ipynb`](notebooks/01_eda.ipynb)):
   distributions, missing values, and how the features relate to the premium.
2. **Baseline model** ([`02_baseline.ipynb`](notebooks/02_baseline.ipynb)):
   XGBoost on the log of the premium, with a train/validation split and early stopping.

## Findings

- The premium is right-skewed. The competition metric is RMSLE, so the model is trained on
  `log1p(premium)` and optimises the metric directly.
- The data is synthetic and very noisy: most categories are split almost evenly
  (e.g. 33/33/33%), and most features barely affect the premium on their own.
- Missing values are informative: customers with missing `Customer Feedback` pay about 13%
  more on average. So there is no imputation, XGBoost handles missing values natively.
- `Annual Income` has a correlation of -0.09 with the premium, but by income decile the top
  two groups pay about 28% less, so the relationship is non-linear.
- The premium rises with the number of previous claims. Missing claims behave much like zero claims.

## Results

| # | Model | Validation | Public LB | Private LB |
|---|---|---|---|---|
| 0 | Mean as prediction | 1.0964 |  |  |
| 1 | XGBoost, `log1p` target | 1.0469 | 1.04547 | 1.04747 |

Metric: RMSLE, lower is better.

XGBoost is about 4.5% better than the mean baseline. Because the data is so noisy, the top
solutions in the competition are not much further ahead. The validation and leaderboard scores
are close, so local validation is reliable for further experiments.

## License

The code is licensed under [Apache 2.0](LICENSE). The competition data is also licensed under Apache 2.0.
