# data_analysis_by_creating_statistical_tests
# Data Analysis By Creating Statistical Tests

This repository contains a Python class, `StatAnalyser`, that allows for data analysis by performing various statistical tests. The class is designed to work with datasets stored in CSV format and provides methods to analyze different scenarios involving independent and target variables. The `data` folder contains three CSV files that can be used for analysis.

## Class: StatAnalyser

### Attributes

1. `result_dict`: Stores the results of the statistical tests.
2. `explain_series`: A series of explanations of the test results.
3. `p_series`: A series of p-values associated with the test results.
4. `independent_columns`: A list of independent columns in the dataset.

### Methods

```python
class StatAnalyser:
    def __init__(self):
        """
        Initializes the StatAnalyser class.
        """

    def cat_binORcat_bin(self, data, col, target):
        """
        Performs a chi-square test of independence for categorical and binary variables.
        """

    def normality_check(self, data, var, target):
        """
        Performs a normality check on a variable with respect to the target variable.
        """

    def num_binTest(self, data, col, target):
        """
        Performs a statistical test for a numerical predictor and a binary target variable.
        """

    def num_catTest(self, data, col, target):
        """
        Performs a statistical test for a numerical predictor and a categorical target variable.
        """

    def num_numTest(self, data, col, target):
        """
        Performs a statistical test for two numerical variables.
        """

    def fit(self, data, target):
        """
        Performs statistical tests for all predictors in the dataset.
        """

    def plot_pvalues(self):
        """
        Plots the results of the statistical tests.
        """
