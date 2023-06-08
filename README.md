# data_analysis_by_creating_statistical_tests
# Data Analysis By Creating Statistical Tests

This repository contains a Python class, `StatAnalyser`, that allows for data analysis by performing various statistical tests. The class is designed to work with datasets stored in CSV format and provides methods to analyze different scenarios involving independent and target variables. The `data` folder contains three CSV files that can be used for analysis.
 ### Description
<img src="https://github.com/Asmaa-Mohy/Data-Visualization-Dashboard/blob/main/data/Screenshot%202022-05-01%20173437.png?raw=true">
The `StatAnalyser` class enables data analysis by conducting statistical tests to determine independent variables in three different scenarios:

**Binary Target Variable**: In this scenario, categorical variables are assessed using the chi-squared test of independence. Numerical variables undergo either a t-test or a Mann-Whitney U test, depending on the data's normality.

**Multi Categorical Target Variable**: When dealing with a multi-categorical target variable, the appropriate statistical test for categorical variables is the chi-squared test of independence. Numerical variables are subjected to an analysis of variance (ANOVA) or a Kruskal-Wallis test, depending on the normality of the data.

**Continuous Target Variable**: If the target variable is continuous, the appropriate statistical tests for categorical variables include t-tests, Mann-Whitney U tests, ANOVA, or Kruskal-Wallis tests, depending on the number of categories. Numerical variables are analyzed using Pearson correlation or Spearman correlation, depending on the data's normality.

The `StatAnalyser` class also provides a method for performing a Shapiro-Wilk test to check the normality of the data.

To use the `StatAnalyser` class effectively, please refer to the `stat_tests.ipynb` notebook. It contains deployment examples and comprehensive explanations of the statistical analysis process.

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
