##### imports #####
import numpy as np
import pandas as pd
from scipy.stats import shapiro, ttest_ind, pearsonr, spearmanr, f_oneway, kruskal,chi2_contingency,mannwhitneyu
import matplotlib.pyplot as plt

class StatAnalyser:
  """Methods for building Analysis."""

  def __init__(self):
    pass
  
  def cat_binORcat_bin(self,data,col,target):
    
    """Perform chi-square test of independence for categorical and binary variables.

      Args:
          data (pandas.DataFrame): The input dataframe.
          col (str): The name of the predictor column (categorical and binary).
          target (str): The name of the target column (categorical and binary).

      Returns:
          float: The p-value from the chi-square test.

      """ 
    # predictor : categorical and binary
    # target : categorical and binary

    # create cross table 
    contingency_table = pd.crosstab(data[col], data[target])  
    
    # check dependency of column 
    res = chi2_contingency(contingency_table)
    pvalue = res.pvalue

    # explain results
    explain = f""" 
          Chi-square test of independence of variables : predctor {col} and target variable {target}
          Null hypothesis : independence is true (no relationship) 
          Alternative hypothesis: independence is false (relationship)
          the probability of observing a chi-squared test statistic as extreme as the one computed from the contingency table, assuming the null hypothesis of independence is true= {pvalue}
    """
    return explain, pvalue

  
  def normality_check(self,data,var,target):
    
    """Perform normality check on a variable with respect to the target variable.
  
      Args:
          data (pandas.DataFrame): The input dataframe.
          var (str): The name of the variable to check for normality.
          target (str): The name of the target variable.
  
      Returns:
          bool: True if the variable is approximately normally distributed with respect to the target variable, False otherwise.
  
      """
    # check datatype of target
    isNum = data[target].dtype != 'O'
    
  
    # target varible is categorical
    if not isNum:
      
      # create list of categorical samples 
      samples =[data[data[target] == category][var] for category in data[target].unique()]
      # assert statement samples not empty
      pvalues=[]
      # iterate on each sample and check normality
      for sample in samples:
        pvalues.append(shapiro(sample).pvalue)
  
      # if False , is not normal
      return all(p > 0.05 for p in pvalues)
    
    else:
      
      # apply Shapiro-Wilk test on target 
      p= shapiro(data[target]).pvalue
      p2 = shapiro(data [var]).pvalue
      return p > 0.05 or p2 > 0.05
  

  def num_binTest(self,data,col,target):
    
    """Perform statistical test for a numerical predictor and binary target variable.
    
      Args:
          data (pandas.DataFrame): The input dataframe.
          col (str): The name of the numerical predictor column.
          target (str): The name of the binary target column.
    
      Returns:
          float: The p-value from the statistical test.
    
      Raises:
          AssertionError: If the target column does not have two unique values.
      """
    # predictor : numerical 
    # target : binary
    # Make sure that target is binary
    assert len( data[target].unique()) == 2 , f"Column {target} must be a binary variable with two unique values."
    # assert target must be binary
    
    # check normality of column 
    isNorm = self.normality_check(data,col,target)
    
    if isNorm:
      # perform the unpaired t-test
      res = ttest_ind(*[data[data[target] == category][col] for category in data[target].unique()], equal_var=False)
      pvalue = res.pvalue
      # explain results
      explain = f""" 
            T-test of independence of variables : predctor {col} and target variable {target}
            Null hypothesis : independence is true (no relationship) 
            Alternative hypothesis: independence is false (relationship)
            the probability of observing a T-test statistic as extreme as the one observed in the sample, assuming the null hypothesis of independence is true= {pvalue}
      """
    else:
      # perform Mann-Whitney U test
      res = mannwhitneyu(*[data[data[target] == category][col] for category in data[target].unique()])
      pvalue = res.pvalue
      # explain results
      explain = f""" 
            Mann-Whitney of independence of variables : predctor {col} and target variable {target}
            Null hypothesis : independence is true (no relationship) 
            Alternative hypothesis: independence is false (relationship)
            the probability of observing a Mann-Whitney statistic as extreme as the one observed in the sample, assuming the null hypothesis of independence is true= {pvalue}
      """
    return explain, pvalue


  def num_catTest(self,data,col,target):
    
    """Perform statistical test for a numerical predictor and categorical target variable.

      Args:
          data (pandas.DataFrame): The input dataframe.
          col (str): The name of the numerical predictor column.
          target (str): The name of the categorical target column.

      Returns:
          float: The p-value from the statistical test.

      Raises:
          AssertionError: If the target column is not categorical.
      """
    # predictor : numerical 
    # target : categorical
    # Make sure that target is categorical
    assert data[target].dtype == "object" or data[target].dtype.name == "category", f"Column {target} must be a categorical variable (string or category)."

    # check normality of column 
    isNorm = self.normality_check(data,col,target)

    if isNorm:
      # perform the ANOVA test
      res = f_oneway(*[data[data[target] == category][col] for category in data[target].unique()], equal_var=False)
      pvalue = res.pvalue
      # explain results
      explain = f""" 
            ANOVA test of independence of variables : predctor {col} and target variable {target}
            Null hypothesis : independence is true (no relationship) 
            Alternative hypothesis: independence is false (relationship)
            the probability of observing a ANOVA statistic as extreme as the one observed in the sample, assuming the null hypothesis of independence is true= {pvalue}
      """
    else:
      # perform Kruskal-Wallis H test
      res = kruskal(*[data[data[target] == category][col] for category in data[target].unique()])
      pvalue = res.pvalue
      # explain results
      explain = f""" 
            Kruskal-Wallis H test of independence of variables : predctor {col} and target variable {target}
            Null hypothesis : independence is true (no relationship) 
            Alternative hypothesis: independence is false (relationship)
            the probability of observing a Kruskal-Wallis statistic as extreme as the one observed in the sample, assuming the null hypothesis of independence is true= {pvalue}
      """
    return explain,pvalue

  def num_numTest(self,data,col,target):
    
    """Perform statistical test for two numerical variables.
  
      Args:
          data (pandas.DataFrame): The input dataframe.
          col (str): The name of the predictor column.
          target (str): The name of the target column.
  
      Returns:
          float: The p-value from the statistical test.
      """
    # predictor : numerical 
    # target : numerical
  
    # check normality of column 
    isNorm = self.normality_check(data,col,target)
  
    if isNorm:
      # perform the Pearson correlation
      res = pearsonr(data[col], data[target])
      pvalue = res.pvalue
      # explain results
      explain = f""" 
            Pearson correlation of variables : predctor {col} and target variable {target}
            Null hypothesis : independence is true (no correlation between the two variables) 
            Alternative hypothesis: independence is false (correlation between the two variables)
            probability of observing a correlation as strong as the one computed, assuming that there is actually no correlation between the two variables, assuming the null hypothesis of independence is true= {pvalue}
      """
    else:
      # perform Spearman correlation
      res = spearmanr(data[col], data[target])
      pvalue = res.pvalue
      # explain results
      explain = f""" 
            Spearman correlation of variables : predctor {col} and target variable {target}
            Null hypothesis : independence is true (no correlation between the two variables) 
            Alternative hypothesis: independence is false (correlation between the two variables)
            probability of observing a correlation as strong as the one computed, assuming that there is actually no correlation between the two variables, assuming the null hypothesis of independence is true= {pvalue}
      """
    return explain, pvalue

  def fit(self,data, target):

    """Perform statistical test for all predictors in data.
    
      Args:
          data (pandas.DataFrame): The input dataframe.
          target (str): The name of the binary target column.
    
      Returns:
          None
      """

    # check datatype of target
    isNum = data[target].dtype != 'O'

    # select cat_vars / num_vars
    if isNum:
      categorical_vars = data.select_dtypes(include=['object','category']).columns.tolist()
      numerical_vars = data.drop(columns=target).select_dtypes(include='number').columns.tolist()
    else:
      categorical_vars = data.drop(columns=target).select_dtypes(include='object').columns.tolist()
      numerical_vars = data.select_dtypes(include='number').columns.tolist()

    #intialize resluts
    self.result_dict ={}
    # check three cases of target 

    # First case : When target is binary
    if len( data[target].unique()) == 2 :
      # iterate on categorical_vars
      for cat in categorical_vars:
        e,p = self.cat_binORcat_bin(data,cat,target)
        #store resluts
        self.result_dict[cat]={}
        self.result_dict[cat]['explain'] = e
        self.result_dict[cat]['p-value'] = p
      
      # iterate on numerical_vars
      for num in numerical_vars:
        e,p = self.num_binTest(data,num,target)
        #store resluts
        self.result_dict[num]={}
        self.result_dict[num]['explain'] = e
        self.result_dict[num]['p-value'] = p

    # Secand case:  When target is category
    elif (data[target].dtype == "object" or data[target].dtype.name == "category") and len( data[target].unique()) > 2:

      # iterate on categorical_vars
      for cat in categorical_vars:
        e,p = self.cat_binORcat_bin(data,cat,target)
        #store resluts
        self.result_dict[cat]={}
        self.result_dict[cat]['explain'] = e
        self.result_dict[cat]['p-value'] = p
      
      # iterate on numerical_vars
      for num in numerical_vars:
        e,p = self.num_catTest(data,num,target)
        #store resluts
        self.result_dict[num]={}
        self.result_dict[num]['explain'] = e
        self.result_dict[num]['p-value'] = p
    # Last case : when target is numerical
    else:

      # iterate on categorical_vars
      for cat in categorical_vars:

        # check binary 
        if len( data[cat].unique()) == 2:
          e,p = self.num_binTest(data,target,cat)
        else:
          e,p = self.num_catTest(data,target,cat)
        #store resluts
        self.result_dict[cat]={}
        self.result_dict[cat]['explain'] = e
        self.result_dict[cat]['p-value'] = p

      # iterate on numerical_vars
      for num in numerical_vars:
        e,p = self.num_numTest(data,num,target)
        #store resluts
        self.result_dict[num]={}
        self.result_dict[num]['explain'] = e
        self.result_dict[num]['p-value'] = p

    # Create series for 'explain' values
    self.explain_series = pd.Series({col: self.result_dict[col]['explain'] for col in self.result_dict})

    # Create series for 'p-value' values
    self.p_series = pd.Series({col: self.result_dict[col]['p-value'] for col in self.result_dict})

    self.independent_columns = [col for col in self.result_dict if self.result_dict[col]['p-value']> 0.05]

  def plot_pvalues(self):
    """ Plot Results """
    self.p_series.plot(kind='barh')
    # Label axes
    plt.xlabel("P-values")
    plt.ylabel("Columns ")
    # Add title
    plt.title(f"P-values for test dependency on target")
  

  

  
      





