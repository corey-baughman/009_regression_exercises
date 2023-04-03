import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


def plot_residuals(df, y, yhat):
    '''
    function takes in a DataFrame, y(target actuals), and yhat(target predictions)
    and outputs a seaborn lmplot with the regression line for predicted values and a
    scatterplot of the actuals
    '''
    if len(df) > 1_000:
        df = df.sample(1_000)
        sns.lmplot(x=y, y=yhat,
            data = df, line_kws={'color': 'black'}, scatter_kws={'alpha': 0.05})
        plt.show()
    else:
        sns.lmplot(x=y, y=yhat, 
                   data = df, line_kws={'color': 'black'}, scatter_kws={'alpha': 0.05})
        plt.show()
    
    
def regression_errors_mean(df, y, yhat):
    '''
    function takes in a DataFrame, y(target actuals), and yhat(target predictions) and returns 
    SSE(Sum of Squared Errors), ESS(Explained Sum of Squares),
    TSS(Total Sum of Squares), MSE(Mean Squared Error), RMSE(Root Means Squared Error), and 
    R-squared(Proportion of variance in y explained by x).
    Those values are returned as variables and also in a dictionary called 'regression_errors'.
    The ESS and TSS calculate the baseline based on the mean value of y.
    '''
    SSE = ((df[yhat] - df[y]) ** 2).sum()
    ESS = ((df[yhat] - df[y].mean())**2).sum()
    TSS = ((df[y] - df[y].mean())**2).sum()
    MSE = SSE / len(df)
    RMSE = MSE ** (1/2)
    R2 = ESS / TSS 
    regression_errors = {'SSE':SSE, 'ESS':ESS, 'TSS':TSS, 'MSE':MSE, 'RMSE':RMSE, 'R2':R2}
    return SSE, ESS, TSS, MSE, RMSE, R2, regression_errors


def regression_errors_median(df, y, yhat):
    '''
    function takes in a DataFrame, y(target actuals), and yhat(target predictions) and returns 
    SSE(Sum of Squared Errors), ESS(Explained Sum of Squares),
    TSS(Total Sum of Squares), MSE(Mean Squared Error), RMSE(Root Means Squared Error), and 
    R-squared(Proportion of variance in y explained by x).
    Those values are returned as variables and also in a dictionary called 'regression_errors'.
    The ESS and TSS calculate the baseline based on the median value of y.
    '''
    SSE = ((df[yhat] - df[y]) ** 2).sum()
    ESS = ((df[yhat] - df[y].median())**2).sum()
    TSS = ((df[y] - df[y].median())**2).sum()
    MSE = SSE / len(df)
    RMSE = MSE ** (1/2)
    R2 = ESS / TSS 
    regression_errors = {'SSE':SSE, 'ESS':ESS, 'TSS':TSS, 'MSE':MSE, 'RMSE':RMSE, 'R2':R2}
    return SSE, ESS, TSS, MSE, RMSE, R2, regression_errors


def baseline_errors_mean(df, y):
    '''
    function takes in a DataFrame, y(target actuals) and returns SSE_b(Sum of Squared Errors),
    MSE_b(Mean Squared Error), and RMSE_b(Root Means Squared Error). Those values are returned 
    as variables and also in a dictionary called 'regression_errors_baseline'. The baseline is
    calculated based on the mean value of y.
    '''
    SSE_b = ((df.y.mean() - df.y) ** 2).sum()
    MSE_b = SSE_b / len(df)
    RMSE_b = MSE_b ** (1/2)
    regression_errors_baseline = {'SSE_b':SSE_b, 'MSE_b':MSE_b, 'RMSE_b':RMSE_b}
    return SSE_b, MSE_b, RMSE_b, regression_errors_baseline

def baseline_errors_median(df, y):
    '''
    function takes in a DataFrame, y(target actuals) and returns SSE_b(Sum of Squared Errors),
    MSE_b(Mean Squared Error), and RMSE_b(Root Means Squared Error). Those values are returned 
    as variables and also in a dictionary called 'regression_errors_baseline'. The baseline is
    calculated based on the median value of y.
    '''
    SSE_b = ((df.y.mean() - df.y) ** 2).sum()
    MSE_b = SSE_b / len(df)
    RMSE_b = MSE_b ** (1/2)
    regression_errors_baseline = {'SSE_b':SSE_b, 'MSE_b':MSE_b, 'RMSE_b':RMSE_b}
    return SSE_b, MSE_b, RMSE_b, regression_errors_baseline


def better_than_baseline_mean(df, y, yhat):
    '''
    function takes in a DataFrame, y(target actuals), and yhat(target predictions)
    and returns True if the model performed better than the baseline by the RMSE
    metric and False if it was equivalent or worse than baseline. Baseline is based
    on the mean of y.
    '''
    if (((df.yhat - df.y) ** 2).sum() / len(df) ** (1/2)) - (
        ((df.y.mean() - df.y) ** 2).sum() / len(df) ** (1/2)) > 0:
        print('The model performed better than baseline')
        return True
    else:
        print('The model performed equivalent or worse than baseline = ')
        return False
    
    
def better_than_baseline_median(df, y, yhat):
    '''
    function takes in a DataFrame, y(target actuals), and yhat(target predictions)
    and returns True if the model performed better than the baseline by the RMSE
    metric and False if it was equivalent or worse than baseline. Baseline is based
    on the median of y.
    '''
    if (((df.yhat - df.y) ** 2).sum() / len(df) ** (1/2)) - (
        ((df.y.median() - df.y) ** 2).sum() / len(df) ** (1/2)) > 0:
        print('The model performed better than baseline')
        return True
    else:
        print('The model performed equivalent or worse than baseline = ')
        return False
    

    
    