# Data-Visualization-with-Python
Part of Data Science Path

```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
```


```python
# libs
import numpy as np # for numerical computaion
import matplotlib.pyplot as plt # for Visualization
import seaborn as sns # visualization of statistical view
import pandas as pd # for reading files
```

# Dataset 


```python
df = pd.read_csv('Datasets/haberman.csv')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>30</th>
      <th>64</th>
      <th>1</th>
      <th>1.1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>30</td>
      <td>62</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>30</td>
      <td>65</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>31</td>
      <td>59</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>31</td>
      <td>65</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>33</td>
      <td>58</td>
      <td>10</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
features = df.columns

# columns names
features
```




    Index(['30', '64', '1', '1.1'], dtype='object')




```python
# target variable 
df['1.1'].value_counts()

```




    1    224
    2     81
    Name: 1.1, dtype: int64



**observation**  
Two type of element 1 and 2

# Visualization in 2D  
Will check different combination to understand the separation of the datasets.  
We have 3 combination (30,64) , (30,1) (64,1)


```python
plt.scatter(df['30'],df['64'],marker='.')
plt.xlabel("30")
plt.ylabel('64')
plt.show()
```


    
![png](output_8_0.png)
    



```python
plt.scatter(df['30'],df['1'],marker='.')
plt.xlabel("30")
plt.ylabel('1')
plt.show()
```


    
![png](output_9_0.png)
    



```python
plt.scatter(df['64'],df['1'],marker='.')
plt.xlabel("64")
plt.ylabel('1')
plt.show()
```


    
![png](output_10_0.png)
    


# Using Seaborn


```python
g = sns.FacetGrid(df  ,hue='1.1' ,height=4)
g.map(plt.scatter ,'30' ,'1').add_legend()
plt.show()

g = sns.FacetGrid(df  ,hue='1.1' ,height=4)
g.map(plt.scatter ,'64' ,'1').add_legend()
plt.show()

g = sns.FacetGrid(df  ,hue='1.1' ,height=4)
g.map(plt.scatter ,'64' ,'30').add_legend()
plt.show()
```


    
![png](output_12_0.png)
    



    
![png](output_12_1.png)
    



    
![png](output_12_2.png)
    


**observation**  
* feature 64 and 1 to mixed with each other . it's not linear separable.

# PairPlots



```python
sns.pairplot(df , hue='1.1' ,height=5)
```




    <seaborn.axisgrid.PairGrid at 0x7f7ba9304b50>




    
![png](output_15_1.png)
    


**observation**
* pair (1,30) try to separate using polynomial features.

# Univeriate Analysis


```python
g =sns.FacetGrid(df,hue='1.1',height=5)
g.map(sns.distplot,'64' ).add_legend()
plt.show()

# feature 30
g =sns.FacetGrid(df,hue='1.1',height=5)
g.map(sns.histplot ,'30' ).add_legend()
plt.show()

# features 1
g =sns.FacetGrid(df,hue='1.1',height=5)
g.map(sns.histplot ,'1' ).add_legend()
plt.show()
```

    /home/iffishells/anaconda3/lib/python3.8/site-packages/seaborn/distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    /home/iffishells/anaconda3/lib/python3.8/site-packages/seaborn/distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)



    
![png](output_18_1.png)
    



    
![png](output_18_2.png)
    



    
![png](output_18_3.png)
    


**Obervation**
* most of the data is stucked into each other

# Boxplot


```python
#boxplot give us more idea to understan the datasets
sns.boxplot(x = '1.1' ,y='64' ,data=df)
```




    <AxesSubplot:xlabel='1.1', ylabel='64'>




    
![png](output_21_1.png)
    



```python
# features 30
sns.boxplot(x = '1.1' ,y='30' ,data=df)
```




    <AxesSubplot:xlabel='1.1', ylabel='30'>




    
![png](output_22_1.png)
    



```python
sns.boxplot(x = '1.1' ,y='1' ,data=df)
```




    <AxesSubplot:xlabel='1.1', ylabel='1'>




    
![png](output_23_1.png)
    


# violinplot


```python
sns.violinplot(data=df,
              x = '1.1',
              y = '30')
```




    <AxesSubplot:xlabel='1.1', ylabel='30'>




    
![png](output_25_1.png)
    



```python
sns.violinplot(data=df,
              x = '1.1',
              y = '64')
```




    <AxesSubplot:xlabel='1.1', ylabel='64'>




    
![png](output_26_1.png)
    



```python
sns.violinplot(data=df,
              x = '1.1',
              y = '1')
```




    <AxesSubplot:xlabel='1.1', ylabel='1'>




    
![png](output_27_1.png)
    



```python

```


```python

```
