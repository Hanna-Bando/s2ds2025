# Exercise 3. - GroupBy

### Introduction:

GroupBy can be summarized as Split-Apply-Combine.

Special thanks to: https://github.com/justmarkham for sharing the dataset and materials.

Check out this [Diagram](http://i.imgur.com/yjNkiwL.png)  

Check out [Alcohol Consumption Exercises Video Tutorial](https://youtu.be/az67CMdmS6s) to watch a data scientist go through the exercises


### Step 1. Import the necessary libraries


```python
import pandas as pd
```

### Step 2. Import the dataset from this [address](https://raw.githubusercontent.com/justmarkham/DAT8/master/data/drinks.csv). 


```python
url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/drinks.csv'
df = pd.read_csv(url, sep = ',')
```

### Step 3. Assign it to a variable called drinks.


```python
drinks = df
```

             country  beer_servings  spirit_servings  wine_servings  \
    0    Afghanistan              0                0              0   
    1        Albania             89              132             54   
    2        Algeria             25                0             14   
    3        Andorra            245              138            312   
    4         Angola            217               57             45   
    ..           ...            ...              ...            ...   
    188    Venezuela            333              100              3   
    189      Vietnam            111                2              1   
    190        Yemen              6                0              0   
    191       Zambia             32               19              4   
    192     Zimbabwe             64               18              4   
    
         total_litres_of_pure_alcohol continent  
    0                             0.0        AS  
    1                             4.9        EU  
    2                             0.7        AF  
    3                            12.4        EU  
    4                             5.9        AF  
    ..                            ...       ...  
    188                           7.7        SA  
    189                           2.0        AS  
    190                           0.1        AS  
    191                           2.5        AF  
    192                           4.7        AF  
    
    [193 rows x 6 columns]
    

### Step 4. Which continent drinks more beer on average?


```python
beer = drinks.groupby('continent').beer_servings.mean()
beer.sort_values(ascending=False)
```




    continent
    EU    193.777778
    SA    175.083333
    OC     89.687500
    AF     61.471698
    AS     37.045455
    Name: beer_servings, dtype: float64



### Step 5. For each continent print the statistics for wine consumption.


```python
print(drinks.groupby('continent').wine_servings.describe())
```

               count        mean        std  min   25%    50%     75%    max
    continent                                                               
    AF          53.0   16.264151  38.846419  0.0   1.0    2.0   13.00  233.0
    AS          44.0    9.068182  21.667034  0.0   0.0    1.0    8.00  123.0
    EU          45.0  142.222222  97.421738  0.0  59.0  128.0  195.00  370.0
    OC          16.0   35.625000  64.555790  0.0   1.0    8.5   23.25  212.0
    SA          12.0   62.416667  88.620189  1.0   3.0   12.0   98.50  221.0
    

### Step 6. Print the mean alcohol consumption per continent for every column


```python
drinks.groupby('continent').mean(numeric_only=True)
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
      <th>beer_servings</th>
      <th>spirit_servings</th>
      <th>wine_servings</th>
      <th>total_litres_of_pure_alcohol</th>
    </tr>
    <tr>
      <th>continent</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AF</th>
      <td>61.471698</td>
      <td>16.339623</td>
      <td>16.264151</td>
      <td>3.007547</td>
    </tr>
    <tr>
      <th>AS</th>
      <td>37.045455</td>
      <td>60.840909</td>
      <td>9.068182</td>
      <td>2.170455</td>
    </tr>
    <tr>
      <th>EU</th>
      <td>193.777778</td>
      <td>132.555556</td>
      <td>142.222222</td>
      <td>8.617778</td>
    </tr>
    <tr>
      <th>OC</th>
      <td>89.687500</td>
      <td>58.437500</td>
      <td>35.625000</td>
      <td>3.381250</td>
    </tr>
    <tr>
      <th>SA</th>
      <td>175.083333</td>
      <td>114.750000</td>
      <td>62.416667</td>
      <td>6.308333</td>
    </tr>
  </tbody>
</table>
</div>



### Step 7. Print the median alcohol consumption per continent for every column


```python
drinks.groupby('continent').median(numeric_only=True)
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
      <th>beer_servings</th>
      <th>spirit_servings</th>
      <th>wine_servings</th>
      <th>total_litres_of_pure_alcohol</th>
    </tr>
    <tr>
      <th>continent</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AF</th>
      <td>32.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>2.30</td>
    </tr>
    <tr>
      <th>AS</th>
      <td>17.5</td>
      <td>16.0</td>
      <td>1.0</td>
      <td>1.20</td>
    </tr>
    <tr>
      <th>EU</th>
      <td>219.0</td>
      <td>122.0</td>
      <td>128.0</td>
      <td>10.00</td>
    </tr>
    <tr>
      <th>OC</th>
      <td>52.5</td>
      <td>37.0</td>
      <td>8.5</td>
      <td>1.75</td>
    </tr>
    <tr>
      <th>SA</th>
      <td>162.5</td>
      <td>108.5</td>
      <td>12.0</td>
      <td>6.85</td>
    </tr>
  </tbody>
</table>
</div>



### Step 8. Print the mean, min and max values for spirit consumption.
#### This time output a DataFrame


```python
drinks.groupby('continent').spirit_servings.agg(['mean', 'min', 'max'])
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
      <th>mean</th>
      <th>min</th>
      <th>max</th>
    </tr>
    <tr>
      <th>continent</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AF</th>
      <td>16.339623</td>
      <td>0</td>
      <td>152</td>
    </tr>
    <tr>
      <th>AS</th>
      <td>60.840909</td>
      <td>0</td>
      <td>326</td>
    </tr>
    <tr>
      <th>EU</th>
      <td>132.555556</td>
      <td>0</td>
      <td>373</td>
    </tr>
    <tr>
      <th>OC</th>
      <td>58.437500</td>
      <td>0</td>
      <td>254</td>
    </tr>
    <tr>
      <th>SA</th>
      <td>114.750000</td>
      <td>25</td>
      <td>302</td>
    </tr>
  </tbody>
</table>
</div>


