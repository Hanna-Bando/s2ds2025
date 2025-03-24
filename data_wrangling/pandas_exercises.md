# Exercise 1. - Getting and Knowing your Data

This time we are going to pull data directly from the internet.
Special thanks to: https://github.com/justmarkham for sharing the dataset and materials.

Check out [Occupation Exercises Video Tutorial](https://www.youtube.com/watch?v=W8AB5s-L3Rw&list=PLgJhDSE2ZLxaY_DigHeiIDC1cD09rXgJv&index=4) to watch a data scientist go through the exercises

### Step 1. Import the necessary libraries


```python
import pandas as pd
```

### Step 2. Import the dataset from this [address](https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.user). 


```python
url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.user'
df = pd.read_csv(url, sep = '|')
```

### Step 3. Assign it to a variable called users and use the 'user_id' as index


```python
users = df
users.set_index('user_id')
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
      <th>age</th>
      <th>gender</th>
      <th>occupation</th>
      <th>zip_code</th>
    </tr>
    <tr>
      <th>user_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>24</td>
      <td>M</td>
      <td>technician</td>
      <td>85711</td>
    </tr>
    <tr>
      <th>2</th>
      <td>53</td>
      <td>F</td>
      <td>other</td>
      <td>94043</td>
    </tr>
    <tr>
      <th>3</th>
      <td>23</td>
      <td>M</td>
      <td>writer</td>
      <td>32067</td>
    </tr>
    <tr>
      <th>4</th>
      <td>24</td>
      <td>M</td>
      <td>technician</td>
      <td>43537</td>
    </tr>
    <tr>
      <th>5</th>
      <td>33</td>
      <td>F</td>
      <td>other</td>
      <td>15213</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>939</th>
      <td>26</td>
      <td>F</td>
      <td>student</td>
      <td>33319</td>
    </tr>
    <tr>
      <th>940</th>
      <td>32</td>
      <td>M</td>
      <td>administrator</td>
      <td>02215</td>
    </tr>
    <tr>
      <th>941</th>
      <td>20</td>
      <td>M</td>
      <td>student</td>
      <td>97229</td>
    </tr>
    <tr>
      <th>942</th>
      <td>48</td>
      <td>F</td>
      <td>librarian</td>
      <td>78209</td>
    </tr>
    <tr>
      <th>943</th>
      <td>22</td>
      <td>M</td>
      <td>student</td>
      <td>77841</td>
    </tr>
  </tbody>
</table>
<p>943 rows × 4 columns</p>
</div>



### Step 4. See the first 25 entries


```python
#users[0:25]
users.head(25)
#I believe both are correct
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
      <th>user_id</th>
      <th>age</th>
      <th>gender</th>
      <th>occupation</th>
      <th>zip_code</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>24</td>
      <td>M</td>
      <td>technician</td>
      <td>85711</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>53</td>
      <td>F</td>
      <td>other</td>
      <td>94043</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>23</td>
      <td>M</td>
      <td>writer</td>
      <td>32067</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>24</td>
      <td>M</td>
      <td>technician</td>
      <td>43537</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>33</td>
      <td>F</td>
      <td>other</td>
      <td>15213</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>42</td>
      <td>M</td>
      <td>executive</td>
      <td>98101</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>57</td>
      <td>M</td>
      <td>administrator</td>
      <td>91344</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>36</td>
      <td>M</td>
      <td>administrator</td>
      <td>05201</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>29</td>
      <td>M</td>
      <td>student</td>
      <td>01002</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>53</td>
      <td>M</td>
      <td>lawyer</td>
      <td>90703</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>39</td>
      <td>F</td>
      <td>other</td>
      <td>30329</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>28</td>
      <td>F</td>
      <td>other</td>
      <td>06405</td>
    </tr>
    <tr>
      <th>12</th>
      <td>13</td>
      <td>47</td>
      <td>M</td>
      <td>educator</td>
      <td>29206</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14</td>
      <td>45</td>
      <td>M</td>
      <td>scientist</td>
      <td>55106</td>
    </tr>
    <tr>
      <th>14</th>
      <td>15</td>
      <td>49</td>
      <td>F</td>
      <td>educator</td>
      <td>97301</td>
    </tr>
    <tr>
      <th>15</th>
      <td>16</td>
      <td>21</td>
      <td>M</td>
      <td>entertainment</td>
      <td>10309</td>
    </tr>
    <tr>
      <th>16</th>
      <td>17</td>
      <td>30</td>
      <td>M</td>
      <td>programmer</td>
      <td>06355</td>
    </tr>
    <tr>
      <th>17</th>
      <td>18</td>
      <td>35</td>
      <td>F</td>
      <td>other</td>
      <td>37212</td>
    </tr>
    <tr>
      <th>18</th>
      <td>19</td>
      <td>40</td>
      <td>M</td>
      <td>librarian</td>
      <td>02138</td>
    </tr>
    <tr>
      <th>19</th>
      <td>20</td>
      <td>42</td>
      <td>F</td>
      <td>homemaker</td>
      <td>95660</td>
    </tr>
    <tr>
      <th>20</th>
      <td>21</td>
      <td>26</td>
      <td>M</td>
      <td>writer</td>
      <td>30068</td>
    </tr>
    <tr>
      <th>21</th>
      <td>22</td>
      <td>25</td>
      <td>M</td>
      <td>writer</td>
      <td>40206</td>
    </tr>
    <tr>
      <th>22</th>
      <td>23</td>
      <td>30</td>
      <td>F</td>
      <td>artist</td>
      <td>48197</td>
    </tr>
    <tr>
      <th>23</th>
      <td>24</td>
      <td>21</td>
      <td>F</td>
      <td>artist</td>
      <td>94533</td>
    </tr>
    <tr>
      <th>24</th>
      <td>25</td>
      <td>39</td>
      <td>M</td>
      <td>engineer</td>
      <td>55107</td>
    </tr>
  </tbody>
</table>
</div>



### Step 5. See the last 10 entries


```python
#users[-10:]
users.tail(10)
#I believe both are correct
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
      <th>user_id</th>
      <th>age</th>
      <th>gender</th>
      <th>occupation</th>
      <th>zip_code</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>932</th>
      <td>933</td>
      <td>28</td>
      <td>M</td>
      <td>student</td>
      <td>48105</td>
    </tr>
    <tr>
      <th>933</th>
      <td>934</td>
      <td>61</td>
      <td>M</td>
      <td>engineer</td>
      <td>22902</td>
    </tr>
    <tr>
      <th>934</th>
      <td>935</td>
      <td>42</td>
      <td>M</td>
      <td>doctor</td>
      <td>66221</td>
    </tr>
    <tr>
      <th>935</th>
      <td>936</td>
      <td>24</td>
      <td>M</td>
      <td>other</td>
      <td>32789</td>
    </tr>
    <tr>
      <th>936</th>
      <td>937</td>
      <td>48</td>
      <td>M</td>
      <td>educator</td>
      <td>98072</td>
    </tr>
    <tr>
      <th>937</th>
      <td>938</td>
      <td>38</td>
      <td>F</td>
      <td>technician</td>
      <td>55038</td>
    </tr>
    <tr>
      <th>938</th>
      <td>939</td>
      <td>26</td>
      <td>F</td>
      <td>student</td>
      <td>33319</td>
    </tr>
    <tr>
      <th>939</th>
      <td>940</td>
      <td>32</td>
      <td>M</td>
      <td>administrator</td>
      <td>02215</td>
    </tr>
    <tr>
      <th>940</th>
      <td>941</td>
      <td>20</td>
      <td>M</td>
      <td>student</td>
      <td>97229</td>
    </tr>
    <tr>
      <th>941</th>
      <td>942</td>
      <td>48</td>
      <td>F</td>
      <td>librarian</td>
      <td>78209</td>
    </tr>
    <tr>
      <th>942</th>
      <td>943</td>
      <td>22</td>
      <td>M</td>
      <td>student</td>
      <td>77841</td>
    </tr>
  </tbody>
</table>
</div>



### Step 6. What is the number of observations in the dataset?


```python

users.info()
#more tidy way:
len(users)
#or even:
users.shape[0]
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 943 entries, 0 to 942
    Data columns (total 5 columns):
     #   Column      Non-Null Count  Dtype 
    ---  ------      --------------  ----- 
     0   user_id     943 non-null    int64 
     1   age         943 non-null    int64 
     2   gender      943 non-null    object
     3   occupation  943 non-null    object
     4   zip_code    943 non-null    object
    dtypes: int64(2), object(3)
    memory usage: 37.0+ KB
    




    943



### Step 7. What is the number of columns in the dataset?


```python
len(users.columns)
```




    5



### Step 8. Print the name of all the columns.


```python
print(users.columns)
#more tidy way:
print(list(users.columns))
```

    Index(['user_id', 'age', 'gender', 'occupation', 'zip_code'], dtype='object')
    ['user_id', 'age', 'gender', 'occupation', 'zip_code']
    

### Step 9. How is the dataset indexed?


```python
users.index
```




    RangeIndex(start=0, stop=943, step=1)



### Step 10. What is the data type of each column?


```python
users.info()
#more tidy way:
print(users.dtypes)
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 943 entries, 0 to 942
    Data columns (total 5 columns):
     #   Column      Non-Null Count  Dtype 
    ---  ------      --------------  ----- 
     0   user_id     943 non-null    int64 
     1   age         943 non-null    int64 
     2   gender      943 non-null    object
     3   occupation  943 non-null    object
     4   zip_code    943 non-null    object
    dtypes: int64(2), object(3)
    memory usage: 37.0+ KB
    user_id        int64
    age            int64
    gender        object
    occupation    object
    zip_code      object
    dtype: object
    

### Step 11. Print only the occupation column


```python
print(users['occupation'])
#or
print(users.occupation)
```

    0         technician
    1              other
    2             writer
    3         technician
    4              other
               ...      
    938          student
    939    administrator
    940          student
    941        librarian
    942          student
    Name: occupation, Length: 943, dtype: object
    0         technician
    1              other
    2             writer
    3         technician
    4              other
               ...      
    938          student
    939    administrator
    940          student
    941        librarian
    942          student
    Name: occupation, Length: 943, dtype: object
    

### Step 12. How many different occupations are in this dataset?


```python
users.occupation.nunique()
#or
users['occupation'].nunique()
```




    21



### Step 13. What is the most frequent occupation?


```python
#to see the values 
users.occupation.value_counts()

#to only see the name of the most frequent occupation
users.occupation.value_counts().idxmax()
```




    'student'



### Step 14. Summarize the DataFrame.


```python
users.describe()
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
      <th>user_id</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>943.000000</td>
      <td>943.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>472.000000</td>
      <td>34.051962</td>
    </tr>
    <tr>
      <th>std</th>
      <td>272.364951</td>
      <td>12.192740</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>7.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>236.500000</td>
      <td>25.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>472.000000</td>
      <td>31.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>707.500000</td>
      <td>43.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>943.000000</td>
      <td>73.000000</td>
    </tr>
  </tbody>
</table>
</div>



### Step 15. Summarize all the columns


```python
users.describe(include='all')
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
      <th>user_id</th>
      <th>age</th>
      <th>gender</th>
      <th>occupation</th>
      <th>zip_code</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>943.000000</td>
      <td>943.000000</td>
      <td>943</td>
      <td>943</td>
      <td>943</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>2</td>
      <td>21</td>
      <td>795</td>
    </tr>
    <tr>
      <th>top</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>M</td>
      <td>student</td>
      <td>55414</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>670</td>
      <td>196</td>
      <td>9</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>472.000000</td>
      <td>34.051962</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>std</th>
      <td>272.364951</td>
      <td>12.192740</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>7.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>236.500000</td>
      <td>25.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>472.000000</td>
      <td>31.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>707.500000</td>
      <td>43.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>max</th>
      <td>943.000000</td>
      <td>73.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



### Step 16. Summarize only the occupation column


```python
users.occupation.describe()
#or
users['occupation'].describe()

```




    count         943
    unique         21
    top       student
    freq          196
    Name: occupation, dtype: object



### Step 17. What is the mean age of users?


```python
print(users.age.mean())
#or
print(users['age'].mean())
```

    34.05196182396607
    34.05196182396607
    

### Step 18. What is the age with least occurrence?


```python
users.age.value_counts(ascending=True)
```




    age
    73     1
    10     1
    66     1
    7      1
    11     1
          ..
    27    35
    28    36
    22    37
    25    38
    30    39
    Name: count, Length: 61, dtype: int64


# Exercise 2. - Filtering and Sorting Data

Check out [Euro 12 Exercises Video Tutorial](https://youtu.be/iqk5d48Qisg) to watch a data scientist go through the exercises

This time we are going to pull data directly from the internet.

### Step 1. Import the necessary libraries


```python
import pandas as pd
```

### Step 2. Import the dataset from this [address](https://raw.githubusercontent.com/kflisikowsky/pandas_exercises/refs/heads/main/Euro_2012_stats_TEAM.csv). 


```python
url = 'https://raw.githubusercontent.com/kflisikowsky/pandas_exercises/refs/heads/main/Euro_2012_stats_TEAM.csv'
df = pd.read_csv(url, sep = ',')
```

### Step 3. Assign it to a variable called euro12.


```python
euro12 = df
```

### Step 4. Select only the Goal column.


```python
euro12.Goals
#or
euro12['Goals']
```




    0      4
    1      4
    2      4
    3      5
    4      3
    5     10
    6      5
    7      6
    8      2
    9      2
    10     6
    11     1
    12     5
    13    12
    14     5
    15     2
    Name: Goals, dtype: int64



### Step 5. How many team participated in the Euro2012?


```python
print(euro12.Team.count())
#or
print(euro12.shape[0])
```

    16
    

### Step 6. What is the number of columns in the dataset?


```python
len(euro12.columns)
#or
euro12.shape[1]
```




    35



### Step 7. View only the columns Team, Yellow Cards and Red Cards and assign them to a dataframe called discipline


```python
discipline = euro12[['Team','Yellow Cards','Red Cards']]
print(discipline)
```

                       Team  Yellow Cards  Red Cards
    0               Croatia             9          0
    1        Czech Republic             7          0
    2               Denmark             4          0
    3               England             5          0
    4                France             6          0
    5               Germany             4          0
    6                Greece             9          1
    7                 Italy            16          0
    8           Netherlands             5          0
    9                Poland             7          1
    10             Portugal            12          0
    11  Republic of Ireland             6          1
    12               Russia             6          0
    13                Spain            11          0
    14               Sweden             7          0
    15              Ukraine             5          0
    

### Step 8. Sort the teams by Red Cards, then to Yellow Cards


```python
print(discipline.sort_values(['Red Cards', 'Yellow Cards'], ascending=False))
```

                       Team  Yellow Cards  Red Cards
    6                Greece             9          1
    9                Poland             7          1
    11  Republic of Ireland             6          1
    7                 Italy            16          0
    10             Portugal            12          0
    13                Spain            11          0
    0               Croatia             9          0
    1        Czech Republic             7          0
    14               Sweden             7          0
    4                France             6          0
    12               Russia             6          0
    3               England             5          0
    8           Netherlands             5          0
    15              Ukraine             5          0
    2               Denmark             4          0
    5               Germany             4          0
    

### Step 9. Calculate the mean Yellow Cards given per Team


```python
discipline['Yellow Cards'].mean()
```




    np.float64(7.4375)



### Step 10. Filter teams that scored more than 6 goals


```python
euro12[euro12.Goals > 6]
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
      <th>Team</th>
      <th>Goals</th>
      <th>Shots on target</th>
      <th>Shots off target</th>
      <th>Shooting Accuracy</th>
      <th>% Goals-to-shots</th>
      <th>Total shots (inc. Blocked)</th>
      <th>Hit Woodwork</th>
      <th>Penalty goals</th>
      <th>Penalties not scored</th>
      <th>...</th>
      <th>Saves made</th>
      <th>Saves-to-shots ratio</th>
      <th>Fouls Won</th>
      <th>Fouls Conceded</th>
      <th>Offsides</th>
      <th>Yellow Cards</th>
      <th>Red Cards</th>
      <th>Subs on</th>
      <th>Subs off</th>
      <th>Players Used</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>Germany</td>
      <td>10</td>
      <td>32</td>
      <td>32</td>
      <td>47.8%</td>
      <td>15.6%</td>
      <td>80</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>10</td>
      <td>62.6%</td>
      <td>63</td>
      <td>49</td>
      <td>12</td>
      <td>4</td>
      <td>0</td>
      <td>15</td>
      <td>15</td>
      <td>17</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Spain</td>
      <td>12</td>
      <td>42</td>
      <td>33</td>
      <td>55.9%</td>
      <td>16.0%</td>
      <td>100</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>15</td>
      <td>93.8%</td>
      <td>102</td>
      <td>83</td>
      <td>19</td>
      <td>11</td>
      <td>0</td>
      <td>17</td>
      <td>17</td>
      <td>18</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 35 columns</p>
</div>



### Step 11. Select the teams that start with G


```python
euro12[euro12.Team.str.startswith('G')]
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
      <th>Team</th>
      <th>Goals</th>
      <th>Shots on target</th>
      <th>Shots off target</th>
      <th>Shooting Accuracy</th>
      <th>% Goals-to-shots</th>
      <th>Total shots (inc. Blocked)</th>
      <th>Hit Woodwork</th>
      <th>Penalty goals</th>
      <th>Penalties not scored</th>
      <th>...</th>
      <th>Saves made</th>
      <th>Saves-to-shots ratio</th>
      <th>Fouls Won</th>
      <th>Fouls Conceded</th>
      <th>Offsides</th>
      <th>Yellow Cards</th>
      <th>Red Cards</th>
      <th>Subs on</th>
      <th>Subs off</th>
      <th>Players Used</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>Germany</td>
      <td>10</td>
      <td>32</td>
      <td>32</td>
      <td>47.8%</td>
      <td>15.6%</td>
      <td>80</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>10</td>
      <td>62.6%</td>
      <td>63</td>
      <td>49</td>
      <td>12</td>
      <td>4</td>
      <td>0</td>
      <td>15</td>
      <td>15</td>
      <td>17</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Greece</td>
      <td>5</td>
      <td>8</td>
      <td>18</td>
      <td>30.7%</td>
      <td>19.2%</td>
      <td>32</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>13</td>
      <td>65.1%</td>
      <td>67</td>
      <td>48</td>
      <td>12</td>
      <td>9</td>
      <td>1</td>
      <td>12</td>
      <td>12</td>
      <td>20</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 35 columns</p>
</div>



### Step 12. Select the first 7 columns


```python
euro12.iloc[:, :7]
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
      <th>Team</th>
      <th>Goals</th>
      <th>Shots on target</th>
      <th>Shots off target</th>
      <th>Shooting Accuracy</th>
      <th>% Goals-to-shots</th>
      <th>Total shots (inc. Blocked)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Croatia</td>
      <td>4</td>
      <td>13</td>
      <td>12</td>
      <td>51.9%</td>
      <td>16.0%</td>
      <td>32</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Czech Republic</td>
      <td>4</td>
      <td>13</td>
      <td>18</td>
      <td>41.9%</td>
      <td>12.9%</td>
      <td>39</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Denmark</td>
      <td>4</td>
      <td>10</td>
      <td>10</td>
      <td>50.0%</td>
      <td>20.0%</td>
      <td>27</td>
    </tr>
    <tr>
      <th>3</th>
      <td>England</td>
      <td>5</td>
      <td>11</td>
      <td>18</td>
      <td>50.0%</td>
      <td>17.2%</td>
      <td>40</td>
    </tr>
    <tr>
      <th>4</th>
      <td>France</td>
      <td>3</td>
      <td>22</td>
      <td>24</td>
      <td>37.9%</td>
      <td>6.5%</td>
      <td>65</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Germany</td>
      <td>10</td>
      <td>32</td>
      <td>32</td>
      <td>47.8%</td>
      <td>15.6%</td>
      <td>80</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Greece</td>
      <td>5</td>
      <td>8</td>
      <td>18</td>
      <td>30.7%</td>
      <td>19.2%</td>
      <td>32</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Italy</td>
      <td>6</td>
      <td>34</td>
      <td>45</td>
      <td>43.0%</td>
      <td>7.5%</td>
      <td>110</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Netherlands</td>
      <td>2</td>
      <td>12</td>
      <td>36</td>
      <td>25.0%</td>
      <td>4.1%</td>
      <td>60</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Poland</td>
      <td>2</td>
      <td>15</td>
      <td>23</td>
      <td>39.4%</td>
      <td>5.2%</td>
      <td>48</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Portugal</td>
      <td>6</td>
      <td>22</td>
      <td>42</td>
      <td>34.3%</td>
      <td>9.3%</td>
      <td>82</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Republic of Ireland</td>
      <td>1</td>
      <td>7</td>
      <td>12</td>
      <td>36.8%</td>
      <td>5.2%</td>
      <td>28</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Russia</td>
      <td>5</td>
      <td>9</td>
      <td>31</td>
      <td>22.5%</td>
      <td>12.5%</td>
      <td>59</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Spain</td>
      <td>12</td>
      <td>42</td>
      <td>33</td>
      <td>55.9%</td>
      <td>16.0%</td>
      <td>100</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Sweden</td>
      <td>5</td>
      <td>17</td>
      <td>19</td>
      <td>47.2%</td>
      <td>13.8%</td>
      <td>39</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Ukraine</td>
      <td>2</td>
      <td>7</td>
      <td>26</td>
      <td>21.2%</td>
      <td>6.0%</td>
      <td>38</td>
    </tr>
  </tbody>
</table>
</div>



### Step 13. Select all columns except the last 3.


```python
euro12.iloc[:, :-3]
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
      <th>Team</th>
      <th>Goals</th>
      <th>Shots on target</th>
      <th>Shots off target</th>
      <th>Shooting Accuracy</th>
      <th>% Goals-to-shots</th>
      <th>Total shots (inc. Blocked)</th>
      <th>Hit Woodwork</th>
      <th>Penalty goals</th>
      <th>Penalties not scored</th>
      <th>...</th>
      <th>Clean Sheets</th>
      <th>Blocks</th>
      <th>Goals conceded</th>
      <th>Saves made</th>
      <th>Saves-to-shots ratio</th>
      <th>Fouls Won</th>
      <th>Fouls Conceded</th>
      <th>Offsides</th>
      <th>Yellow Cards</th>
      <th>Red Cards</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Croatia</td>
      <td>4</td>
      <td>13</td>
      <td>12</td>
      <td>51.9%</td>
      <td>16.0%</td>
      <td>32</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>10</td>
      <td>3</td>
      <td>13</td>
      <td>81.3%</td>
      <td>41</td>
      <td>62</td>
      <td>2</td>
      <td>9</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Czech Republic</td>
      <td>4</td>
      <td>13</td>
      <td>18</td>
      <td>41.9%</td>
      <td>12.9%</td>
      <td>39</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>10</td>
      <td>6</td>
      <td>9</td>
      <td>60.1%</td>
      <td>53</td>
      <td>73</td>
      <td>8</td>
      <td>7</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Denmark</td>
      <td>4</td>
      <td>10</td>
      <td>10</td>
      <td>50.0%</td>
      <td>20.0%</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>10</td>
      <td>5</td>
      <td>10</td>
      <td>66.7%</td>
      <td>25</td>
      <td>38</td>
      <td>8</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>England</td>
      <td>5</td>
      <td>11</td>
      <td>18</td>
      <td>50.0%</td>
      <td>17.2%</td>
      <td>40</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>2</td>
      <td>29</td>
      <td>3</td>
      <td>22</td>
      <td>88.1%</td>
      <td>43</td>
      <td>45</td>
      <td>6</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>France</td>
      <td>3</td>
      <td>22</td>
      <td>24</td>
      <td>37.9%</td>
      <td>6.5%</td>
      <td>65</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>7</td>
      <td>5</td>
      <td>6</td>
      <td>54.6%</td>
      <td>36</td>
      <td>51</td>
      <td>5</td>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Germany</td>
      <td>10</td>
      <td>32</td>
      <td>32</td>
      <td>47.8%</td>
      <td>15.6%</td>
      <td>80</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>11</td>
      <td>6</td>
      <td>10</td>
      <td>62.6%</td>
      <td>63</td>
      <td>49</td>
      <td>12</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Greece</td>
      <td>5</td>
      <td>8</td>
      <td>18</td>
      <td>30.7%</td>
      <td>19.2%</td>
      <td>32</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>23</td>
      <td>7</td>
      <td>13</td>
      <td>65.1%</td>
      <td>67</td>
      <td>48</td>
      <td>12</td>
      <td>9</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Italy</td>
      <td>6</td>
      <td>34</td>
      <td>45</td>
      <td>43.0%</td>
      <td>7.5%</td>
      <td>110</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>2</td>
      <td>18</td>
      <td>7</td>
      <td>20</td>
      <td>74.1%</td>
      <td>101</td>
      <td>89</td>
      <td>16</td>
      <td>16</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Netherlands</td>
      <td>2</td>
      <td>12</td>
      <td>36</td>
      <td>25.0%</td>
      <td>4.1%</td>
      <td>60</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>9</td>
      <td>5</td>
      <td>12</td>
      <td>70.6%</td>
      <td>35</td>
      <td>30</td>
      <td>3</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Poland</td>
      <td>2</td>
      <td>15</td>
      <td>23</td>
      <td>39.4%</td>
      <td>5.2%</td>
      <td>48</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>8</td>
      <td>3</td>
      <td>6</td>
      <td>66.7%</td>
      <td>48</td>
      <td>56</td>
      <td>3</td>
      <td>7</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Portugal</td>
      <td>6</td>
      <td>22</td>
      <td>42</td>
      <td>34.3%</td>
      <td>9.3%</td>
      <td>82</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>2</td>
      <td>11</td>
      <td>4</td>
      <td>10</td>
      <td>71.5%</td>
      <td>73</td>
      <td>90</td>
      <td>10</td>
      <td>12</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Republic of Ireland</td>
      <td>1</td>
      <td>7</td>
      <td>12</td>
      <td>36.8%</td>
      <td>5.2%</td>
      <td>28</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>23</td>
      <td>9</td>
      <td>17</td>
      <td>65.4%</td>
      <td>43</td>
      <td>51</td>
      <td>11</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Russia</td>
      <td>5</td>
      <td>9</td>
      <td>31</td>
      <td>22.5%</td>
      <td>12.5%</td>
      <td>59</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>8</td>
      <td>3</td>
      <td>10</td>
      <td>77.0%</td>
      <td>34</td>
      <td>43</td>
      <td>4</td>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Spain</td>
      <td>12</td>
      <td>42</td>
      <td>33</td>
      <td>55.9%</td>
      <td>16.0%</td>
      <td>100</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>5</td>
      <td>8</td>
      <td>1</td>
      <td>15</td>
      <td>93.8%</td>
      <td>102</td>
      <td>83</td>
      <td>19</td>
      <td>11</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Sweden</td>
      <td>5</td>
      <td>17</td>
      <td>19</td>
      <td>47.2%</td>
      <td>13.8%</td>
      <td>39</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>12</td>
      <td>5</td>
      <td>8</td>
      <td>61.6%</td>
      <td>35</td>
      <td>51</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Ukraine</td>
      <td>2</td>
      <td>7</td>
      <td>26</td>
      <td>21.2%</td>
      <td>6.0%</td>
      <td>38</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>4</td>
      <td>4</td>
      <td>13</td>
      <td>76.5%</td>
      <td>48</td>
      <td>31</td>
      <td>4</td>
      <td>5</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>16 rows × 32 columns</p>
</div>



### Step 14. Present only the Shooting Accuracy from England, Italy and Russia


```python
euro12.set_index('Team', inplace=True)
euro12.loc[['England', 'Italy', 'Russia'], 'Shooting Accuracy']
```




    Team
    England    50.0%
    Italy      43.0%
    Russia     22.5%
    Name: Shooting Accuracy, dtype: object


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


