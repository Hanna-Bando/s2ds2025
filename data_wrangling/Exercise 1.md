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


