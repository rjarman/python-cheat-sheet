# Python Cheat Sheet for Data Science

> Introduction to Python

- `list[starting(included):ending(excluded)]`
- `list_1[...] + list_2[...] = [......]` make new list by concatenating both list
- `del(list[-4:-2])` will delete fourth and third element from bottom
- `list()` help to create a list and make a copy of another list with different memory address
- `complex(re, img)` help to make complex numbers
- `sorted(iterable, reverse=False)`

###### String Methods

- `str.upper()`
- `str.count()` finds the number of given character present in the string

###### List Methods

- `list.index()` finds the given value's index
- `list.count()` finds the number of given value present in the list
- `list.append()` it add the value to the end of the existing list but does not work like `np.append`
- `list.reverse()` it reverse the existing list

###### NumPy

- Some differences between `list` and `numpy`
  ```python
  [1, 2, 3] * 2 >> [1, 2, 3, 1, 2, 3]
  np.array([1, 2, 3]) * 2 >> array([2, 4, 6])
  ```
  - `list` does not support other arithmetic operators like +, -, / but `numpy` operates all of them element wise.
    ```python
    a = np.arange(1.0, 100.0, 12.546)
    b = a < 35
    b[c]
    ```
    ```python
    [4, 3, 0] + [0, 2, 2] >> [4, 3, 0, 0, 2, 2]
    np.array([4, 3, 0]) + np.array([0, 2, 2]) >> array([4, 5, 2])
    ```
  - Boolean subsetting
    ```python
    a = np.array([191, 184, 185, 180, 181])
    b = np.array(['GK', 'M', 'A', 'D', 'M'])
    c = a[b == 'M'] >> array([184, 181])
    d = a[b != 'M'] >> array([191, 185, 180])
    ```
- Subsetting for multidimensional array because one dimensional rules are same as `list`
  - `mul_arr[49, :]` selects 50th row
  - `mul_arr[:, 1]` selects second column of every row
  - `mul_arr[123, 0]` selects 124th 1st column value
- Statistics
  - `np.mean()`
  - `np.median()`
  - `np.std()`
  - `np.corrcoef()`
- Hacker Statistics
  - `np.random.seed()`
  - `np.random.rand()` generates random float
  - `np.random.randint(start(include), end(exclude))` generates random integer
  - `np.transpose()`
  - `np.random.rand()` generates random value between 0 and 1
- Logical operations
  - `np.logical_and()`
  - `np.logical_or()`
  - `np.logical_not()`


> Intermediate Python

###### Matplotlib

- `plt.plot()` for line plot
- `plt.scatter()`
  `plt.scatter(x, y, s = [...], c=[...], alpha=0.8)` here s defines size of the dots, c defines various dot color list it can be different color for the different values and alpha defines the depth of the color
- `plt.xscale()` for setting the horizontal scaling size
- `plt.show()`
- `plt.hist()`
- `plt.clf()` for clearing the current plotting view otherwise if there are two `plt.show()` without `plt.subplot()` then those will be overlapped
- `plt.xlabel()`
- `plt.ylabel()`
- `plt.title()`
- `plt.xticks(tick_val, tick_lab)`
  ```python
  tick_val = [1000, 10000, 100000]
  tick_lab = ['1k', '10k', '100k']
  ```
  then it will mark the horizontal axis on 1000, 10000, 100000 as 1k, 10k, 100k
- `plt.text(x, y, 'text')` it will write a text on (x, y) positions
- `plt.grid()` defines whether there need to put grid lines on view or not

###### Dictionaries

- `dict.values()`
- `dict.keys()`
- `del(dict[key])`

###### Pandas

- `pd.DataFrame(dict)`
- `pd.DataFrame(dict).index` changes `index` values
  ```python
  names = ['United States', 'Australia', 'Japan', 'India', 'Russia', 'Morocco', 'Egypt']
  dr = [True, False, False, False, True, True, True]
  cpc = [809, 731, 588, 18, 200, 70, 45]
  cars_dict = { 'country':names, 'drives_right':dr, 'cars_per_cap':cpc }
  cars = pd.DataFrame(cars_dict)
  row_labels = ['US', 'AUS', 'JPN', 'IN', 'RU', 'MOR', 'EG']
  cars.index = row_labels
  ```
- `pd.read_csv('name.csv')`
- `pd.read_csv('name.csv', index_col=0)` reads without index values
- Some code examples

  - Subsetting

    ```python
    # Print out country column as Pandas Series
      print(cars['country'])
      """
      US     United States
      AUS        Australia
      JPN            Japan
      IN             India
      RU            Russia
      MOR          Morocco
      EG             Egypt
      Name: country, dtype: object
      """

    # Print out country column as Pandas DataFrame
      print(cars[['country']])
      """
                country
      US   United States
      AUS      Australia
      JPN          Japan
      IN           India
      RU          Russia
      MOR        Morocco
      EG           Egypt
      """

    # Print out DataFrame with country and drives_right columns
      print(cars[['country', 'drives_right']])
      """
                country  drives_right
      US   United States          True
      AUS      Australia         False
      JPN          Japan         False
      IN           India         False
      RU          Russia          True
      MOR        Morocco          True
      EG           Egypt          True
      """
    ```

    ```python
    # Import cars data
    import pandas as pd
    cars = pd.read_csv('cars.csv', index_col = 0)

    # Print out first 3 observations
    print(cars[:3])
    """
        cars_per_cap        country  drives_right
    US            809  United States          True
    AUS           731      Australia         False
    JPN           588          Japan         False
    """

    # Print out fourth, fifth and sixth observation
    print(cars[3:6])
    """
        cars_per_cap  country  drives_right
    IN             18    India         False
    RU            200   Russia          True
    MOR            70  Morocco          True
    """
    ```

- `loc`

  ```python
  # Import cars data
  import pandas as pd
  cars = pd.read_csv('cars.csv', index_col = 0)

  # Print out observation for Japan
  print(cars.loc['JPN'])
  """
  cars_per_cap      588
  country         Japan
  drives_right    False
  Name: JPN, dtype: object
  """

  # Print out observations for Australia and Egypt
  print(cars.loc[['AUS', 'EG']])
  """
      cars_per_cap    country  drives_right
  AUS           731  Australia         False
  EG             45      Egypt          True
  """

  # Print out drives_right value of Morocco
  print(cars.loc[['MOR'], ['drives_right']])
  """
       drives_right
  MOR          True
  """

  # Print sub-DataFrame
  print(cars.loc[['RU', 'MOR'], ['country', 'drives_right']])
  """
     country  drives_right
  RU    Russia          True
  MOR  Morocco          True
  """
  ```

- `loc` and `iloc`

  ```python
  cars.loc['IN', 'cars_per_cap']
  cars.iloc[3, 0]

  cars.loc[['IN', 'RU'], 'cars_per_cap']
  cars.iloc[[3, 4], 0]

  cars.loc[['IN', 'RU'], ['cars_per_cap', 'country']]
  cars.iloc[[3, 4], [0, 1]]

  cars.loc[:, 'country']
  cars.iloc[:, 1]

  cars.loc[:, ['country','drives_right']]
  cars.iloc[:, [1, 2]]

  # Print out drives_right column as Series
  print(cars.loc[:, 'drives_right'])
  """
  US      True
  AUS    False
  JPN    False
  IN     False
  RU      True
  MOR     True
  EG      True
  Name: drives_right, dtype: bool
  """

  # Print out drives_right column as DataFrame
  print(cars.loc[:, ['drives_right']])
  """
       drives_right
  US           True
  AUS         False
  JPN         False
  IN          False
  RU           True
  MOR          True
  EG           True
  """

  # Print out cars_per_cap and drives_right as DataFrame
  print(cars.loc[:, ['cars_per_cap', 'drives_right']])
  """
       cars_per_cap  drives_right
  US            809          True
  AUS           731         False
  JPN           588         False
  IN             18         False
  RU            200          True
  MOR            70          True
  EG             45          True
  """
  ```

- Conditions with `DataFrame`

  ```python
  #Extract drives_right column as Series: dr
  dr = cars['drives_right']

  # Use dr to subset cars: sel
  sel = cars[dr]

  # Print sel
  print(sel)
  """
       cars_per_cap        country  drives_right
  US            809  United States          True
  RU            200         Russia          True
  MOR            70        Morocco          True
  EG             45          Egypt          True
  """

  # Create car_maniac: observations that have a cars_per_cap over 500
  car_maniac = cars[cars['cars_per_cap'] > 500]

  # Print car_maniac
  print(car_maniac)
  """
       cars_per_cap        country  drives_right
  US            809  United States          True
  AUS           731      Australia         False
  JPN           588          Japan         False
  """

  # Create medium: observations with cars_per_cap between 100 and 500
  medium = cars[np.logical_and(cars['cars_per_cap'] > 100, cars['cars_per_cap'] < 500)]

  # Print medium
  print(medium)
  """
      cars_per_cap country  drives_right
  RU           200  Russia          True
  """
  ```

- `pd.DataFrame()[...].apply()` applies a function to all the selected rows or columns

  ```python
  # Import cars data
  import pandas as pd
  cars = pd.read_csv('cars.csv', index_col = 0)

  # Use .apply(str.upper)
  cars['COUNTRY'] = cars['country'].apply(str.upper)
  print(cars)
  """
       cars_per_cap        country  drives_right        COUNTRY
  US            809  United States          True  UNITED STATES
  AUS           731      Australia         False      AUSTRALIA
  JPN           588          Japan         False          JAPAN
  """
  ```

###### Looping

- `enumerate(...)` gives index and value respectively
- `dict.items()` gives key and value respectively
- `np.nditer()` helps to loop over multidimensional arrays

  ```python
  # Import numpy as np
  import numpy as np

  # For loop over np_height
  print(np_height[:5])
  # [74 74 72]

  for h in np.nditer(np_height[:5]):
      print(str(h) + ' inches')
  """
  74 inches
  74 inches
  72 inches
  """

  # For loop over np_baseball
  print(np_baseball[:3])
  """
  [[ 74 180]
  [ 74 215]
  [ 72 210]
  """
  for b in np.nditer(np_baseball[:5]):
      print(b)
  """
  74
  74
  72
  180
  215
  210
  """
  ```

- `pd.DataFrame().iterrows()` loops over DataFrame

  ```python
  # Import cars data
  import pandas as pd
  cars = pd.read_csv('cars.csv', index_col = 0)[:2]

  # Iterate over rows of cars
  for lab, row in cars.iterrows():
      print(lab)
      print(row)
  """
  US
  cars_per_cap              809
  country         United States
  drives_right             True
  Name: US, dtype: object
  AUS
  cars_per_cap          731
  country         Australia
  drives_right        False
  Name: AUS, dtype: object
  """

  # Adapt for loop
  for lab, row in cars.iterrows() :
      print(lab + ': ' + str(row['cars_per_cap']))
  """
  US: 809
  AUS: 731
  """

  # Code for loop that adds COUNTRY column
  for lab, row in cars.iterrows():
      cars.loc[lab, 'COUNTRY'] = row['country'].upper()
  ```
