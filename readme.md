# Python Cheat Sheet for Data Science

> Introduction to Python

- `list[starting(included):ending(excluded)]`
- `list_1[...] + list_2[...] = [......]` make new list by concatenating both list
- `del(list[-4:-2])` will delete fourth and third element from bottom
- `list()` help to create a list and make a copy of another list with different memory address
- `complex(re, img)` help to make complex numbers
- `sorted(iterable, reverse=False)`
- `sum()`

###### String Methods

- `str.upper()`
- `str.count()` finds the number of given character present in the string

###### List Methods

- `list.index()` finds the given value's index
- `list.count()` finds the number of given value present in the list
- `list.append()` it add the value to the end of the existing list but does not work like `np.append`
- `list.reverse()` it reverse the existing list

###### NumPy

- `np.reshape(arr, newshape)`
- `np.linspace(start, stop)` returns evenly spaced numbers over a specified interval.
- `np.logspace(start, stop)` returns numbers spaced evenly on a log scale.
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
- `np.sqrt()`
- `np.sort()`
- `np.arange(start(include), stop(exclude), step)` returns evenly spaced values within a given interval
- Subsetting for multidimensional array because one dimensional rules are same as `list`
  - `mul_arr[49, :]` selects 50th row
  - `mul_arr[:, 1]` selects second column of every row
  - `mul_arr[123, 0]` selects 124th 1st column value
- Statistics

  - `np.mean()`
  - `np.median()`
  - `np.var()`

    - The mean squared distance of the data from their mean
    - Informally, a measure of the spread of data

    ![Variance formula](assets/variance_formula.png 'Variance Formula')

  - `np.std()`
    When we calculate the variance it involves square function so that to convert them to the same unit there is a term called standard deviation. It is clear that the standard deviation is a reasonable metric for the typical spread of the data.
  - `np.cov(..., ...)`
    We want a number that summarizes how Obama's vote share varies with the total vote count.
    There are two types:

    1. Positive
       If x and y both tend to be above, or both below their respective means together. This means that they are positively correlated. For example, when the county is populous, it has more votes for Obama.
    1. Negative
       If x is high while y is low, the covariance is negative, and the data are negatively correlated, or anti-correlated.

    ![Covariance formula](assets/covariance_formula.png 'Covariance Formula')

  - `np.corrcoef(..., ...)`
    It is dimensionless and ranges from -1 (for complete anti-correlation) to 1 (for complete correlation).
    ![CorrCoef formula](assets/corrcoef_formula.png 'CorrCoef Formula')
    For example:
    ![CorrCoef example](assets/corrcoef_example.png 'CorrCoef Example')
  - `np.percentile()`
    ```python
    # Specify array of percentiles: percentiles
    percentiles = np.array([2.5, 25, 50, 75, 97.5])
    # Compute percentiles: ptiles_vers
    ptiles_vers = np.percentile(versicolor_petal_length, percentiles)
    # Print the result
    print(ptiles_vers)
    ```

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
- `plt.margins()` makes sure none of the data points run over the side of the plot area. Choosing a value of .02 gives a 2% buffer all around the plot.
- `plt.legend()`

###### Dictionaries

- `dict.values()`
- `dict.keys()`
- `del(dict[key])`
- `dict` and `zip` converts `list` to `dict`

  ```python
  # Zip lists: zipped_lists
  zipped_lists = zip(['CountryName', 'CountryCode', 'IndicatorName', 'IndicatorCode', 'Year', 'Value'], ['Arab World', 'ARB', 'Adolescent fertility rate (births per 1,000 women ages 15-19)', 'SP.ADO.TFRT', '1960', '133.56090740552298'])

  # Create a dictionary: rs_dict
  rs_dict = dict(zipped_lists)

  # Print the dictionary
  print(rs_dict)
  # {'CountryName': 'Arab World', 'CountryCode': 'ARB', 'IndicatorName': 'Adolescent fertility rate (births per 1,000 women ages 15-19)', 'IndicatorCode': 'SP.ADO.TFRT', 'Year': '1960', 'Value': '133.56090740552298'}
  ```

###### Pandas

- `pd.DataFrame().info()`, `pd.DataFrame().head()`, `pd.DataFrame().describe()`
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
- `pd.read_csv(csv_file, chunksize=c_size)` reads `c_size` numbers of rows at a time
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

- Pandas and iterators

  ```python
  # Import the pandas package
  import pandas as pd

  # Initialize reader object: df_reader
  df_reader = pd.read_csv('ind_pop.csv', chunksize=10)

  # Print two chunks
  print(next(df_reader))
  print(next(df_reader))
  ```

- Pandas and plot
  ```python
  pd.DataFrame().plot(kind='scatter', x='Year', y='Total Urban Population')
  plt.show()
  ```
- `pd.dataFrame(...).boxplot(y_col_name, x_col_name, rot)`
- Correlation with `pandas`
  `corr()` measures the correlation coefficient of every pair in DataFrame.
  ```python
  sns.heatmap(pd.DataFrame().corr(), square=True, cmap='RdYlGn)
  ```
  ![Heatmap Sample](assets/corr_heatmap_sample.svg 'Heatmap Sample')

###### Looping

- `enumerate(...)` gives index and value respectively
- `zip()`

  ```python
  for value1, value2, value3 in zip([1, 2, 3], ['a', 'b', 'c'], [.5, .1, .15]):
    print(value1, value2, value3)
  """
  1 a 0.5
  2 b 0.1
  3 c 0.15
  """
  ```

- `*zip()` unzips the zip

  ```python
  mutants = ('charles xavier', 'bobby drake', 'kurt wagner', 'max eisenhardt', 'kitty pryde')
  powers = ('telepathy', 'thermokinesis', 'teleportation', 'magnetokinesis', 'intangibility')
  # Create a zip object from mutants and powers: z1
  z1 = zip(mutants, powers)

  # Print the tuples in z1 by unpacking with *
  print(*z1)

  # Re-create a zip object from mutants and powers: z1
  z1 = zip(mutants, powers)

  # 'Unzip' the tuples in z1 by unpacking with * and zip(): result1, result2
  result1, result2 = zip(*z1)

  # Check if unpacked tuples are equivalent to original tuples
  print(result1 == mutants)
  print(result2 == powers)
  # ('charles xavier', 'telepathy') ('bobby drake', 'thermokinesis') ('kurt wagner', 'teleportation') ('max eisenhardt', 'magnetokinesis') ('kitty pryde', 'intangibility')
  # True
  # True
  ```

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

> Python Data Science Toolbox (Part 1)

- `global` makes variable available globally
- `nonlocal` make nested function's variable available for its parent functions

  ```python
  # Define echo_shout()
  def echo_shout(word):
      """Change the value of a nonlocal variable"""

      # Concatenate word with itself: echo_word
      echo_word = word + word

      # Print echo_word
      print(echo_word)

      # Define inner function shout()
      def shout():
          """Alter a variable in the enclosing scope"""
          # Use echo_word in nonlocal scope
          nonlocal echo_word

          # Change echo_word to echo_word concatenated with '!!!'
          echo_word = echo_word + '!!!'

      # Call function shout()
      shout()

      # Print echo_word
      print(echo_word)

  # Call function echo_shout() with argument 'hello'
  echo_shout('hello')
  ```

- To get all the builtins in python
  ```python
  import builtins
  dir(builtins)
  ```
- `map(func, *iterables)`
- `lambda arg: statement`

  ```python
  # Create a list of strings: spells
  spells = ["protego", "accio", "expecto patronum", "legilimens"]

  # Use map() to apply a lambda function over spells: shout_spells
  shout_spells = map(lambda a: a + '!!!', spells)

  # Convert shout_spells to a list: shout_spells_list
  shout_spells_list = list(shout_spells)

  # Print the result
  print(shout_spells_list)
  # ['protego!!!', 'accio!!!', 'expecto patronum!!!', 'legilimens!!!']
  ```

- `filter(function or None, iterable)`

  ```python
  # Create a list of strings: fellowship
  fellowship = ['frodo', 'samwise', 'merry', 'pippin', 'aragorn', 'boromir', 'legolas', 'gimli', 'gandalf']

  # Use filter() to apply a lambda function over fellowship: result
  result = filter(lambda a: len(a) > 6, fellowship)

  # Convert result to a list: result_list
  result_list = list(result)

  # Print result_list
  print(result_list)
  # ['samwise', 'aragorn', 'boromir', 'legolas', 'gandalf']
  ```

- `functools.reduce(...)` is useful for performing some computation on a `list` and, unlike `map()` and `filter()`, returns a single value as a result.

  ```python
  # Import reduce from functools
  from functools import reduce

  # Create a list of strings: stark
  stark = ['robb', 'sansa', 'arya', 'brandon', 'rickon']

  # Use reduce() to apply a lambda function over stark: result
  result = reduce(lambda item1, item2: item1 + item2, stark)

  # Print the result
  print(result)
  # robbsansaaryabrandonrickon
  ```

> Python Data Science Toolbox (Part 2)

- `iter()` and `next()`

  ```python
  # Create a list of strings: flash
  flash = ['jay garrick', 'barry allen', 'wally west', 'bart allen']

  # Print each list item in flash using a for loop
  for i in flash:
      print(i)


  # Create an iterator for flash: superhero
  superhero = iter(flash)

  # Print each item from the iterator
  print(next(superhero))
  print(next(superhero))
  print(next(superhero))
  print(next(superhero))

  ```

###### List Comprehensions

- Dict comprehensions

  ```python
  # Create a list of strings: fellowship
  fellowship = ['frodo', 'samwise', 'merry', 'aragorn', 'legolas', 'boromir', 'gimli']

  # Create dict comprehension: new_fellowship
  new_fellowship = {v: len(v) for v in fellowship}

  # Print the new dictionary
  print(new_fellowship)
  ```

- List comprehensions

  ```python
  # Create a 5 x 5 matrix using a list of lists: matrix
  matrix = [[col for col in range(0, 5)] for row in range(0, 5)]

  # Print the matrix
  for row in matrix:
      print(row)
  ```

- Conditionals list comprehensions

  ```python
  # Create a list of strings: fellowship
  fellowship = ['frodo', 'samwise', 'merry', 'aragorn', 'legolas', 'boromir', 'gimli']

  # Create list comprehension: new_fellowship
  new_fellowship = [member for member in fellowship if len(member) >= 7]

  # Print the new list
  print(new_fellowship)

  """
  In the previous part, you used an if conditional statement in the predicate expression part of a list comprehension to evaluate an iterator variable. In this exercise, you will use an if-else statement on the output expression of the list.
  """
  # Create a list of strings: fellowship
  fellowship = ['frodo', 'samwise', 'merry', 'aragorn', 'legolas', 'boromir', 'gimli']

  # Create list comprehension: new_fellowship
  new_fellowship = [member if len(member) >= 7 else '' for member in fellowship]

  # Print the new list
  print(new_fellowship)
  ```

- `pd.get_dummies(df)` returns converted categorical data as dummy variables
- `pd.get_dummies(df, drop_first=True)` returns converted data as dummy variables from DataFrame but without the first categorical column
- `df.dropna()` drops the null values
- `df.isnull()` generates values as true or false as there anu nul values or not

###### Generator Expressions

- Examples

  ```python
  # Create generator object: result
  result = (num for num in range(31))

  # Print the first 5 values
  print(next(result))
  print(next(result))

  # Print the rest of the values
  for value in result:
      print(value)
  ```

- Custom generators

  ```python
  # Create a list of strings
  lannister = ['cersei', 'jaime', 'tywin', 'tyrion', 'joffrey']

  # Define generator function get_lengths
  def get_lengths(input_list):
      """Generator function that yields the
      length of the strings in input_list."""

      # Yield the length of a string
      for person in input_list:
          yield len(person)

  # Print the values generated by get_lengths()
  for value in get_lengths(lannister):
      print(value)
  ```

###### Files

- `file.readline()`
  ```python
  with open(file_name) as file:
    file.readline()
  ```
- File and generators

  ```python
  # Define read_large_file()
  def read_large_file(file_object):
      """A generator function to read a large file lazily."""

      # Loop indefinitely until the end of the file
      while True:

          # Read a line from the file: data
          data = file_object.readline()

          # Break if this is the end of the file
          if not data:
              break

          # Yield the line of data
          yield data
  # Open a connection to the file
  with open('world_dev_ind.csv') as file:

      # Create a generator object for the file: gen_file
      gen_file = read_large_file(file)

      # Print the first three lines of the file
      print(next(gen_file))
      print(next(gen_file))
      print(next(gen_file))

  # Initialize an empty dictionary: counts_dict
  counts_dict = {}
  # Open a connection to the file
  with open('world_dev_ind.csv') as file:

      # Iterate over the generator from read_large_file()
      for line in read_large_file(file):

          row = line.split(',')
          first_col = row[0]

          if first_col in counts_dict.keys():
              counts_dict[first_col] += 1
          else:
              counts_dict[first_col] = 1
  # Print
  print(counts_dict)
  ```

> Statistical Thinking in Python (Part 1)

###### Seaborn

- sns.set() sets `seaborn` as default style for `matplotlib`
  ```python
  # Import plotting modules
  import matplotlib.pyplot as plt, seaborn as sns
  # Set default Seaborn style
  sns.set()
  # Plot histogram of versicolor petal lengths
  plt.hist(versicolor_petal_length)
  # Show histogram
  plt.show()
  ```
- `sns.swarmplot(x='df_col_name', y='df_col_name', data=df)`
- `sns.boxplot(x='df_col_name', y='df_col_name', data=df)`
  ![Boxplot demo](assets/boxplot_demo.png 'Boxplot Demo')
- `sns.countplot(x, y, hue, data, palette)`
  ```python
  plt.figure()
  sns.countplot(x='missile', hue='party', data=df, palette='RdBu')
  plt.xticks([0,1], ['No', 'Yes'])
  plt.show()
  ```
- `sns.heatmap(data, square, cmap)`

###### Empirical cumulative distribution function (ECDF)

The bee swarm plot has a real problem. The edges have overlapping data points, which was necessary in order to fit all points onto the plot. We are now obfuscating data. So, using a bee swarm plot here is not the best option. As an alternative, we can compute an empirical cumulative distribution function, or ECDF.
![ECDF snapshot](assets/ecdf_snap.png 'An ECDF snapshot')

```python
def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    # Number of data points: n
    n = len(data)

    # x-data for the ECDF: x
    x = np.sort(data)

    # y-data for the ECDF: y
    y = np.arange(1, n + 1) / n

    return x, y
# Compute ECDFs
x_set, y_set = ecdf(setosa_petal_length)
x_vers, y_vers = ecdf(versicolor_petal_length)
x_virg, y_virg = ecdf(virginica_petal_length)

# Plot all ECDFs on the same plot
plt.plot(x_set, y_set, marker='.', linestyle='none')
plt.plot(x_vers, y_vers, marker='.', linestyle='none')
plt.plot(x_virg, y_virg, marker='.', linestyle='none')

# Annotate the plot
plt.legend(('setosa', 'versicolor', 'virginica'), loc='lower right')
_ = plt.xlabel('petal length (cm)')
_ = plt.ylabel('ECDF')

# Display the plot
plt.show()
```

![ECDF example](assets/ecdf_example.svg 'An ECDF example')

###### Statistical inference

It is the process through which inferences about a population are made based on certain statistics calculated from a sample of data drawn from that population.
![Statistical inference](assets/statistical_inference.gif 'Statistical Inference')

###### Hacker Statistics

- `np.random.seed()`
- `np.random.random()` returns random floats in the half-open interval `[0.0, 1.0)`
- `np.random.rand()` generates random float
- `np.random.randint(start(include), end(exclude))` generates random integer
- `np.transpose()`
- `np.random.rand()` generates random value between 0 and 1
- `np.empty()`

###### Probability Distribution(Discrete Variables)

A mathematical description of outcomes.

- Bernoulli Trials

  You can think of a Bernoulli trial as a flip of a possibly biased coin. Specifically, each coin flip has a probability of landing heads (success) and probability of landing tails (failure).

  ```python
  def perform_bernoulli_trials(n, p):
      """Perform n Bernoulli trials with success probability p
      and return number of successes."""
      # Initialize number of successes: n_success
      n_success = 0

      # Perform trials
      for i in range(n):
          # Choose random number between zero and one: random_number
          random_number = np.random.random()

          # If less than p, it's a success so add one to n_success
          if random_number < p:
              n_success+=1

      return n_success
  # example
  for i in range(1000):
      n_defaults[i] = perform_bernoulli_trials(100, 0.05)
  print(n_defaults[:5])
  # [6. 5. 7. 8. 5.]
  ```

- Probability Mas Function(PMF)
  The set of probabilities of discrete outcomes. It's a property of discrete probability distribution.
- Discrete uniform distribution
  The outcome of rolling a single fair die is
  - Discrete
  - Uniformly distributed
- Binomial distribution

  - The number r of successes in n Bernoulli trials with probability p of success, is Binomially distributed.
  - The number r of heads in 4 coin flips with probability 0.5 of heads, is Binomially distributed.
  - `np.random.binomial(n, p, size)` operates `size` number of samples with `n` number of bernoulli trials with probability of success `p`
  - The binomial PMF

  ```python
  # Compute bin edges: bins
  bins = np.arange(0, max(n_defaults) + 1.5) - 0.5
  # Generate histogram
  plt.hist(n_defaults, normed=True, bins=bins)
  # Label axes
  plt.xlabel('numbers of defaults')
  plt.ylabel('probability')
  # Show the plot
  plt.show()
  ```

  - The binomial CDF

  ```python
  # Take 10,000 samples out of the binomial distribution: n_defaults
  n_defaults = np.random.binomial(n=100, p=0.05, size=10000)
  # Compute CDF: x, y
  x, y = ecdf(n_defaults)
  # Plot the CDF with axis labels
  plt.plot(x, y, marker='.', linestyle='none')
  plt.xlabel('number of defaults')
  plt.ylabel('CDF')
  # Show the plot
  plt.show()
  ```

  ![Binomial CDF](assets/binomial_cdf.svg 'Binomial CDF')

- Poisson process
  A process is a poisson process if the timing of the next event is completely independent of when the previous event happened.
  Examples of poisson process
  - Natural births in a given hospital
  - Hit on a website during a given hour
  - Meteor strikes
  - Molecular collisions in a gas
  - Aviation incidents
  - Arrival of busses in a bus stand
- Poisson distribution
  - The number r of arrivals of a poisson process in a given time interval with average rate of ? arrivals per interval is poisson distributed.
  - The number r of hits on a website in one hour with an average hit rate of 6 hits per hour is poisson distributed.
  - It's a limit of the binomial distribution for low probability of success and large number of trials and also for rare events.
- Relationship between Binomial and Poisson distributions
  You just heard that the Poisson distribution is a limit of the Binomial distribution for rare events. This makes sense if you think about the stories. Say we do a Bernoulli trial every minute for an hour, each with a success probability of `0.1`. We would do `60` trials, and the number of successes is Binomially distributed, and we would expect to get about `6` successes. This is just like the Poisson story we discussed in the video, where we get on average `6` hits on a website per hour. So, the Poisson distribution with arrival rate equal to `np` approximates a Binomial distribution for `n` Bernoulli trials with probability `p` of success (with `n` large and `p` small). Importantly, the Poisson distribution is often simpler to work with because it has only one parameter instead of two for the Binomial distribution.

  ```python
  # Draw 10,000 samples out of Poisson distribution: samples_poisson
  samples_poisson = np.random.poisson(10, 10000)

  # Print the mean and standard deviation
  print('Poisson: ', np.mean(samples_poisson), np.std(samples_poisson))

  # Specify values of n and p to consider for Binomial: n, p
  n, p = [20, 100, 1000], [0.5, 0.1, 0.01]

  # Draw 10,000 samples for each n,p pair: samples_binomial
  for i in range(3):
      samples_binomial = np.random.binomial(n[i], p[i], 10000)

      # Print results
      print('n =', n[i], 'Binom:', np.mean(samples_binomial), np.std(samples_binomial))
  """
  Poisson:      10.0186 3.144813832327758
  n = 20 Binom: 9.9637 2.2163443572694206
  n = 100 Binom: 9.9947 3.0135812433050484
  n = 1000 Binom: 9.9985 3.139378561116833
  """
  ```

  Example:

  1. How many no-hitters in a season?
     In baseball, a no-hitter is a game in which a pitcher does not allow the other team to get a hit. This is a rare event, and since the beginning of the so-called modern era of baseball (starting in 1901), there have only been 251 of them through the 2015 season in over 200,000 games. The ECDF of the number of no-hitters in a season is shown to the right. Which probability distribution would be appropriate to describe the number of no-hitters we would expect in a given season?
     Ans: Both Binomial and Poisson, though Poisson is easier to model and compute.
  1. Was 2015 anomalous?
     1990 and 2015 featured the most no-hitters of any season of baseball (there were seven). Given that there are on average 251/115 no-hitters per season, what is the probability of having seven or more in a season?

     ```python
     # Draw 10,000 samples out of Poisson distribution: n_nohitters
     n_nohitters = np.random.poisson(251/115, 10000)

     # Compute number of samples that are seven or greater: n_large
     n_large = np.sum(n_nohitters >= 7)

     # Compute probability of getting seven or more: p_large
     p_large = n_large / len(n_nohitters)

     # Print the result
     print('Probability of seven or more no-hitters:', p_large)
     ```

###### Probability Distribution(Continuos Variables)

- Probability density function(PDF)

  - It's a continuos analog to the PMF
  - It's a mathematical description of the relative likelihood of observing a value of a continuous variables
    ![Normal PDF](assets/normal_pdf.png 'Normal PDF')
    Remember that the CDF gives the probability the measured speed of light will be less than the value on the x-axis.
    ![Normal CDF](assets/normal_cdf.png 'Normal CDF')

- Normal distribution
  Describes a continuous variable whose PDF has a single symmetric peak.
  ![Normal snap](assets/normal_snap.png 'Normal Distribution Snap')
  | Parameters of normal distribution | Status | Calculated from data while EDA |
  | --------------------------------- | ------ | ------------------------------ |
  | mean | &#8800;| mean |
  | standard deviation | &#8800;| standard deviation |

  - `np.random.normal(mean, std_deviation, size)`
  - The normal PDF

    ```python
    # Draw 100000 samples from Normal distribution with stds of interest: samples_std1, samples_std3, samples_std10
    samples_std1, samples_std3, samples_std10 = np.random.normal(20, 1, 100000), np.random.normal(20, 3, 100000), np.random.normal(20, 10, 100000)

    # Make histograms
    plt.hist(samples_std1, bins=100, normed=True, histtype='step')
    plt.hist(samples_std3, bins=100, normed=True, histtype='step')
    plt.hist(samples_std10, bins=100, normed=True, histtype='step')

    # Make a legend, set limits and show plot
    _ = plt.legend(('std = 1', 'std = 3', 'std = 10'))
    plt.ylim(-0.01, 0.42)
    plt.show()
    ```

    ![Normal PDF](assets/normal_pdf_2.svg 'Normal PDF')

  - The normal CDF

    ```python
    # Generate CDFs
    x_std1, y_std1 = ecdf(samples_std1)
    x_std3, y_std3 = ecdf(samples_std3)
    x_std10, y_std10 = ecdf(samples_std10)

    # Plot CDFs
    plt.plot(x_std1, y_std1, marker=".", linestyle="none")
    plt.plot(x_std3, y_std3, marker=".", linestyle="none")
    plt.plot(x_std10, y_std10, marker=".", linestyle="none")

    # Make a legend and show the plot
    _ = plt.legend(('std = 1', 'std = 3', 'std = 10'), loc='lower right')
    plt.show()
    ```

    ![Normal CDF](assets/normal_cdf.svg 'Normal CDF')

  - Light tails of the Normal distribution
    If we look at the Normal distribution, the probability of being more than four standard deviations from the mean is very small. This means that when you are modeling data as Normally distributed, outliers are extremely unlikely. Real data sets often have extreme values, and when this happens, the Normal distribution might not be the best description of your data.
    ![Normal disadvantage](assets/normal_disadvantage.png 'Normal Disadvantage')
  - Example

    1. Are the Belmont Stakes results Normally distributed?
       Since 1926, the Belmont Stakes is a 1.5 mile-long race of 3-year old thoroughbred horses. Secretariat ran the fastest Belmont Stakes in history in 1973. While that was the fastest year, 1970 was the slowest because of unusually wet and sloppy conditions. With these two outliers removed from the data set, compute the mean and standard deviation of the Belmont winners' times. Sample out of a Normal distribution with this mean and standard deviation using the np.random.normal() function and plot a CDF. Overlay the ECDF from the winning Belmont times. Are these close to Normally distributed?

       ```python
        # Compute mean and standard deviation: mu, sigma
        mu, sigma = belmont_no_outliers.mean(), belmont_no_outliers.std()

        # Sample out of a normal distribution with this mu and sigma: samples
        samples = np.random.normal(mu, sigma, 10000)

        # Get the CDF of the samples and of the data
        x, y = ecdf(belmont_no_outliers)
        x_theor, y_theor = ecdf(samples)

        # Plot the CDFs and show the plot
        _ = plt.plot(x_theor, y_theor)
        _ = plt.plot(x, y, marker='.', linestyle='none')
        _ = plt.xlabel('Belmont winning time (sec.)')
        _ = plt.ylabel('CDF')
        plt.show()
       ```

    1. What are the chances of a horse matching or beating Secretariat's record?
       Assume that the Belmont winners' times are Normally distributed (with the 1970 and 1973 years removed), what is the probability that the winner of a given Belmont Stakes will run it as fast or faster than Secretariat?

       ```python
       # Take a million samples out of the Normal distribution: samples
       samples = np.random.normal(mu, sigma, 1000000)

       # Compute the fraction that are faster than 144 seconds: prob
       prob = np.sum(samples <= 144) / len(samples)

       # Print the result
       print('Probability of besting Secretariat:', prob)
       ```

- Exponential distribution
  The waiting time between arrivals of a poisson process is exponentially distributed.
  Examples:

  1. Nuclear incidents:
     Timing of one is independent of all others so it's a poisson process and the time is exponentially distributed.
  1. How might we expect the time between Major League no-hitters to be distributed? Be careful here: a few exercises ago, we considered the probability distribution for the number of no-hitters in a season. Now, we are looking at the probability distribution of the time between no hitters.
     Ans: Exponential
  1. Unfortunately, Justin was not alive when Secretariat ran the Belmont in 1973. Do you think he will get to see a performance like that? To answer this, you are interested in how many years you would expect to wait until you see another performance like Secretariat's. How is the waiting time until the next performance as good or better than Secretariat's distributed? Choose the best answer.

     1. Normal, because the distribution of Belmont winning times are Normally distributed.
     1. Normal, because there is a most-expected waiting time, so there should be a single peak to the distribution.
     1. Exponential: It is very unlikely for a horse to be faster than Secretariat, so the distribution should decay away to zero for high waiting time.
     1. Exponential: A horse as fast as Secretariat is a rare event, which can be modeled as a Poisson process, and the waiting time between arrivals of a Poisson process is Exponentially distributed.

     Ans: 4

- `np.random.exponential(mean, size)`

  ```python
  # Draw samples of waiting times: waiting_times
  waiting_times = np.random.exponential(715, 100000)

  # Make the histogram
  plt.hist(waiting_times, bins=100, normed=True, histtype='step')

  # Label axes
  plt.xlabel('waiting games')
  plt.ylabel('PDF')

  # Show the plot
  plt.show()
  ```

  ![Exponential PDF](assets/exp_pdf.svg 'Exponential PDF')

  ```python
  # Draw samples of waiting times: waiting_times
  waiting_times = np.random.exponential(715, 100000)

  x, y = ecdf(waiting_times)
  plt.plot(x, y)

  # Label axes
  plt.xlabel('waiting games')
  plt.ylabel('CDF')

  # Show the plot
  plt.show()
  ```

  ![Exponential CDF](assets/exp_cdf.svg 'Exponential CDF')

> Supervised Learning with scikit-learn

###### Datasets

- `sklearn.datasets.load_digits()`, sample program related to `datasets` module is given as follows:

  ```python
  # Import necessary modules
  from sklearn import datasets
  import matplotlib.pyplot as plt

  # Load the digits dataset: digits
  digits = datasets.load_digits()

  # Print the keys and DESCR of the dataset
  print(digits.keys())
  # dict_keys(['data', 'target', 'target_names', 'images', 'DESCR'])
  print(digits['DESCR'])

  # Print the shape of the images and data keys
  print(digits.images.shape)
  print(digits.data.shape)
  # (1797, 8, 8)
  # (1797, 64)

  # Display digit 1010
  plt.imshow(digits.images[1010], cmap=plt.cm.gray_r, interpolation='nearest')
  plt.show()
  ```

###### Classification

- `sklearn.neighbors.KNeighborsClassifier()`
- `sklearn.neighbors.KNeighborsClassifier().fit(x_train, y_train)`
- `sklearn.neighbors.KNeighborsClassifier().predict(x_test)`
- `sklearn.neighbors.KNeighborsClassifier().score(x_test, y_test)`

###### Regression

- `sklearn.linear_model.LinearRegression()`

###### Regularized regression

- Linear regression minimizes a loss function. It chooses a coeffient for each feature variable and large coefficient can lead to overfitting.
  So the regularization is to penalize this large coefficients.
  ![OLS, Redge and Lasso formula](assets/ols_redge_lasso.png 'OLS, Redge and Lasso formula')
  Here,
  λ = we need to choose
  Θ = coeeficients
  k = number of coeeficients
  λ value can be picked as like as hyperparameter tuning and it controls model complexity like:
  - λ = 0: we got back OLS(can lead to overfitting)
  - very high λ: can lead to underfitting
- Lasso(least absolute shrinkage and selection operator) regression can be used to select important features of a dataset because it shrinks the coefficient to less important features to exactly 0.
  ![Lasso feature](assets/lasso_feature.png 'Lasso feature')
  Example

  ```python
  # Import Lasso
  from sklearn.linear_model import Lasso

  # Instantiate a lasso regressor: lasso
  lasso = Lasso(alpha=0.4, normalize=True)

  # Fit the regressor to the data
  lasso.fit(X, y)

  # Compute and print the coefficients
  lasso_coef = lasso.coef_
  print(lasso_coef)

  # Plot the coefficients
  plt.plot(range(len(df_columns)), lasso_coef)
  plt.xticks(range(len(df_columns)), df_columns.values, rotation=60)
  plt.margins(0.02)
  plt.show()
  ```

  ![LASSO output](assets/lasso_output.svg 'LASSO output')
  It seems like `child_mortality` is the most important feature when predicting life expectancy.

- Lasso is great for feature selection, but when building regression models, Ridge regression should be your first choice.
  Recall that lasso performs regularization by adding to the loss function a penalty term of the absolute value of each coefficient multiplied by some alpha. This is also known as **L<sub>1</sub>** regularization because the regularization term is the **L<sub>1</sub>** norm of the coefficients. This is not the only way to regularize, however.
  If instead you took the sum of the squared values of the coefficients multiplied by some alpha - like in Ridge regression - you would be computing the **L<sub>2</sub>** norm.

  ```python
  # Import necessary modules
  from sklearn.linear_model import Ridge
  from sklearn.model_selection import cross_val_score

  def display_plot(cv_scores, cv_scores_std):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alpha_space, cv_scores)

    std_error = cv_scores_std / np.sqrt(10)

    ax.fill_between(alpha_space, cv_scores + std_error, cv_scores - std_error, alpha=0.2)
    ax.set_ylabel('CV Score +/- Std Error')
    ax.set_xlabel('Alpha')
    ax.axhline(np.max(cv_scores), linestyle='--', color='.5')
    ax.set_xlim([alpha_space[0], alpha_space[-1]])
    ax.set_xscale('log')
    plt.show()

  # Setup the array of alphas and lists to store scores
  alpha_space = np.logspace(-4, 0, 50)
  ridge_scores = []
  ridge_scores_std = []

  # Create a ridge regressor: ridge
  ridge = Ridge(normalize=True)

  # Compute scores over range of alphas
  for alpha in alpha_space:

      # Specify the alpha value to use: ridge.alpha
      ridge.alpha = alpha

      # Perform 10-fold CV: ridge_cv_scores
      ridge_cv_scores = cross_val_score(ridge, X, y, cv=10)

      # Append the mean of ridge_cv_scores to ridge_scores
      ridge_scores.append(np.mean(ridge_cv_scores))

      # Append the std of ridge_cv_scores to ridge_scores_std
      ridge_scores_std.append(np.std(ridge_cv_scores))

  # Display the plot
  display_plot(ridge_scores, ridge_scores_std)
  ```

  ![Ridge out](assets/ridge_out.svg 'Ridge Out')

###### Logistic Regression

- `sklearn.linear_model.LogisticRegression()`
- `sklearn.linear_model.LogisticRegression().predict_proba(x_test)` The returned estimates for all classes are ordered by the label of classes.
  ```python
  y_pred_prob = logreg.predict_proba(X_test)
  print(y_pred_proba[:5])
  >>> array([[0.60409835, 0.39590165],
       [0.76042394, 0.23957606],
       [0.79670177, 0.20329823],
       [0.77236009, 0.22763991],
       [0.57194882, 0.42805118]])
  y_pred_prob[:5, 0] + y_pred_prob[:5, 1]
  >>> array([1., 1., 1., 1., 1.])
  ```
  Which means that this method output the binary predicted value for both classes. 0 index consists of '0' class prediction value and 1 number index consists of '1' class prediction value.

###### ElasticNet

Remember lasso and ridge regression from the previous chapter? Lasso used the **L<sub>1</sub>** penalty to regularize, while ridge used the **L<sub>2</sub>** penalty. There is another type of regularized regression known as the elastic net. In elastic net regularization, the penalty term is a linear combination of the **L<sub>1</sub>** and **L<sub>2</sub>** penalties:
<img src="https://render.githubusercontent.com/render/math?math=\LARGE a \times L_1 %2B b \times L_2">
In scikit-learn, this term is represented by the `l1_ratio` parameter: An `l1_ratio` of 1 corresponds to an **L<sub>1</sub>** penalty, and anything lower is a combination of **L<sub>1</sub>** and **L<sub>2</sub>**.
Example:

```python
# Import necessary modules
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)

# Create the hyperparameter grid
l1_space = np.linspace(0, 1, 30)
param_grid = {'l1_ratio': l1_space}

# Instantiate the ElasticNet regressor: elastic_net
elastic_net = ElasticNet()

# Setup the GridSearchCV object: gm_cv
gm_cv = GridSearchCV(elastic_net, param_grid, cv=5)
```

###### Measuring model performance

- `sklearn.model_selection.train_test_split(x, y, test_size, random_state, stratify)`
  This stratify parameter makes a split so that the proportion of values in the sample produced will be the same as the proportion of values provided to parameter stratify.
  For example, if variable y is a binary categorical variable with values 0 and 1 and there are 25% of zeros and 75% of ones, stratify=y will make sure that your random split has 25% of 0's and 75% of 1's.[[src](https://stackoverflow.com/a/38889389/8809538)]
  ```python
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state=42, stratify=y)
  ```
- Overfitting and Underfitting test code example

  ```python
  # Setup arrays to store train and test accuracies
  neighbors = np.arange(1, 9)
  train_accuracy = np.empty(len(neighbors))
  test_accuracy = np.empty(len(neighbors))

  # Loop over different values of k
  for i, k in enumerate(neighbors):
      # Setup a k-NN Classifier with k neighbors: knn
      knn = KNeighborsClassifier(n_neighbors=k)

      # Fit the classifier to the training data
      knn.fit(X_train, y_train)

      #Compute accuracy on the training set
      train_accuracy[i] = knn.score(X_train, y_train)

      #Compute accuracy on the testing set
      test_accuracy[i] = knn.score(X_test, y_test)

  # Generate plot
  plt.title('k-NN: Varying Number of Neighbors')
  plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
  plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
  plt.legend()
  plt.xlabel('Number of Neighbors')
  plt.ylabel('Accuracy')
  plt.show()
  ```

  ![Overfitting and Underfitting](assets/overfit_underfit.svg 'Overfitting and Underfitting')
  It looks like the test accuracy is highest when using 3 and 5 neighbors. Using 8 neighbors or more seems to result in a simple model that underfits the data.
  There is a good [article](https://towardsdatascience.com/learning-curve-to-identify-overfitting-underfitting-problems-133177f38df5).

- `sklearn.metrics.mean_squared_error(y_true, y_pred)`
  ![MSE Error](assets/mse_formula.png 'MSE Error')
- `model.score()` gives the R<sup>2</sup> score
  ![R^2 score](assets/R_Squared_Computation.png 'R^2 Score')
- Example code for those two:

  ```python
  # Import necessary modules
  from sklearn.linear_model import LinearRegression
  from sklearn.model_selection import train_test_split
  from sklearn.metrics import mean_squared_error

  # Create training and test sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state=42)

  # Create the regressor: reg_all
  reg_all = LinearRegression()

  # Fit the regressor to the training data
  reg_all.fit(X_train, y_train)

  # Predict on the test data: y_pred
  y_pred = reg_all.predict(X_test)

  # Compute and print R^2 and RMSE
  print("R^2: {}".format(reg_all.score(X_test, y_test)))
  # R^2: 0.838046873142936
  rmse = np.sqrt(mean_squared_error(y_test, y_pred))
  print("Root Mean Squared Error: {}".format(rmse))
  # Root Mean Squared Error: 3.2476010800377213
  ```

- Pitfall of `train_test_split`
  - Model performance is dependent on way the data is split, if the data for train set was being selected as a way that those are not representative of the rest, then model is not performed well and this can not be oevercome by `train_test_split` because it's a random selection process.
  - Not representative of the model's ability to generalize
    Solution: k-fold cross validation but more folds add more computational power.
- `sklearn.model_selection.cross_val_score(estimator, X, y, cv)`

  ```python
  # Import the necessary modules
  from sklearn.linear_model import LinearRegression
  from sklearn.model_selection import cross_val_score

  # Create a linear regression object: reg
  reg = LinearRegression()

  # Compute 5-fold cross-validation scores: cv_scores
  cv_scores = cross_val_score(reg, X, y, cv=5)

  # Print the 5-fold cross-validation scores
  print(cv_scores)
  # [0.81720569 0.82917058 0.90214134 0.80633989 0.94495637]
  print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))
  # Average 5-Fold CV Score: 0.8599627722793232
  ```

###### `sklearn` precaustions

- If we want to use one feature in `sklearn` from `pandas` then we have to reshape the data same as follows:

  ```python
  # Import numpy and pandas
  import numpy as np
  import pandas as pd

  # Read the CSV file into a DataFrame: df
  df = pd.read_csv('gapminder.csv')

  # Create arrays for features and target variable
  y = df.life
  X = df.fertility

  # Print the dimensions of y and X before reshaping
  print("Dimensions of y before reshaping: ", y.shape)
  print("Dimensions of X before reshaping: ", X.shape)

  # Dimensions of y before reshaping:  (139,)
  # Dimensions of X before reshaping:  (139,)

  # Reshape X and y
  y_reshaped = y.reshape(-1, 1)
  X_reshaped = X.reshape(-1, 1)

  # Print the dimensions of y_reshaped and X_reshaped
  print("Dimensions of y after reshaping: ", y_reshaped.shape)
  print("Dimensions of X after reshaping: ", X_reshaped.shape)

  # Dimensions of y after reshaping:  (139, 1)
  # Dimensions of X after reshaping:  (139, 1)
  """
  before:
  0    2.73
  1    6.43
  2    2.24
  3    1.40
  4    1.96
  Name: fertility, dtype: float

  after:
  array([[2.73],
        [6.43],
        [2.24],
        [1.4 ],
        [1.96]])
  """
  ```

- Sometimes prediction space for one feature can be genreted as follows:
  ```python
  prediction_space = np.linspace(min(X_fertility), max(X_fertility)).reshape(-1,1)
  print(prediction_space.shape)
  # (50, 1)
  ```

###### Fine-tuning your model

- Measuring model performance with accuracy:

  - Fraction of correctly classified samples
  - Not always a useful metric

  Consider a spam classification problem in which 99% of emails are real and only 1% are spam. I could build a model that classifies all emails as real; this model would be correct 99% of the time and thus have an accuracy of 99%, which sounds great. However, this naive classifier does a horrible job of predicting spam: it never predicts spam at all, so it completely fails at its original purpose. The situation when one class is more frequent is called class imbalance because the class of real emails contains way more instances than the class of spam. This is a very common situation in practice and requires a more nuanced metric to assess the performance of our model.
  [How to solve the zero-frequency problem in Naive Bayes?](https://medium.com/atoti/how-to-solve-the-zero-frequency-problem-in-naive-bayes-cd001cabe211)
  For these problems some other metric can be helpful.

- Confusion matrix
  |Classes|Predicted: Spam Email|Predicted: Spam Email|
  ---- | ---- | ---- |
  |Actual: Spam Email|True Positive|False Negative|
  |Actual: Real Email|False Positive|True Negative|

  - True Positive means number of spam emails correctly labeled
  - True Negative means number of real emails correcly labeled
  - False Negative means number of spam emails incorrectly labeled
  - False Positive means number of real emails incorrectly labeled

  Usually the _class of interest_ is called the positive class. Here we are trying to detect spam emails so it makes spam emails as positive class.
  <img src="https://render.githubusercontent.com/render/math?math=\LARGE accuracy=\frac{t_p %2B t_n}{t_p %2B t_n %2B f_p %2B f_n}">
  <img src="https://render.githubusercontent.com/render/math?math=\LARGE precision=\frac{t_p}{t_p %2B f_p}">
  It is also called _Positive Predictive Value(PPV)_. In this case it can be written as follows:
  <img src="https://render.githubusercontent.com/render/math?math=\LARGE precision=\frac{total \space\space number \space\space of \space\space correctly \space\space labeled \space\space spam}{total \space\space number \space\space of \space\space emails \space\space classified \space\space as \space\space spam}">

    <img src="https://render.githubusercontent.com/render/math?math=\LARGE specificity= \frac{t_n}{t_n %2B f_p}">

    <img src="https://render.githubusercontent.com/render/math?math=\LARGE recall \space\space or \space\space sensitivity=\frac{t_p }{t_p %2B f_n}">

  This is also called _sensitivity_, _hit rate_ or _true positive rate_.
  _Note that in binary classification, recall of the positive class is also known as “sensitivity”; recall of the negative class is “specificity”._[[src](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)]
  <img src="https://render.githubusercontent.com/render/math?math=\LARGE f_1=2 \times \frac{precision \times recall}{precision %2B recall}">
  It is also called the _harmonic mean of precision and recall_.

  - High precision(low false positive rate): Not many real emails were predicted as being spam.
  - High recall: Predicted most positive or spam emails correctly.
  - `sklearn.metrics.confusion_matrix(y_true, y_pred)`
  - `sklearn.metrics.classification_report(y_true, y_pred)`
  - Example:

    ```python
    # Import necessary modules
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, classification_report

    # Create training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)

    # Instantiate a k-NN classifier: knn
    knn = KNeighborsClassifier(n_neighbors=6)

    # Fit the classifier to the training data
    knn.fit(X_train, y_train)

    # Predict the labels of the test data: y_pred
    y_pred = knn.predict(X_test)

    # Generate the confusion matrix and classification report
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    """
    [[176  30]
    [ 52  50]]
                precision    recall  f1-score   support

              0       0.77      0.85      0.81       206
              1       0.62      0.49      0.55       102

    avg / total       0.72      0.73      0.72       308
    """
    ```

    Here avg/total means(sample for one example):
    avg is the weighted mean of those three categories.
    <img src="https://render.githubusercontent.com/render/math?math=\LARGE avg =\frac{0.77 \times 206 %2B 0.62 \times 102 }{206 %2B 102}=0.72">
    <img src="https://render.githubusercontent.com/render/math?math=\LARGE total=206 %2B 102 = 308">

- Receiver Operating Characteristic(ROC) curve
  ![ROC curve](assets/roc_curve.png 'ROC Curve')

  - `false_pos_rate, true_pos_rate, thresholds = sklearn.metrics.roc_curve(y_true, y_score)`

    ```python
    # Import necessary modules
    from sklearn.metrics import roc_curve

    # Compute predicted probabilities: y_pred_prob
    y_pred_prob = logreg.predict_proba(X_test)[:,1]

    # Generate ROC curve values: fpr, tpr, thresholds
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

    # Plot ROC curve
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()
    ```

    ![ROC curve example](assets/roc_curve_example.svg 'ROC curve example')
    _Note: the y-axis (True positive rate) is also known as recall._

- Area under the ROC curve(AUC)
  Larger area under the ROC curve = better model
  Say you have a binary classifier that in fact is just randomly making guesses. It would be correct approximately 50% of the time, and the resulting ROC curve would be a diagonal line in which the True Positive Rate and False Positive Rate are always equal. The Area under this ROC curve would be 0.5.
  It can be computed in direct classifier or with cross validation as follows:

  ```python
  # Import necessary modules
  from sklearn.model_selection import cross_val_score
  from sklearn.metrics import roc_auc_score

  # Compute predicted probabilities: y_pred_prob
  y_pred_prob = logreg.predict_proba(X_test)[:,1]

  # Compute and print AUC score
  print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))

  # Compute cross-validated AUC scores: cv_auc
  cv_auc = cross_val_score(LogisticRegression(), X, y, cv=5, scoring='roc_auc')

  # Print list of AUC scores
  print("AUC scores computed using 5-fold cross-validation: {}".format(cv_auc))

  ```

- Precision-Recall curve
  _Note that here, the class is positive (1) if the individual has diabetes._
  ![Precision-Recall curve](assets/precision_recall_curve.svg 'Precision-Recall curve')
  - A recall of 1 corresponds to a classifier with a low threshold in which all females who contract diabetes were correctly classified as such, at the expense of many misclassifications of those who did not have diabetes.
  - Precision is undefined for a classifier which makes no positive predictions, that is, classifies everyone as not having diabetes.
  - When the threshold is very close to 1, precision is also 1, because the classifier is absolutely certain about its predictions.
  - Precision and recall do not take true negatives into consideration. They do not appear at all in the definitions of precision and recall.
- Hyperparameter tuning
  These are the parameters those can be expicitely learned by fitting the model.
  Algorithm:

  - Try a bunch of different hyperparameter values
  - Fit all of them seprately
  - See how well each performs
  - Choose the best performing one
  - It is essential to use cross-validation
  - `sklearn.model_selection.GridSearchCV(estimator, param_grid, cv)`
    ![GridSearchCV](assets/grid_search_cv.png 'GridSearchCV')
    - `.best_params_()`
    - `.best_score_()`
  - `sklearn.model_selection.RandomizedSearchCV(estimator, param_grid, cv)`
    GridSearchCV can be computationally expensive, especially if you are searching over a large hyperparameter space and dealing with multiple hyperparameters. A solution to this is to use RandomizedSearchCV, in which not all hyperparameter values are tried out. Instead, a fixed number of hyperparameter settings is sampled from specified probability distributions.

    ```python # Import necessary modules
    from scipy.stats import randint
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.tree import DecisionTreeClassifier

    # Setup the parameters and distributions to sample from: param_dist
    param_dist = {"max_depth": [3, None],
                  "max_features": randint(1, 9),
                  "min_samples_leaf": randint(1, 9),
                  "criterion": ["gini", "entropy"]}
    # Instantiate a Decision Tree classifier: tree
    tree = DecisionTreeClassifier()

    # Instantiate the RandomizedSearchCV object: tree_cv
    tree_cv = RandomizedSearchCV(tree, param_dist, cv=5)
    # you can use train_test_split as well
    tree_cv.fit(X, y)
    # Print the tuned parameters and score
    print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
    print("Best score is {}".format(tree_cv.best_score_))
    # Tuned Decision Tree Parameters: {'criterion': 'gini', 'max_depth': 3, 'max_features': 5, 'min_samples_leaf': 2}
    # Best score is 0.7395833333333334
    ```

    _Note that RandomizedSearchCV will never outperform GridSearchCV. Instead, it is valuable because it saves on computation time._

- Some important thoughts to maintain

  - How well can the model performon never before seen data?
  - Using all data for cross-validation is not ideal
  - Split data into training and hold-out set at the begining
  - Perform grid search cross-validation on training set
  - Choose best hyperparameters and evaluate on hold-out set

- Handling missing data and Pipeline
  When many values in your dataset are missing, if you drop them, you may end up throwing away valuable information along with the missing data. It's better instead to develop an imputation strategy. This is where domain knowledge is useful, but in the absence of it, you can impute missing values with the mean or the median of the row or column that the missing value is in.

  ```python
  # Import necessary modules
  from sklearn.preprocessing import Imputer
  from sklearn.pipeline import Pipeline
  from sklearn.svm import SVC

  # Setup the pipeline steps: steps
  steps = [('imputation', Imputer(missing_values='NaN', strategy='most_frequent', axis=0)),
          ('SVM', SVC())]

  # Create the pipeline: pipeline
  pipeline = Pipeline(steps)

  # Create training and test sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)

  # Fit the pipeline to the train set
  pipeline.fit(X_train, y_train)

  # Predict the labels of the test set
  y_pred = pipeline.predict(X_test)

  # Compute metrics
  print(classification_report(y_test, y_pred))
  """
              precision    recall  f1-score   support

    democrat       0.96      0.99      0.98        83
  republican       0.98      0.94      0.96        48

  avg / total       0.97      0.97      0.97       131
  """
  ```

- Centering and scaling
  Ways to normalize data

  - Standardization: Substract the mean and divided by variance
  - All features are centered around zero and have variance one
  - Can also substract the minimum and divide by the range
  - Minimum zero and maximum one
  - Can also normalize so the data ranges from -1 to 1
  - `sklearn.preprocessing.scale(x)`
    Example with standard scaler:

    ```python
    # Import the necessary modules
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    # Setup the pipeline steps: steps
    steps = [('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier())]

    # Create the pipeline: pipeline
    pipeline = Pipeline(steps)

    # Create train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)

    # Fit the pipeline to the training set: knn_scaled
    knn_scaled = pipeline.fit(X_train, y_train)

    # Instantiate and fit a k-NN classifier to the unscaled data
    knn_unscaled = KNeighborsClassifier().fit(X_train, y_train)

    # Compute and print metrics
    print('Accuracy with Scaling: {}'.format(knn_scaled.score(X_test, y_test)))
    print('Accuracy without Scaling: {}'.format(knn_unscaled.score(X_test, y_test)))
    # Accuracy with Scaling: 0.7700680272108843
    # Accuracy without Scaling: 0.6979591836734694
    ```

  - Example with pipeline:

    ```python
    # Setup the pipeline steps: steps
    steps = [('imputation', Imputer(missing_values='NaN', strategy='mean', axis=0)),
            ('scaler', StandardScaler()),
            ('elasticnet', ElasticNet())]

    # Create the pipeline: pipeline
    pipeline = Pipeline(steps)

    # Specify the hyperparameter space
    parameters = {'elasticnet__l1_ratio':np.linspace(0,1,30)}

    # Create train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)

    # Create the GridSearchCV object: gm_cv
    gm_cv = GridSearchCV(pipeline, parameters)

    # Fit to the training set
    gm_cv.fit(X_train, y_train)

    # Compute and print the metrics
    r2 = gm_cv.score(X_test, y_test)
    print("Tuned ElasticNet Alpha: {}".format(gm_cv.best_params_))
    print("Tuned ElasticNet R squared: {}".format(r2))
    # Tuned ElasticNet Alpha: {'elasticnet__l1_ratio': 1.0}
    # Tuned ElasticNet R squared: 0.8862016570888217
    ```

> References

- [Images](https://google.com/)
- [DataCamp](https://learn.datacamp.com/)
- [stackoverflow](https://stackoverflow.com/)
- [towardsdatascience](https://towardsdatascience.com/)
- [medium](https://medium.com/)
- [scikit-learn](https://scikit-learn.org/stable/)
