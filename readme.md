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

> References

- [Images](https://google.com/)
- [DataCamp](https://learn.datacamp.com/)
