import pandas as pd
pd.__version__

# create a series.
pd.Series(['San Francisco', 'San Jose', 'Sacramento'])

#create Data table

city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
population = pd.Series([100, 200, 300])

pd.DataFrame({'City name': city_names, 'Population': population})

california_housing_dataframe = pd.read_csv(
    "https://storage.googleapis.com/mledu-datasets/california_housing_train.csv", sep=",")

california_housing_dataframe.describe()

california_housing_dataframe.head()

california_housing_dataframe.hist('housing_median_age')

from matplotlib import pyplot as plt

plt.show()

cites = pd.DataFrame({'City name': city_names, 'Population': population})

print(type(cites['City name']))
cites['City name']

print(type(cites['City name'][1]))
cites['City name'][1]

population / 10

import numpy as np

np.log(population)

population.apply(lambda val: val > 10)

cites['new col'] = pd.Series([1,2,3])

cites['new col2'] = cites['Population'] / cites['new col']


cites['City name'].apply(lambda name: name.startswith('San')) & (cites['Population'] > 10)

city_names.index.values

cites.index.values

cites2 = cites.reindex([2, 0, 1])

cites2.reindex(np.random.permutation(cites2.index))

cites2.reindex([3, 4, 5])
