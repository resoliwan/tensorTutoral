import math
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

pd.options.display.max_rows =  10
california_housing_dataframe = pd.read_csv("https://storage.googleapis.com/mledu-datasets/california_housing_train.csv", sep=",")

california_housing_dataframe.describe()
california_housing_dataframe.columns

california_housing_dataframe[0:10]

california_housing_dataframe = california_housing_dataframe.reindex(np.random.permutation(california_housing_dataframe.index))
california_housing_dataframe["median_house_value"] /= 1000.0
california_housing_dataframe

california_housing_dataframe[0:10]

# 1. Define features and configure feature columns
my_feature = california_housing_dataframe[['total_rooms']]

feature_columns = [tf.feature_column.numeric_column('total_rooms')]

# 2. Define the Target.
targets = california_housing_dataframe["median_house_value"]

# 3. Configure linearRegression

my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-7)
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

linear_regressor = tf.estimator.LinearRegressor(
    feature_columns=feature_columns,
    optimizer=my_optimizer
    )

# 4. Define the input function
# - preprocess
# - batch, shuffle
# - repeat
def my_input_fn(features, targets, batch_size=1, shuffle=True, num_eponchs=None):
  # features = {key: np.array(value)for key,value in dict(features).items()}
  # Convert pandas data into a dict of np arrays.
  features = {key:np.array(value) for key,value in dict(features).items()}                                           
    # Construct a dataset, and configure batching/repeating.
  ds = Dataset.from_tensor_slices((features, targets))
  ds = ds.batch(batch_size).repeat(num_eponchs)
  if shuffle:
    ds = ds.shuffle(buffer_size=10000)
  features, labels = ds.make_one_shot_iterator().get_next()
  return features, labels

my_feature.shape, targets.shape
my_input_fn(my_feature, targets)


_ = linear_regressor.train(
    input_fn = lambda:my_input_fn(my_feature, targets),
    steps=100
)

prediction_input_fn = lambda: my_input_fn(my_feature, targerts, num_eponchs=1, shuffle=False)

predictions = linear_regressor.predict(input_fn=prediction_input_fn)

# Format predictions as a NumPy array, so we can calculate error metrics.
predictions = np.array([item['predictions'][0] for item in predictions])

# Print Mean Squared Error and Root Mean Squared Error.
mean_squared_error = metrics.mean_squared_error(predictions, targets)

root_mean_squared_error = math.sqrt(mean_squared_error)

print "Mean Squared Error (on training data): %0.3f" % mean_squared_error
print "Root Mean Squared Error (on training data): %0.3f" % root_mean_squared_error



