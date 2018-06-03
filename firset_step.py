import math
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt

tf.logging.set_verbosity(tf.logging.INFO)
pd.options.display.max_rows = 10
california_housing_dataframe = pd.read_csv(
    "https://storage.googleapis.com/mledu-datasets/california_housing_train.csv",
    sep=",")

california_housing_dataframe.describe()

california_housing_dataframe.columns
california_housing_dataframe[0:10]
california_housing_dataframe = california_housing_dataframe.reindex(
    np.random.permutation(california_housing_dataframe.index))
california_housing_dataframe["median_house_value"] /= 1000.0
california_housing_dataframe

# 1. Define features and configure feature columns
my_feature = california_housing_dataframe[['total_rooms']]
feature_columns = [tf.feature_column.numeric_column('total_rooms')]

# 2. Define the Target.
targets = california_housing_dataframe["median_house_value"]

# 3. Configure linearRegression

my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-7)
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

linear_regressor = tf.estimator.LinearRegressor(
    feature_columns=feature_columns, optimizer=my_optimizer)


# 4. Define the input function
# - preprocess
# - batch, shuffle
# - repeat
def my_input_fn(features,
                targets,
                batch_size=1,
                shuffle=True,
                num_eponchs=None):
    # features = {key: np.array(value)for key,value in dict(features).items()}
    # Convert pandas data into a dict of np arrays.
    features = {key: np.array(value) for key, value in dict(features).items()}
    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features, targets))
    ds = ds.batch(batch_size).repeat(num_eponchs)
    if shuffle:
        ds = ds.shuffle(buffer_size=10000)
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


my_feature.shape, targets.shape
my_input_fn(my_feature, targets)

sess = tf.Session()
next_element = my_input_fn(my_feature, targets)
for i in range(10):
    value = sess.run(next_element)
    print('value', value)

_ = linear_regressor.train(
    input_fn=lambda: my_input_fn(my_feature, targets), steps=1000)

#
# prediction_input_fn = lambda: my_input_fn(my_feature, targets, num_eponchs=1, shuffle=False)
# predictions = linear_regressor.predict(input_fn=prediction_input_fn)
# # Format predictions as a NumPy array, so we can calculate error metrics.
# predictions = np.array([item['predictions'][0] for item in predictions])
# # Print Mean Squared Error and Root Mean Squared Error.
# mean_squared_error = metrics.mean_squared_error(predictions, targets)
# root_mean_squared_error = math.sqrt(mean_squared_error)
# print( "Mean Squared Error (on training data): %0.3f" % mean_squared_error)
# print ("Root Mean Squared Error (on training data): %0.3f" % root_mean_squared_error)
#
# min_house_value = california_housing_dataframe['median_house_value'].min()
# print('min_house_value', min_house_value)
# max_house_value = california_housing_dataframe['median_house_value'].max()
# print('max_house_value', max_house_value)
# min_max_difference = max_house_value - min_house_value
# print('min_max_difference', min_max_difference)
#
# california_data = pd.DataFrame()
# california_data['predictions'] = pd.Series(predictions)
# california_data['targets'] = pd.Series(targets)
# california_data.describe()
#
# sample = california_housing_dataframe.sample(n=300)
#
#
# x_0 = sample['total_rooms'].min()
# x_1 = sample['total_rooms'].max()
#
# weight = linear_regressor.get_variable_value('linear/linear_model/total_rooms/weights')[0]
# bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')
#
# y_0 = weight * x_0 + bias
# y_1 = weight * x_1 + bias
#
# plt.plot([x_0, x_1], [y_0, y_1], c='r')
# plt.ylabel('median_house_value')
# plt.xlabel('total_rooms')
#
# plt.scatter(sample['total_rooms'], sample['median_house_value'])
#
# plt.show()


def train_model(learning_rate, steps, batch_size, input_feature="total_rooms"):
    periods = 10
    steps_per_peried = steps / periods
    my_feature = input_feature
    my_feature_data = california_housing_dataframe[[my_feature]]
    my_label = "median_house_value"
    #
    targets = california_housing_dataframe[my_label]
    feature_columns = [tf.feature_column.numeric_column(my_feature)]
    traing_inut_fn = lambda: my_input_fn(my_feature_data, targets, bath_size=bath_size)
    prediction_inut_fn = lambda: my_input_fn(my_feature_data, targets, num_epocsh=1, bath_size=bath_size)
    my_optimizer = tf.train.GradientDescentOptimizer(
        learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(
        my_optimizer, 5.0)
    linear_regressor = tf.estimator.LinearRegressor(
        feature_columns=feature_columns, optimizer=my_optimizer)
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.title('Learned Line by Period')
    plt.xlabel(my_feature)
    plt.ylabel(my_label)
    sample = california_housing_dataframe.sample(n=300)
    plt.scatter(sample[my_feature], sample[my_label])
    colors = [cm.coolwarm(x) for x in np.linspace(-1, 1, periods)]
    root_mean_squared_errors = []
    for period in range(0, preriods):
        linear_regressor.train(input_fn=traing_inut_fn, steps=steps_per_peried)
        predictions = linear_regressor.predict(input_fn=prediction_input_fn)
        predictions = np.array(
            [items['predictions'][0] for item in predictions])
        root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(predictions, targets))
        print('periods %02d : %0.2f' % (period, root_mean_squared_error))
        root_mean_squared_errors.append(root_mean_squared_error)
        y_extents = np.array([0, sample[my_label].max()])

        weight = linear_regressor.get_variable_value(
            'linear/linear_model/%s/weights' % input_feature)[0]
        bias = linear_regressor.get_variable_value(
            'linear/linear_model/bias_weights')

        x_extents = (y_extents - bias) / weight
        x_extents = np.maximu(np.minimum)

    plt.subplot(1, 2, 2)
    plt.title('Root Mean Squared')
    plt.xlabel('RMSE')
    plt.ylabel('Preiods')


train_model(learning_rate=1e-4, steps=100, batch_size=1)
