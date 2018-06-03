import math
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
df = pd.read_csv("./data/california_housing_train.csv", sep=",")
df["target"] = df["median_house_value"] / 1000
df.describe()

# 1. Define feature and Define target
# - Define features and configure feature columns
# - Define the target
# - Define the input function

# 2. Define the Model
# - Configure the liner regressor
# - Train the model

# 3. Evalute the Model
# - Predict
# - Measure score.

# total # of train examples = batch size * step
# periods controls granularity of reporting.
# # of train examples in each period = batch size * step


def train_model(learning_rate, steps, batch_size, input_feature="total_rooms"):  
    periods = 10
    steps_per_period = steps / periods
    X_train = df[[input_feature]]
    target = "target"
    y_train = df[target]
    feature_columns = [tf.feature_column.numeric_column(input_feature)]

    def input_fn(X_data, y_data, batch_size=1, repeat=1, shuffle=True):
        X_tensor = {key: np.array(value) for key, value in dict(X_data).items()}
        ds = Dataset.from_tensor_slices((X_tensor, y_data))
        ds = ds.batch(int(batch_size)).repeat(repeat)
        X, y = ds.make_one_shot_iterator().get_next()
        return X, y

    train_input_fn = lambda: input_fn(X_train, y_train, batch_size=batch_size, repeat=1)
    predict_input_fn = lambda: input_fn(X_train, y_train, batch_size=1, repeat=1)

    gdo = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    gdo = tf.contrib.estimator.clip_gradients_by_norm(gdo, 5.0)
    linear_regressor = tf.estimator.LinearRegressor(
        feature_columns=feature_columns, optimizer=gdo)

    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.title("Learned Line by Period")
    plt.ylabel(target)
    plt.xlabel(input_feature)
    sample = df.sample(n=300)
    plt.scatter(sample[input_feature], sample[target])
    colors = [cm.coolwarm(x) for x in np.linspace(-1, 1, periods)]

    root_mean_squared_errors = []
    for period in range(0, periods):
        linear_regressor.train(input_fn=train_input_fn, steps=steps_per_period)
        predictions = linear_regressor.predict(input_fn=predict_input_fn)
        predictions = np.array([item['predictions'][0] for item in predictions])
        mean_squared_error = metrics.mean_squared_error(predictions, y_train)
        # print('mean_squared_error', mean_squared_error)
        root_mean_squared_error = math.sqrt(mean_squared_error)
        # print('root_mean_squared_error', root_mean_squared_error)
        root_mean_squared_errors.append(root_mean_squared_error)
        print("period %02d: %0.2f" % (period, root_mean_squared_error))
        weight = linear_regressor.get_variable_value('linear/linear_model/%s/weights' % input_feature)[0][0]
        bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')[0]
        x_min = sample[input_feature].min()
        x_max = sample[input_feature].max()
        y_min = x_min * weight + bias
        y_max = x_max * weight + bias
        plt.plot([x_min, x_max], [y_min, y_max], c=colors[period])

    plt.subplot(1, 2, 2)
    plt.ylabel('RMSE')
    plt.xlabel('Periods')
    plt.tight_layout()
    np.arange(len(root_mean_squared_errors))

    predictions

    result_df = pd.DataFrame({"prediction": pd.Series(predictions), "target": df["target"]})
    print(result_df.describe())
    print("RMSE: %0.2f" % root_mean_squared_error)

    plt.plot(np.arange(len(root_mean_squared_errors)), root_mean_squared_errors)
    plt.show()

train_model(learning_rate=1e-4, steps=1e+6, batch_size=1e+3, input_feature="population")
