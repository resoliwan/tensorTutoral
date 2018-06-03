import math
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib import cm
from datetime import datetime

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10

df = pd.read_csv('./data/bike_train.csv')

df.columns


def get_days_of_year(date):
    date = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
    new_year_day = datetime(year=date.year, month=1, day=1)
    return (date - new_year_day).days + 1


df['day_of_year'] = df['datetime'].apply(get_days_of_year)
df['hour'] = df['datetime'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").hour)
df['workday'] = df['datetime'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").weekday())

df.describe()
# 1. Define features and Define target
# - define featureas and configure feature columnsA
# - define target
# - define input function


def train_model(learning_rate, steps, batch_size, periods, x_feature, input_features=['hour']):
    steps_per_period = int(steps / periods)
    print('learning_rate', learning_rate)
    print('steps', steps)
    print('steps_per_period', steps_per_period)
    feature_columns = [tf.feature_column.numeric_column(column) for column in input_features]
    X_train = df[input_features]
    target_feature = 'count'
    y_train = df[target_feature]

    def input_fn(X_data, y_data, batch_size=1, repeat=None, shuffle=False):
        X_tensors = {key: np.array(value) for key, value in dict(X_data).items()}
        ds = tf.data.Dataset.from_tensor_slices((X_tensors, y_data))
        ds = ds.batch(int(batch_size)).repeat(repeat)
        if shuffle:
            ds.shuffle(buffer_size=10000)
        X, y = ds.make_one_shot_iterator().get_next()
        return X, y
    # input_fn(X_train, y_train)

    # 2. Define Model
    # - Configure optimizer
    # - Confiture algorithmn
    # - Train the model

    train_input_fn = lambda: input_fn(X_train, y_train, batch_size=batch_size, repeat=100, shuffle=True)
    predict_input_fn = lambda: input_fn(X_train, y_train, batch_size=batch_size, repeat=1, shuffle=False)

    gdo = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    gdo = tf.contrib.estimator.clip_gradients_by_norm(gdo, 5.0)

    linear_regressor = tf.estimator.LinearRegressor(feature_columns=feature_columns, optimizer=gdo)

    input_feature_num = len(input_features)
    row = int(math.sqrt(input_feature_num))
    column = int(math.sqrt(input_feature_num))
    plt.figure(figsize=(15, 15))

    plt.subplot(1, 2, 1)
    plt.ylabel(target_feature)
    plt.xlabel(x_feature)
    sample = df.sample(n=300)
    plt.scatter(sample[x_feature], sample[target_feature])
    colors = [cm.coolwarm(x) for x in np.linspace(-1, 1, periods)]

    root_mean_squared_errors = []
    for period in range(periods):
        linear_regressor.train(input_fn=train_input_fn, steps=steps_per_period)
        predictions = linear_regressor.predict(input_fn=predict_input_fn)
        predictions = [item['predictions'][0] for item in predictions]
        mean_squared_error = metrics.mean_squared_error(predictions, y_train)
        root_mean_squared_error = math.sqrt(mean_squared_error)
        root_mean_squared_errors.append(root_mean_squared_error)
        print('period: %02d: %0.2f' % (period, root_mean_squared_error))
        weight = linear_regressor.get_variable_value('linear/linear_model/%s/weights' % x_feature)[0][0]
        bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')[0]
        x_min = sample[x_feature].min()
        x_max = sample[x_feature].max()
        y_min = x_min * weight + bias
        y_max = x_max * weight + bias
        plt.plot([x_min, x_max], [y_min, y_max], c=colors[period])

    # 3. Evalute the Model
    # - Predict
    # - Measure score
    print("RMSE: on traning %0.2f" % root_mean_squared_error)
    result_df = pd.DataFrame({
        'perdiction': pd.Series(predictions),
        'target': df[target_feature]})
    print(result_df.describe())
    print("RMSE: on traning %0.2f" % root_mean_squared_error) 
    plt.subplot(1, 2, 2)
    plt.ylabel('RMSE')
    plt.xlabel('Periods')

    plt.plot(np.arange(len(root_mean_squared_errors)), root_mean_squared_errors)
    plt.show()


if __name__ == '__main__':
    def is_interactive():
        import __main__ as main
        return not hasattr(main, '__file__')

    if not is_interactive():
        train_model(
                learning_rate=1e-4,
                steps=100,
                batch_size=100,
                periods=3,
                x_feature='hour',
                input_features=['hour', 'atemp'])
