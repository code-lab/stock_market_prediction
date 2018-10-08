import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))


def importing_the_training_set_testing_set():
    dataset_train = pd.read_csv('data/training.csv')
    training_set = dataset_train.iloc[:, 1:5].values
    output_set = dataset_train.iloc[:, 3:4].values
    return [training_set, output_set]


def feature_scaling(training_set, output_set):
    training_set_scaled = sc.fit_transform(training_set)
    output_set_scaled = sc.fit_transform(output_set)
    return [training_set_scaled, output_set_scaled]


def reshaping(X_train, rows):
    return np.reshape(X_train, (rows-1, 1, 4))


def building_the_rnn(X_train, Y_train):
    # Importing the Keras libraries and packages
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import GRU

    # Initialising the RNN
    regressor = Sequential()

    # Adding the input layer and the GRU layer
    regressor.add(GRU(units=4, activation='sigmoid', input_shape=(1, 4)))

    # Adding the output layer
    regressor.add(Dense(units=1))

    # Compailing the RNN
    regressor.compile(optimizer='adam', loss='mean_squared_error')

    # Fitting the RNN to the Training set
    regressor.fit(X_train, Y_train, batch_size=32, epochs=200)

    return regressor


def getting_the_real_stock_price():
    dataset_test = pd.read_csv('data/testing.csv')
    rows_t, columns_t = dataset_test.shape
    real_stock_price = dataset_test.iloc[0:rows_t-1, 1:5].values
    real_stock_price_output = dataset_test.iloc[1:rows_t, 3:4].values
    real_stock_price_output_df = pd.DataFrame(real_stock_price_output)
    real_stock_price_output_df.to_csv('data/real_stock_price_output.csv')
    return (
        real_stock_price,
        rows_t,
        real_stock_price_output,
        real_stock_price_output_df
    )


def getting_the_predicted_stock_price(real_stock_price, rows_t, regressor):
    inputs = real_stock_price
    inputs = sc.transform(inputs)
    inputs = np.reshape(inputs, (rows_t-1, 1, 4))
    predicted_stock_price = regressor.predict(inputs)
    predicted_stock_price_output = sc.inverse_transform(predicted_stock_price)
    predicted_stock_price_df = pd.DataFrame(predicted_stock_price_output)
    predicted_stock_price_df.to_csv('data/predicted_stock_price.csv')
    return predicted_stock_price_output, predicted_stock_price


def visualizing_the_results(
    real_stock_price_output,
    predicted_stock_price_output
):
    plt.plot(real_stock_price_output, color='red', label='Real Stock Price')
    plt.plot(
        predicted_stock_price_output,
        color='blue',
        label='Predicted Stock Price')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()
    plt.savefig('data/stock_market_prediction.png')


def making_predictions_and_visualizing_results(regressor):
    real_stock_price, rows_t, real_stock_price_output,\
        real_stock_price_output_df = getting_the_real_stock_price()
    predicted_stock_price_output, predicted_stock_price = \
        getting_the_predicted_stock_price(
            real_stock_price,
            rows_t,
            regressor
        )
    visualizing_the_results(
        real_stock_price_output,
        predicted_stock_price_output
    )

    evaluating_the_rnn(
        real_stock_price_output_df,
        real_stock_price_output,
        predicted_stock_price
    )


def evaluating_the_rnn(
    real_stock_price_output_df,
    real_stock_price_output,
    predicted_stock_price
):
    import math
    from sklearn.metrics import mean_squared_error
    mean_real_stock_price_output_df = real_stock_price_output_df.mean()
    rmse = math.sqrt(
        mean_squared_error(real_stock_price_output, predicted_stock_price)
    )
    rmse_percentage = rmse/mean_real_stock_price_output_df
    print(rmse_percentage)
