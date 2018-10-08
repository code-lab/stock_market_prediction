from data_collection import collect_data
from neural_network import (
    importing_the_training_set_testing_set,
    feature_scaling,
    reshaping,
    building_the_rnn,
    making_predictions_and_visualizing_results
)

# collect data for training
collect_data(type="training")

# collect data for testing
collect_data(type="testing")

training_set, output_set = importing_the_training_set_testing_set()

# counting rows
rows, columns = training_set.shape

training_set_scaled, output_set_scaled = feature_scaling(
    training_set, output_set)

# Getting the inputs and the outputs
X_train = training_set_scaled[0:rows-1, :]
Y_train = output_set_scaled[1:rows, :]

X_train = reshaping(X_train, rows)


# Part 2: Building the RNN
regressor = building_the_rnn(X_train, Y_train)

# Part 3:
making_predictions_and_visualizing_results(regressor)
