# recognize-digits
Neural Network (create from scratch) that recognize digits, the model has been trained on the mnist datasets, with images of 28x28 dimension

## Network
The network has 6 layers: [784, 128, 128, 128, 128, 10], use Adam optimization with hyperparameters write in the hyperparameters.py file,
has reached an accuracy of about 97%, i've tried a lot of different hyperparameters and this is the best accuracy that i've been able
to reached

## Files
- dataset.py: load mnist dataset with keras and make some modifcations
- hyperparameters.py: there are all the hyperparameters in a dictionary
- network.py: code of the neural network
- parameters.json: when the network will finished the training the parameters will be saved in a json file if the accuracy reached a certain threshold
