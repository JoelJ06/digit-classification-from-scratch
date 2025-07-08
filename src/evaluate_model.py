import pickle, pandas as pd, matplotlib.pyplot as plt
from neural_network import NeuralNetwork # when using pickle, we need to import class
from image_viewer import ImageNavigator, reshape_images

nn = pickle.load(open('../models/network.p', 'rb'))
test_data = pd.read_csv("../assets/test.csv")

X_test = test_data.values / 255.0 # we have to preprocess data same as how we trained model
test_predictions = nn.predict(X_test)
reshaped_images = reshape_images(X_test) # for display

navigator = ImageNavigator(reshaped_images, test_predictions)
navigator.show()
