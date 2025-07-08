import pickle
import numpy as np
import pandas as pd
from image_viewer import ImageNavigator, reshape_images

# Step 1: Load and preprocess the data from CSV files
def load_data(train_path, test_path):
    # Load train and test data
    train_data = pd.read_csv(train_path) # we divide this into train and validation sets
    test_data = pd.read_csv(test_path)  # this is used at the final to test model on new data
    
    # Separate features and labels for training data
    X_train = train_data.drop(columns=['label']).values
    y_train = train_data['label'].values

    # Normalize pixel values to range [0, 1] (for easier ML purposes)
    X_train = X_train / 255.0
    X_validation = test_data.values / 255.0  # Only pixel values in test data

    # One-hot encode the labels (dont want the ml model to assume any relationships between class values (2 vs 1))
    def one_hot_encode(y, num_classes=10):
        one_hot = np.zeros((y.size, num_classes)) # array filled with zeroes of shape (# of data, # of classes (0-9))
        one_hot[np.arange(y.size), y] = 1 # for every row, at each value (0-9) we set to 9
        return one_hot # we have an array that is similar effect as before

    y_train = one_hot_encode(y_train)

    return X_train, X_validation, y_train, train_data['label'].values

# Step 2: Define the neural network structure
class NeuralNetwork:

    # Input Layer: 784 nodes (for flattened 28x28 input images MNIST)
    # Hidden Layer 1: 128 nodes (first hidden layer)   
    # Hidden Layer 2: 64 nodes (second hidden layer)
    # these 128,64 numbers is found through trial and error and logic
    # Output Layer: 10 nodes (for classification into 10 classes)
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        # Initialize weights and biases
        #  Xavier initialization -> sqrt(2/n) increase variance of weights to train better
        # essentially we need a lot more inner "tricks" to make the model good with plain numpy
        self.weights1 = np.random.randn(input_size, hidden_size1) * np.sqrt(2 / input_size)
        self.bias1 = np.zeros((1, hidden_size1))
        self.weights2 = np.random.randn(hidden_size1, hidden_size2) * np.sqrt(2 / hidden_size1)
        self.bias2 = np.zeros((1, hidden_size2))
        self.weights3 = np.random.randn(hidden_size2, output_size) * np.sqrt(2 / hidden_size2)
        self.bias3 = np.zeros((1, output_size))

    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, z):
        return (z > 0).astype(float)

    # for classifcation problems its better to output a probabiliy of what the model thinks
    # as straight numbers may not have the meaning we think it does (i.e higher is "accurate")
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Stability trick
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, X):
        # Input to first hidden layer
        self.z1 = np.dot(X, self.weights1) + self.bias1
        self.a1 = self.relu(self.z1)

        # First hidden to second hidden layer
        self.z2 = np.dot(self.a1, self.weights2) + self.bias2
        self.a2 = self.relu(self.z2)

        # Second hidden to output layer
        self.z3 = np.dot(self.a2, self.weights3) + self.bias3
        self.a3 = self.softmax(self.z3)
        return self.a3

    def compute_loss(self, y_true, y_pred):
        # Cross-entropy loss with L2 regularization
        m = y_true.shape[0]
        # 0.001 because we focus more on cross_entropy (actual - predicted) rather than penalizing large weights (l2)
        l2_term = 0.001 * (np.sum(np.square(self.weights1)) + np.sum(np.square(self.weights2)) \
                           + np.sum(np.square(self.weights3)))
        cross_entropy = -np.sum(y_true * np.log(y_pred + 1e-8)) / m
        loss = cross_entropy + l2_term
        return loss

    def backward(self, X, y_true):
        m = X.shape[0] # number of training samples

        # Output layer gradients
        dz3 = self.a3 - y_true
        dw3 = np.dot(self.a2.T, dz3) / m + 0.001 * self.weights3  # L2 regularization
        db3 = np.sum(dz3, axis=0, keepdims=True) / m

        # Second hidden layer gradients
        dz2 = np.dot(dz3, self.weights3.T) * self.relu_derivative(self.z2)
        dw2 = np.dot(self.a1.T, dz2) / m + 0.001 * self.weights2
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        # First hidden layer gradients
        dz1 = np.dot(dz2, self.weights2.T) * self.relu_derivative(self.z1)
        dw1 = np.dot(X.T, dz1) / m + 0.001 * self.weights1
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # Update weights and biases with learning rate
        learning_rate = 0.005
        self.weights3 -= learning_rate * dw3
        self.bias3 -= learning_rate * db3
        self.weights2 -= learning_rate * dw2
        self.bias2 -= learning_rate * db2
        self.weights1 -= learning_rate * dw1
        self.bias1 -= learning_rate * db1

    def predict(self, X):
        probabilities = self.forward(X)
        return np.argmax(probabilities, axis=1)

# Step 4: Training loop and prediction
def train_and_predict(train_path, test_path, epochs=30, \
                      show_train_pred=False, save_prediction_to_csv=False, save_neural_network=False):
    # Load data
    X_train, X_validation, y_train, y_true_labels = load_data(train_path, test_path)

    # Initialize the neural network
    nn = NeuralNetwork(input_size=784, hidden_size1=128, hidden_size2=64, output_size=10)

    # Training loop
    batch_size = 64 # to make computation easier, we will divide the dataset into batches of 64
    # each epoch means go through every batch until all examples are done
    # then we repeat that for x
    for epoch in range(epochs):
        # splitting training data into batch increments
        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]

            # Forward and backward pass for each batch
            nn.forward(X_batch)
            nn.backward(X_batch, y_batch)

        # Compute loss on the entire dataset
        y_pred_train = nn.forward(X_train)
        loss = nn.compute_loss(y_train, y_pred_train)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

    # Predict on train data to calculate accuracy
    train_predictions = nn.predict(X_train)
    accuracy = np.mean(train_predictions == y_true_labels)
    print(f"Training Accuracy: {accuracy * 100:.2f}%")

    # Predict on validation data
    test_predictions = nn.predict(X_validation)

    # Save predictions to a CSV file
    if save_prediction_to_csv:
        pd.DataFrame({"ImageId": np.arange(1, len(test_predictions) + 1),\
                       "Label": test_predictions}).to_csv("predictions.csv", index=False)
        print("Predictions saved to predictions.csv")
    if save_neural_network:
        pickle.dump(nn, open('network.p', 'wb'))
        print("Neural Network saved to network.p") 
    if show_train_pred:
        # Reshape X_test from (28000, 784) to a list of 28x28 images
        reshaped_images = reshape_images(X_validation)

        # Initialize the ImageNavigator with the reshaped images and predictions
        navigator = ImageNavigator(reshaped_images, test_predictions)

        # Display the plot and start the interactive session
        navigator.show()

if __name__ == "__main__":
    train_and_predict('assets/train.csv', 'assets/test.csv', epochs=30,\
                       show_train_pred=True, save_neural_network=True)