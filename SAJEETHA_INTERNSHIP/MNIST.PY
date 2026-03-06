import tensorflow as tf #importing the library tensorflow and namining it as "tf"
from tensorflow import keras #from tensorflow importing keras for training testing , its a high level deep learning API used in neural network models
from tensorflow.keras import layers #from keras importing layers for implementing layers(input,hidden,output)


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data() #imports the MNIST dataset contains 0-9 in 70,000 images with 28 x 28 size | 60,000 images used for training 10,000 images used for testing(90:10 rule)
#training =approx(85%)
#testing =approx(14%)  

x_train = x_train / 255.0 #normalizing the data ranges
x_test = x_test / 255.0   #normalizing the data ranges


model = keras.Sequential([ # sequential == stacked up with many layers (sequence= input -> hidden -> output)
    
    layers.Flatten(input_shape=(28, 28)), #converts images into 1D vector (28 * 28 = 784 neurons) 
    
    layers.Dense(128, activation='relu'), #Fully connected hidden layer uses relu(rectified linear unit) to introduce non-linearity f(x) = max(0, x)
    
    layers.Dense(10, activation='softmax') #Output layer which has 10 neurons with the activation function of softmax it converts output into probabilities
])

model.compile( #compiling 
    optimizer='adam', #Adam is an advanced gradient descent algorithm. updates weights efficiently.
    loss='sparse_categorical_crossentropy', #Lables integer and predicts error
    metrics=['accuracy'] #metrics of finding accuracy (NUMBER OF CORRECT PREDICTIONS / NUMBER OF PREDICTIONS)
) 


model.fit(x_train, y_train, epochs=5) #fitting the model with training Runs the complete training process 5 times over the entire dataset
test_loss, test_accuracy = model.evaluate(x_test, y_test) # evaluating the model

print("Test Accuracy:", test_accuracy) #prints the accuarcy 
