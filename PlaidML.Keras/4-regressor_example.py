"""
To know more or get code samples, please visit my website:
https://morvanzhou.github.io/tutorials/
Or search: 莫烦Python
Thank you for supporting!
"""

# please note, all tutorial code are running under python3.5.
# If you use the version like python2.7, please modify the code accordingly

# 4 - Regressor example
import numpy as np

# Install the plaidml backend
import plaidml.keras
plaidml.keras.install_backend()
import plaidml.keras.backend


from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

np.random.seed(1337)  # for reproducibility

# create some data

data_size=64*12
trainig_size= 64*11
test_size=64
batch_size=64
X = np.linspace(-1, 1, data_size)
np.random.shuffle(X)    # randomize the data
Y = 0.5 * X + 2 + np.random.normal(0, 0.05, (data_size, ))
# plot data
plt.scatter(X, Y)
plt.show()

X_train, Y_train = X[:trainig_size], Y[:trainig_size]     # first 160 data points
X_test, Y_test = X[trainig_size:], Y[trainig_size:]       # last 40 data points

# build a neural network from the 1st layer to the last layer
model = Sequential()

#JianYang, batchedTraining


model.add(Dense(units=1, input_dim=1)) 

# choose loss function and optimizing method
model.compile(loss='mse', optimizer='sgd')

# training
print('Training -----------')
if 0:
    for step in range(64*12*3+1):
        cost = model.train_on_batch(X_train, Y_train)
        if step % 300 == 0:
            print('train cost: ', cost)

else: 
    #Jian Yang: June 15, 2018 comments
    #Batch_Size =1,2,4 have predicted result 
    #batch_size =8,16,32,64 does not promise?
    
    thehistory = model.fit( X_train, Y_train, batch_size=4, epochs=8)
    print(thehistory.history) 
        
# test
print('\nTesting ------------')
cost = model.evaluate(X_test, Y_test, batch_size=test_size)
print('test cost:', cost)
W, b = model.layers[0].get_weights()
print('Weights=', W, '\nbiases=', b)

# plotting the prediction
Y_pred = model.predict(X_test)
plt.scatter(X_test, Y_test)
plt.plot(X_test, Y_pred)
plt.show()
