import tensorflow as tf
from tensorflow.python import keras
import numpy as np
import keras
import matplotlib.pyplot as plt
import seaborn as sn



keras.preprocessing.image_dataset_from_directory
# mnist institution added few records for testing
# x_train -> training images, y_train -> corresponding label of images
# x_test -> training images, y_test -> corresponding label of images

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# divide by 255 used to scale and get more accurate prediction
x_train = x_train/255
x_test = x_test/255
x_train_flattened = x_train.reshape(len(x_train), 28*28)
x_test_flattened = x_test.reshape(len(x_test), 28*28)

# to test that what values we have in above x_train, y_train training data
# print("Actual label training data value == ", y_train[0])
# plt.matshow(x_train[0])
# plt.show()
#
#  to test that what values we have in above x_test, y_test data
# print("Actual label test data value == ", y_test[0])
# plt.matshow(x_test[0])
# plt.show()

# to check unique labelled images values
# print("Ytrain == ", np.unique(y_test))


# creating neural network, which will return output as 10 numbers, input neural will be 28*28(784),
# and in neural we have 2 linear regression and logistic which is sigmoid function

model = keras.Sequential(
    [
        keras.layers.Dense(10, input_shape=(784, ), activation='sigmoid')
    ])

# here we are creating a model with optimizer
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# here we are running model against training data set and accuracy is around .98%
print("Model running against training dataset")
model.fit(x_train_flattened, y_train, epochs=5)

# now we are evaluating mode against test data set.
print("Model running against test dataset ..")
model.evaluate(x_test_flattened, y_test)

# now we can do prediction from the model as we know that 0th index has 7 so
# so model should predict it's 7
print("Actual image was ==", plt.matshow(x_test[0]))
plt.show()

# all records
x_test_output_weightage = model.predict(x_test_flattened)

# only 1st record after prediction
result_output_predicted = model.predict(tf.expand_dims(x_test_flattened[0], axis=0))
print("Output 1 image has with weightage", result_output_predicted)

# now we can pick max predicted value that will be our answer out of all predictions
print("Prediction is == ", np.argmax(result_output_predicted))


# no we predicted perfectly but if we see confusion matrix where we can see actual fail pass predictions
# x_test_output_weightage but this has for each index it has all weightage 0- 10
# but we need max from each array to check prediction so we need to create max out of all arrays

total_predicated_result_set = [np.argmax(i) for i in x_test_output_weightage]


confusion_matrix = tf.math.confusion_matrix(labels=y_test, predictions=total_predicated_result_set)
# but this won't give us proper visibility so we need to add seaborn library

# so now below code will give us visibility against test data set and prediction
print("Check accuracy....")
plt.figure(figsize=(10, 7))
sn.heatmap(confusion_matrix, annot=True, fmt='d')
plt.xlabel("Predicted")
plt.ylabel('Truth')
plt.show()

# if we see there are lot of prediction which are not matching only
# in our neural network, we only defined input and output, no hidden neural network,
# we'll try defining 1 more middle network and will see the result.
# added random 100 new neural in middle
# model = keras.Sequential(
#     [
#         keras.layers.Dense(10, input_shape=(784, ), activation='sigmoid')
#     ])

model = keras.Sequential(
    [keras.layers.Dense(100, input_shape=(784, ), activation='relu'), ## this one
        keras.layers.Dense(10, activation='sigmoid')])

# optimizer are GD, ie,-  adam/Batch Gradient Descent/ stochastic GD(SGD)/ mini GD
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("Model running 2nd time against training dataset which is taking more time because"
      " more neural got added in middle")
model.fit(x_train_flattened, y_train, epochs=5)

print("Model running 2nd time against test dataset which is taking more time because"
      " more neural got added in middle")
# now we are evaluating mode against test data set.
model.evaluate(x_test_flattened, y_test)

x_test_output_weightage = model.predict(x_test_flattened)

total_predicated_result_set = [np.argmax(i) for i in x_test_output_weightage]
confusion_matrix = tf.math.confusion_matrix(labels=y_test, predictions=total_predicated_result_set)

# to print prediction and truth difference
print("Check accuracy....")
plt.figure(figsize=(10, 7))
sn.heatmap(confusion_matrix, annot=True, fmt='d')
plt.xlabel("Predicted")
plt.ylabel('Truth')
plt.show()





