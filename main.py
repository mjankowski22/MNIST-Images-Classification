import tensorflow as tf
import numpy as np 

(train_data,train_labels), (test_data, test_labels) = tf.keras.datasets.mnist.load_data()

train_data = train_data /255.0
test_data = test_data /255.0

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(90,activation="relu"),
    tf.keras.layers.Dense(10,activation="relu"),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])

model.fit(train_data,train_labels,epochs=8)
test_loos,test_acc = model.evaluate(test_data,test_labels)
print(f"accuracy: {test_acc}")