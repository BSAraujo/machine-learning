
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

mnist = tf.keras.datasets.mnist

(X, y),(x_test, y_test) = mnist.load_data()
X, x_test = X / 255.0, x_test / 255.0

#X = X[:20000]
#y = y[:20000]

x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

#val_indices = np.random.choice(len(y_train), size=int(np.floor(len(y_train))), replace=False)
#
#x_val = x_train[val_indices]
#y_val = y_train[val_indices]
#
#x_train = x_train[~val_indices]
#y_train = y_train[~val_indices]

train_acc_crpt_list = []
test_acc_list = []
train_acc_list = []

#corruption_fraction_list = np.array([0.0, 0.1]) #[0.0, 0.1, 0.2, 0.3]
corruption_fraction_list = np.linspace(0,0.9,10)

print("Corruption Fraction List =", corruption_fraction_list)

# Label corruption
for corruption_fraction in corruption_fraction_list:    
    print("Corruption Fraction={}".format(corruption_fraction))
    
    corruption_sz = int(np.floor(corruption_fraction*len(y_train)))
    
    corruption_indices = np.random.choice(len(y_train), size=corruption_sz, replace=False)
    
    y_train_crpt = y_train.copy()
    y_train_crpt[corruption_indices] = np.random.choice(y_train_crpt.max()+1, size=corruption_sz)
    
    model = tf.keras.models.Sequential([
            tf.keras.layers.Conv1D(64, 3, input_shape=(28, 28), activation='relu'),
            tf.keras.layers.Conv1D(128, 3, activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adadelta',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    print("Number of model parameters:", model.count_params())
    
    # simple early stopping
    #es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1)
    
    model.fit(x_train, y_train_crpt, validation_data=(x_val, y_val), epochs=200) #, callbacks=[es])
    print("Model evaluation:", model.evaluate(x_test, y_test)[1])
    
    train_acc_crpt = model.evaluate(x_train, y_train_crpt)[1]
    test_acc = model.evaluate(x_test, y_test)[1]
    train_acc = model.evaluate(x_train, y_train)[1]
    
    print("Train accuracy:", train_acc_crpt)
    print("Test accuracy:", test_acc)
    print("Train accuracy wrt true labels:", train_acc)
    
    train_acc_crpt_list.append(train_acc_crpt)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)

#pred_test = model.predict(x_test)
#pred_test = np.argmax(pred_test, axis=1)
#np.sum(pred_test == y_test) / len(y_test)

import matplotlib.pyplot as plt
plt.plot(corruption_fraction_list*100, train_acc_crpt_list, marker='o', label="Train accuracy")
plt.plot(corruption_fraction_list*100, test_acc_list, marker='s', label="Test accuracy")
plt.plot(corruption_fraction_list*100, train_acc_list, color='red', marker='v', 
         label="Train accuracy w.r.t. true labels")
plt.xlabel("Fraction of labels corrupted (%)")
plt.ylabel("Accuracy (%)")
plt.legend()

plt.savefig('mnist_corrupted_labels.png')

plt.show()


print("Number of model parameters:", model.count_params())

import pandas as pd

data = {
    'corruption_fraction': corruption_fraction_list*100,
    'train_acc': train_acc_crpt_list,
    'test_acc': test_acc_list,
    'train_acc_true_labels': train_acc_list
}
pd.DataFrame(data).to_csv("mnist_corrupted_labels.txt", mode='a', sep='\t', index=False)


#plt.imshow(x_test[1])







