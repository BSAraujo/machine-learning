import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.svm import SVC

# load the data

num_samples = 200

# Generate the points
y = np.zeros(num_samples)

r = np.random.rand((num_samples))
theta = np.random.rand(num_samples)*2*np.pi
x = np.vstack([r*np.cos(theta), r*np.sin(theta)]).T

x[:int(num_samples/2),0] = x[:int(num_samples/2),0] + 2
x[:int(num_samples/2),1] = x[:int(num_samples/2),1] + 1
y[:int(num_samples/2)] = 1

x[int(num_samples/2):,0] = x[int(num_samples/2):,0] + 1
x[int(num_samples/2):,1] = x[int(num_samples/2):,1] + 3
y[int(num_samples/2):] = -1


plt.figure()
i = 1
C_values = (0.01, 0.1, 0.5, 1.0, 5.0, 10.0)
for C in C_values:
    clf = SVC(C=C, kernel='linear')
    clf.fit(x, y)
    print("C:", C)

    # get the separating hyperplane
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(0,5,1000)
    yy = a * xx - (clf.intercept_[0]) / w[1]

    margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
    yy_down = yy - np.sqrt(1 + a ** 2) * margin
    yy_up = yy + np.sqrt(1 + a ** 2) * margin

    # plot the line, the points, and the nearest vectors to the plane
    plt.subplot(int(np.ceil(len(C_values)/2)),2,i)
    # plt.clf()
    plt.scatter(x[:,0], x[:,1], c=y)
    plt.plot(xx, yy, linestyle='dashed', color='gray')
    plt.plot(xx, yy_up, linestyle='dashed', color='gray')
    plt.plot(xx, yy_down, linestyle='dashed', color='gray')
    plt.title("C: {}".format(C))
    
    i += 1

plt.show()