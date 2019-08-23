"""
Training a SVM with Mathematical Programming Optimization tool Pyomo
"""

import numpy as np
import matplotlib.pyplot as plt
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


# Optimization model - Quadratic Program
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

model = pyo.ConcreteModel()

model.w1 = pyo.Var(within=pyo.Reals)
model.w2 = pyo.Var(within=pyo.Reals)
model.b = pyo.Var(within=pyo.Reals)

model.c = pyo.ConstraintList()

for i in range(num_samples):
    model.c.add(expr = y[i]*(model.w1*x[i,0] + model.w2*x[i,1] + model.b) >= 1)

model.obj = pyo.Objective(expr = 0.5 * (model.w1*model.w1 + model.w2*model.w2), sense=pyo.minimize)

opt = SolverFactory('couenne') 
results = opt.solve(model) 

model.display()
#model.pprint() 

#results.write(num=1)

# Retrieve equation for the optimal separation line
x_plot = np.linspace(0,5,1000)
y_line = - model.w1.value*x_plot / model.w2.value - model.b.value / model.w2.value

# Equations for the margins
y_margin1 = - model.w1.value*x_plot / model.w2.value - model.b.value / model.w2.value + 1 / model.w2.value
y_margin2 = - model.w1.value*x_plot / model.w2.value - model.b.value / model.w2.value - 1 / model.w2.value

# Plot result
plt.figure()
plt.scatter(x[:,0], x[:,1], c=y)
plt.plot(x_plot, y_line, linestyle='dashed', color='gray')
plt.plot(x_plot, y_margin1, linestyle='dashed', color='gray')
plt.plot(x_plot, y_margin2, linestyle='dashed', color='gray')
plt.xlim((0,5))
plt.ylim((0,5))

plt.show()

