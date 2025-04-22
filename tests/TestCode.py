from mftoolbox.mftoolbox import *
import numpy as np 
import matplotlib.pyplot as plt 
from tqdm import tqdm 
import math

# Plot parameters 
plt.rcParams.update({
    "font.family": "georgia",
    'text.latex.preamble': r'\\usepackage{amsmath}',
    'mathtext.fontset': 'cm',
})

# Define number of low-fidelity functions 
K = 2 

# Define a list of sample sizes
sample_sizes = [10, 25, 50]

# Define the high and low-fidelity functions
funcs = [
    lambda x: np.exp(-x) * np.sin(2*math.pi*x), 
    lambda x: np.sin(2*math.pi*x),
    lambda x: np.exp(-x)
]

xmin = 0
xmax = 5

Xtest = np.linspace(xmin,xmax,1000).reshape(-1,1)
Ytest = funcs[0](Xtest)

# Creating data dictionary
d = {}

# Generating nested data 
for k in range(K+1):
    # Setting seed for repeatability
    np.random.seed(43)
    
    # Generating the input data 
    d[k] = {
        'X':np.random.uniform(xmin, xmax, size=sum(sample_sizes[:k+1])).reshape(-1,1), 
    }
    # Generating the output training data
    d[k]['Y'] = funcs[k](d[k]['X']).ravel()


# Creating multi-fidelity regressor object
model = NARGP(d)

model.fit(np.ones(1)*1e-2, lr = 1e-1, max_iter = 500)

Yhat, Ystd = model.predict(Xtest)

plt.figure(figsize=(10,4), dpi = 200)
plt.plot(Xtest.ravel(), Yhat.ravel(), color = 'orange')
plt.plot(Xtest.ravel(), Ytest.ravel(), color = 'black', linestyle = 'dotted')
plt.fill_between(Xtest.ravel(), Yhat.ravel() - Ystd, Yhat.ravel() + Ystd, color = 'orange', alpha = 0.5)
plt.grid()
plt.show()