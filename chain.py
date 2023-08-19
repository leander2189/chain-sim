# %%

import numpy as np
import scipy
from matplotlib import pyplot as plt

N = 20
G = 9.8
L = 1
k = 2
p = np.array([ [0], [0] ], dtype=np.float64)
q = np.array([ [10], [1] ], dtype=np.float64)

def constraint_x(x):
    d = q[0] - p[0]
    d -= L*np.sum(np.cos(x))
    return d

def constraint_y(x):
    d = q[1] - p[1]
    d -= L*np.sum(np.sin(x))
    return d

def potential(x):
    K = 0
    V = 0

    for i in range(N-1):
        V += G*L*(N-1)*np.sin(x[i])
        K += 0.5*k* (x[i+1] - x[i])**2

    pot = V + K
    return pot

d = q - p
angle = np.arctan2(d[1], d[0])
x0 = np.linspace(angle-0.1, angle+0.1, N)[:, 0]

# print(angle)
# print(x0)

constraints =[{
    'type': 'eq',
    'fun': constraint_x,
},
{
    'type': 'eq',
    'fun': constraint_y,
}]

options = {
    'maxiter': 10000,
    'disp': True
}

solution = scipy.optimize.minimize(potential, x0, constraints=constraints, options=options)
print(solution)

# %%

links = []
links.append(p)
for i in range(N):
    li = links[-1] + L*np.array([[np.cos(solution.x[i])], [np.sin(solution.x[i])]])
    links.append(li)

links = np.array(links)

# print(links)
plt.plot(links[:,0,0], links[:,1,0])
plt.show()
