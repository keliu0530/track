import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from math import sqrt 
import numpy as np

global x, y, W, T
x = range(244000, 264000, 500)
y = range(230000, 250000, 500)
#x = range(112000, 288000, 2000)
#y = range(210000, 275000, 2000)
W = 2
T = 1

def evaluate(f, x, y):
    df = pd.read_csv(f)
    values = df.as_matrix(["x", "y"]).T
    del df
    kde = stats.gaussian_kde(values)
    del values
    return kde.evaluate([np.meshgrid(x,y)[0].T.ravel(), np.meshgrid(x,y)[1].T.ravel()]).reshape(len(x), len(y))

l = ['7am.csv', '8am.csv', '9am.csv']

kde = []
for i in l:
    kde.append(evaluate(i, x, y))
    
    
fig, ax = plt.subplots()
ax.imshow(np.rot90(kde[2]-kde[0]), cmap=plt.cm.gist_earth_r)
plt.show()


def FMap(kde, x, y):
    xyt = np.zeros(3)
    t = len(kde)/2
    ft = kde[t][x][y]
    for p in range(-W, W+1):
        for q in range(-W, W+1):
            for r in range(-T, T+1):
                ftr = kde[t+r][x+p][y+q]
                if sqrt(p*p + q*q) != 0:
                    xyt = xyt + np.array([p, q, r])*ft*ftr/sqrt(p*p + q*q)
    return xyt*(10**22)



u = []
v = []
mv = []
if(len(x) - W > W and len(y) - W > W):
    for i in range(W, len(y) - W):
        x_tr = []
        y_tr = []
        h = []
        for j in range(W, len(x) - W):
            result = FMap(kde, j, i)
            x_tr.append(result[0])
            y_tr.append(result[1])
            h.append((result[0], result[1]))
        u.append(x_tr)
        v.append(y_tr)
        mv.append(h)
    U = np.asarray(u)
    V = np.asarray(v)
    [X, Y] = np.meshgrid(x[W: len(x)-W], y[W: len(y)-W])
            
            
plt.figure()
plt.title('Arrows scale with plot width, not view')
Q = plt.quiver(X, Y, U, V, units='width')
#qk = plt.quiverkey(Q, 0.9, 0.9, 2, r'$2 \frac{m}{s}$', labelpos='E',coordinates='figure')

plt.show()
            
