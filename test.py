import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from math import sqrt 
import numpy as np

global x, y, W, T
x = range(224000, 284000, 200)
y = range(210000, 270000, 200)
# x = range(112000, 288000, 500)
# y = range(210000, 275000, 500)
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
            h.append([result[0], result[1]])
        u.append(x_tr)
        v.append(y_tr)
        mv.append(h)
    U = np.asarray(u)
    V = np.asarray(v)
    [X, Y] = np.meshgrid(x[W: len(x)-W], y[W: len(y)-W])
            
            
plt.figure()
Q = plt.quiver(X, Y, U, V, units='width')
#qk = plt.quiverkey(Q, 0.9, 0.9, 2, r'$2 \frac{m}{s}$', labelpos='E',coordinates='figure')
plt.show()

import numpy as np
import pylab as plt

import lic_internal

dpi = 100
sizex = len(mv)
sizey = len(mv[0])
video = False
texture = np.random.rand(sizey,sizex).astype(np.float32)

plt.bone()
frame=0
vectors = np.asarray(mv, dtype=np.float32)
kernellen = 31
kernel = np.sin(np.arange(kernellen)*np.pi/kernellen)
kernel = kernel.astype(np.float32)

image = lic_internal.line_integral_convolution(vectors, texture, kernel)

plt.clf()
plt.axis('off')
plt.figimage(image)
plt.gcf().set_size_inches((sizey/float(dpi),sizex/float(dpi)))
# plt.gcf().set_size_inches((7,7))
plt.savefig("flow-image.png",dpi=dpi)

