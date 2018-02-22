import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from math import sqrt 
import numpy as np

global x, y, W, T
# x = range(224000, 284000, 1000)
# y = range(210000, 270000, 1000)
x = range(112000, 288000, 500)
y = range(210000, 275000, 500)
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
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# def E(q, r0, x, y):
#     """Return the electric field vector E=(Ex,Ey) due to charge q at r0."""
#     den = np.hypot(x-r0[0], y-r0[1])**3
#     return q * (x - r0[0]) / den, q * (y - r0[1]) / den

# # Grid of x, y points
# nx, ny = 64, 64
# x = np.linspace(-2, 2, nx)
# y = np.linspace(-2, 2, ny)
# X, Y = np.meshgrid(x, y)

# # Create a multipole with nq charges of alternating sign, equally spaced
# # on the unit circle.
# nq = 2**1
# charges = []
# for i in range(nq):
#     q = i%2 * 2 - 1
#     charges.appxend((q, (np.cos(2*np.pi*i/nq), np.sin(2*np.pi*i/nq))))

# # Electric field vector, E=(Ex, Ey), as separate components
# Ex, Ey = np.zeros((ny, nx)), np.zeros((ny, nx))
# for charge in charges:
#     ex, ey = E(*charge, x=X, y=Y)
#     Ex += ex
#     Ey += ey


fig0, ax0 = plt.subplots()
strm = ax0.streamplot(X, Y, U, V, color=np.sqrt(U**2 + V**2), linewidth=2, cmap=plt.cm.autumn)
fig0.colorbar(strm.lines)

# fig = plt.figure()
# ax = fig.add_subplot(111)

# # Plot the streamlines with an appropriate colormap and arrow style
# color = 2 * np.log(np.hypot(Ex, Ey))
# ax.streamplot(x, y, Ex, Ey, color = color, linewidth=1, cmap=plt.cm.inferno,
#               density=2, arrowstyle='->', arrowsize=1.5)

# # Add filled circles for the charges themselves
# # charge_colors = {True: '#aa0000', False: '#0000aa'}
# # for q, pos in charges:
# #     ax.add_artist(Circle(pos, 0.05, color=charge_colors[q>0]))

# ax.set_xlabel('$x$')
# ax.set_ylabel('$y$')
# ax.set_xlim(-2,2)
# ax.set_ylim(-2,2)
# ax.set_aspect('equal')
plt.show()