#!/usr/bin/env python

from scipy.interpolate import interp1d
import numpy as np

y = np.linspace(0,10,10)
x = np.cos(-y**2/8.0)
f = interp1d(x,y)
f2 = interp1d(x,y,kind='cubic')

ynew = np.linspace(0, 10, 40)
import matplotlib.pyplot as plt
plt.plot(x,y,'o')#, f(ynew), xnew,'-', f2(ynew), xnew,'--')
plt.legend(['data', 'linear', 'cubic'], loc='best')
plt.show()
