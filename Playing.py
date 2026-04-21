import matplotlib.pyplot as plt
import numpy as np

def CalcRfSpider(X):
    Rf = 5 + abs(X)*10/40

    return Rf

X = np.linspace(-45,45,22)

RfSpider = [CalcRfSpider(u) for u in X]

print(RfSpider)


plt.plot(X, RfSpider)
plt.show()
