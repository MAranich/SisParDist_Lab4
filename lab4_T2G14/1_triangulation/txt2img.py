import matplotlib.pyplot as plt
import numpy as np
import sys

imgName = str(sys.argv[1])
filein  = imgName + ".txt"

img = np.loadtxt(filein)
num_max = np.max(img[img < 100])
print(num_max)
cmap = plt.cm.jet
cmap.set_under('white')
cmap.set_over('black')
plt.imshow(img, cmap=cmap, vmin=-0.5, vmax=num_max )
plt.show()
