
# %%
import numpy as np
single_point = [3, 4]
points = np.arange(20).reshape((10,2))

d1 = (points - single_point)**2
d2 = np.sum(d1, axis=1)
d3 = np.sqrt(d2)
# %%

