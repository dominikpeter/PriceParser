
import pandas as pd
import numpy as np
import json
import os

p = pd.DataFrame({'A': [1,2,3,4,5]})

c = os.getcwd()

with open(os.path.join(c, 'settings.json')) as j:
    data = json.load(j)


p.iloc[2,:] = np.nan

p2 = p.loc[p['A']>2,:]
