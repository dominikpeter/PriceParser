

import pandas as pd
import numpy as np
import json
import os

p = pd.DataFrame({'A': [1,2,3,4,4,4,5,2,3,5,9,6,2,1,23, 23, 23 ,  23]})


p['Count'] =  p.groupby('A')['A'].transform(lambda x: len(x))

p[p.groupby('A')['A'].cumcount() == 0]
