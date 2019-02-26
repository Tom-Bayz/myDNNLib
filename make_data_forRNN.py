# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 13:15:15 2018

@author: T.Yonezu
"""

import pandas as pd
import numpy as np
import seaborn as sns

steps_per_cycle = 50
number_of_cycles = 100

df = pd.DataFrame(np.arange(steps_per_cycle * number_of_cycles + 1), columns=["t"])
df["sin_t"] = df.t.apply(lambda x: np.sin(x * (2 * np.pi / steps_per_cycle)))
df[["sin_t"]].plot()
df[["sin_t"]].head(steps_per_cycle * 2).plot()

df["sin_t+1"] = df["sin_t"].shift(-1)
df.tail()

df.dropna(inplace=True)
df.tail()

df[["sin_t", "sin_t+1"]].head(steps_per_cycle).plot()

matrix = df[["sin_t", "sin_t+1"]].as_matrix()
np.save("data_sin.npy", matrix)