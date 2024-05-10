# -*- coding: utf-8 -*-
"""
Created on Thu May  9 23:42:40 2024

@author: lisau
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 


data1 = pd.read_csv('bombillo.csv', delimiter=';')

data1['Micro'] = ((data1['Micro']-3500)/1000000)


x = pd.to_numeric(data1['Micro'])
y = pd.to_numeric(data1['Promedio'])
 

plt.scatter(x,y, color='blue' )
plt.grid()
plt.title("Voltaje vs distancia")
plt.xlabel("Desplazamineto(m)")
plt.ylabel("Conteos (N)")
plt.show()