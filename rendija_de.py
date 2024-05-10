# -*- coding: utf-8 -*-
"""
Created on Thu May  9 22:15:50 2024

@author: lisau
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

data2 = pd.read_csv('r_right.csv', delimiter=';')

data2['Micrometro'] = ((data2['Micrometro']-4200)/1000000)
data2['Voltaje'] = (data2['Voltaje']*1000)

x2 = pd.to_numeric(data2['Micrometro'])
y2 = pd.to_numeric(data2['Voltaje'])

a=0.1e-3
d=0.356e-3


def model(x,Vo,B,C,D):
    return Vo*np.cos(B*x+C)**2 + D
    
Vo_guess = 125
B_guess = np.pi*d/670 
C_guess = 15
D_guess = 24
initial_guess = (Vo_guess, B_guess, C_guess, D_guess)


opti_param, covariance = curve_fit(model,x2,y2,p0=initial_guess)


Vo_opti, b_opti, c_opti, d_opti = opti_param
plt.scatter(x2, y2, label='Original Data')


x_range = np.linspace(min(x2), max(x2), 100)
y_new = model(x_range, Vo_opti, b_opti,c_opti, d_opti)
plt.plot(x_range, y_new, color='purple', label='Fitted Curve')


plt.xlabel('Distancia (m)')
plt.ylabel('Voltaje(mV)')
plt.title('Voltaje contra distancia')
plt.legend()
plt.grid(True)
plt.show()


print("Optimized parameters:")
print("Vo:", Vo_opti)
print("b:", b_opti)
print("c:", c_opti)
print("d:", d_opti)
print("incertidumbre (Vo,B,C,D)",covariance[0][0]**(1/2),covariance[1][1]**(1/2),covariance[2][2]**(1/2),covariance[3][3]**(1/2))
print("longitud de onda:" ,np.pi*a/opti_param[1],np.sqrt(np.pi**2*a**2/(opti_param[1]**4)*covariance[1][1]))


res=y2-model(x2,Vo_opti, b_opti, c_opti, d_opti)

plt.scatter(x2, res, c="r")
plt.grid()
plt.xlabel('Distancia (m)')
plt.ylabel('Residuales (mV)')
plt.title('Gráfica de residuales')
plt.show()