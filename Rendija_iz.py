# -*- coding: utf-8 -*-
"""
Created on Thu May  9 20:09:12 2024

@author: lisau
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Lectura de los datos
data1 = pd.read_csv('r_left.csv', delimiter=';')
data1['micrometro'] = (data1['micrometro'] - 3500) / 1000000
data1['Voltaje'] = data1['Voltaje'] * 1000


x1 = pd.to_numeric(data1['micrometro'])
y = pd.to_numeric(data1['Voltaje'])


a=0.1e-3
d=0.356e-3


def model(x,Vo,B,C,D):
    return Vo*np.cos(B*x+C)**2 + D
    
Vo_guess = 175
B_guess = np.pi*d/670 
C_guess = 30
D_guess = 24
initial_guess = (Vo_guess, B_guess, C_guess, D_guess)


opti_param, covariance = curve_fit(model,x1,y,p0=initial_guess)


Vo_opti, b_opti, c_opti, d_opti = opti_param
plt.scatter(x1, y, label='Original Data')


x_range = np.linspace(min(x1), max(x1), 100)
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


res=y-model(x1,Vo_opti, b_opti, c_opti, d_opti)

plt.scatter(x1, res, c="r")
plt.grid()
plt.xlabel('Distancia (m)')
plt.ylabel('Residuales (mV)')
plt.title('Gráfica de residuales')
plt.show()