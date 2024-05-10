##################################################################################
#----------------------- Ajuste para doble rendija--------------------------------
##################################################################################

import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

x_values = []
y_values = []


with open('Doble_r Laser2.csv', newline='', encoding='utf-8-sig') as csvfile:
    csv_reader = csv.reader(csvfile)
    for row in csv_reader:
        x_values.append((((float(row[0])-3100) / 1000000)+0.01723) * 1.4) #Centrar en 0 y pasar a metros
        y_values.append((float(row[1])/100)-0.002) #Ajustar offset y renormalizar


plt.plot(x_values, y_values, marker='o', linestyle='-')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.title('Datos experimentales')
plt.grid(True)
plt.show()

##################################################################################
#------------------ AJUSTE POR MEDIO DE BEST FIT PARA C,D------------------------
##################################################################################

x0=3280 / 1000000 #OFFSET
A=1 #V_0

def model_function(x, c, d):
    return A * np.cos(d * (x - x0))**2 * (np.sin(c * (x - x0)) / (c * (x - x0)))**2

C_guess = 468 #pi*a/lambda en metros
D_guess = 1875 #pi*d/lambda en metros
initial_guess = (C_guess, D_guess)
optimized_params, covariance = curve_fit(model_function, x_values, y_values, p0=initial_guess, maxfev=200000)


c_optimized, d_optimized = optimized_params
plt.scatter(x_values, y_values, label='Original Data')


x_range = np.linspace(min(x_values), max(x_values), 100)
y_predicted = model_function(x_range, c_optimized, d_optimized)
plt.plot(x_range, y_predicted, color='red', label='Fitted Curve')

plt.xlabel('Distancia (m)')
plt.ylabel('Intensidad')
plt.title('Ajuste por busqueda de parametros')
plt.legend()
plt.grid(True)
plt.show()

print("Optimized parameters:")
print("c:", c_optimized)
print("d:", d_optimized)

"""
x=np.linspace(0.017, 0.02087, 10000)
y=model_function(x, C_guess, D_guess)
plt.plot(x,y, color='blue')
plt.show()
"""






