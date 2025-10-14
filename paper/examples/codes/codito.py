import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def linear_model(x, m, b):
    return m * x + b

def fit_regression(x, y, ex, ey):
    # Ajuste ponderado por errores en y
    popt, pcov = curve_fit(linear_model, x, y, sigma=ey, absolute_sigma=True)
    m_opt, b_opt = popt
    perr = np.sqrt(np.diag(pcov))
    m_err = perr[0]  # Incertidumbre en la pendiente
    return m_opt, b_opt, m_err

def plot_regression(x, y, ex, ey, m_opt, b_opt, m_err):
    plt.errorbar(x, y, xerr=ex, yerr=ey, fmt='o', label='Datos', ecolor='gray', capsize=3)
