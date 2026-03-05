#%%
# chi-square distribution
save_figures = True
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

x = np.linspace(0, 20, 1000)
ns = [1, 2, 3, 4, 5]

plt.figure(figsize=(4, 4))

for n in ns:
    plt.plot(x, chi2.pdf(x, n), label=f'n = {n}')

plt.ylim(0, 1)
plt.title('$\chi^2$ distribution')
plt.xlabel('x')
plt.ylabel('dP/dx')
plt.legend(title='$n$: degrees of freedom', loc='upper right')
if save_figures:
    plt.savefig('./probability-distributions-assets/chi-square-distribution.png', dpi=300, bbox_inches='tight')
plt.show()

## Minimum Chi-Square Method
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import chi2

# 1. Define the model function from the image
def model(x, A, B):
    return A * x * np.exp(-B * x)

# 2. Generate synthetic data points with "measurement" errors
np.random.seed(42)
x_data = np.array([0.5, 1.2, 2.1, 3.2, 4.0, 5.1, 5.9, 7.0, 7.9, 9.1, 10.0])
y_err = np.array([0.4, 0.5, 0.6, 0.4, 0.5, 0.4, 0.3, 0.3, 0.3, 0.2, 0.2])

# Generate y values based on A=9, B=0.5 with added noise
y_obs = model(x_data, 9.0, 0.5) + np.random.normal(0, y_err)

# 3. Perform the Minimum Chi-Square Fit
# 'sigma' allows the fit to weigh points by their individual errors
popt, pcov = curve_fit(model, x_data, y_obs, p0=[10, 0.5], sigma=y_err)
A_fit, B_fit = popt

# Calculate Chi-Square statistics
residuals = y_obs - model(x_data, *popt)
chi_sq = np.sum((residuals / y_err)**2)
df = len(x_data) - len(popt)
reduced_chi_sq = chi_sq / df

# 4. Create the dual-panel plot
fig = plt.figure(figsize=(4, 4))
gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.2)

# Top Panel: Data and the Fitted Curve
ax1 = fig.add_subplot(gs[0])
x_smooth = np.linspace(0, 11, 100)
ax1.errorbar(x_data, y_obs, yerr=y_err, fmt='ko', capsize=3, label='Data')
ax1.plot(x_smooth, model(x_smooth, *popt), color='darkblue', lw=2, label='Fit')
ax1.set_ylabel('y')
ax1.set_title(f'Minimum $\chi^2$ fit (reduced $\chi^2$={chi_sq:.2f}/{df:.2f}={reduced_chi_sq:.2f})')
ax1.text(6, 5, r'$y = A\,x\,e^{-B\,x}$', fontsize=15, color='darkblue')
ax1.grid(True, alpha=0.2)

# Bottom Panel: Residuals
ax2 = fig.add_subplot(gs[1], sharex=ax1)
ax2.errorbar(x_data, residuals, yerr=y_err, fmt='ko', capsize=3)
ax2.axhline(0, color='black', linestyle='--')
ax2.set_ylabel('Residuals')
ax2.set_xlabel('x')
ax2.set_ylim(-1, 1)
ax2.grid(True, alpha=0.2)

if save_figures:
    plt.savefig('./probability-distributions-assets/minimum-chi-square-fit.png', dpi=300, bbox_inches='tight')
plt.show()

#%%