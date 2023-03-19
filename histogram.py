# import libraries

import matplotlib.pyplot as plt
import numpy as np


# read/simulate data

# measured times
times = np.array([5.65, 5.69, 5.68, 5.67, 5.70, 5.66, 5.63, 5.66, 5.77, 5.67])

# simulated times
# times = np.random.normal(5.28, 0.05, 100)


# plot histogram

plt.style.use('seaborn-v0_8-whitegrid')     # create plot object with style
plt.figure(figsize=(10,5), dpi=300)         # set size and resolution
plt.title('Histogram of Times')             # add title
plt.xlabel('Time (s)')                      # add x-axis label
plt.ylabel('N')                             # add y-axis label

plt.hist(times, bins=25)                    # plot histogram with N bins
plt.savefig('plots/histogram.pdf')          # display plot object


# calculate mean and errors

N = len(times) # number of data points
t_mean = times.mean() # average value
t_max = max(times) # maximum value
t_min = min(times) # minimum value
dt = (t_max - t_min)/2 # error
dt_pos = [t_max - t_mean] * N # positive error (list)
dt_neg = [t_mean - t_min] * N # negativ error (list)

print(f't = ({t_mean:.2f} ± {dt:.2f}) s')


# residual plot

plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(10,5), dpi=300)
plt.title('Variation of Measured Times')
plt.xlabel('Measurement #')
plt.ylabel('Deviation from Mean Value (s)')

# plot residuals with symmetric error bars
# plt.errorbar(range(1, len(times)+1), times - t_mean, yerr=dt, fmt='.', ecolor='black', capsize=2)

# plot residuals with positive/negative error bars
plt.errorbar(range(1, len(times)+1), times - t_mean, yerr=[dt_pos, dt_neg], fmt='.', ecolor='black', capsize=2)

plt.savefig('plots/residuals.pdf')