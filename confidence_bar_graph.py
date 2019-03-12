import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.DataFrame(np.c_[[0.70, 0.84, 0.86, 0.55, 0.84, 0.83, 0.84, 0.84, 0.88, 0.95, 0.95, 0.94, 0.89, 0.81, 0.80, 0.75, 0.75, 0.58],
                        [0.61, 0.62, 0.63, 0.55, 0.67, 0.69, 0.66, 0.65, 0.81, 0.88, 0.95, 0.95, 0.73, 0.75, 0.72, 0.68, 0.67, 0.59],
                        [0.70, 0.90, 0.92, 0.62, 0.91, 0.90, 0.89, 0.87, 0.90, 0.91, 0.99, 1.00, 0.88, 0.89, 0.88, 0.86, 0.88, 0.60],
                        [0.96, 0.98, 0.98, 0.93, 0.92, 0.89, 0.88, 0.88, 0.99, 0.98, 0.99, 1.00, 0.98, 0.79, 0.77, 0.79, 0.92, 0.94]],
                  columns=['KL score', 'ED score', 'JI score', 'Our score'])



'''[0.91, 0.613, 0.61, 0.91], [0.684, 0.62, 0.78, 0.95], [0.68, 0.63, 0.783, 0.95],
[0.68, 0.55, 0.803, 0.65], [0.757, 0.655, 0.82, 0.57], [0.78, 0.894, 0.81, 0.58],
[0.78, 0.65, 0.82, 0.61], [0.88, 0.81, 0.93, 0.91], [0.95, 0.88, 0.95, 0.92],
[0.55, 0.70, 0.62, 0.69], [0.94, 0.91, 0.97, 0.98], [0.77, 0.71, 0.84, 0.77],
[0.74, 0.68, 0.79, 0.73], [0.74, 0.67, 0.74, 0.76], [0.58, 0.59, 0.60, 0.77]'''

value = df.mean()
print(value)
std = df.std()
print(std)

fig, ax = plt.subplots()
colors = ["red", "green", "blue", "purple"]
# plt.axhline(y=0.87, zorder=0)
plt.bar(np.arange(len(df.columns)), value, width=0.5, bottom=0,
        yerr=std, align='center', alpha=0.5, color=colors)

plt.xticks(range(len(df.columns)), df.columns)
plt.ylabel('AUC score')
plt.title('Confidence Interval Bar Graph')
fig.savefig('/home/ashiq/Pictures/Thesis_image/conf_bar_graph.eps', format='eps')
plt.show()
