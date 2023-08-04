"""
    plot alpha-beta-accuracy 3d picture
"""
import numpy as np
from matplotlib import pyplot as plt

# load alpha-beta-accuracy list
res = ''
with open('alpha_beta_accuracy.txt', 'r') as f:
    res = f.read()

res_list = res.split(' ')

# the range of the parameter beta
beta_list = [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.20]
# the range of the parameter alpha
upper_percent_list = [99.9, 99.8, 99.7, 99.6, 99.5, 99.4, 99.3, 99.2, 99.1, 99.0]

res1 = []
for acc_str in res_list:
    acc_str_splits = acc_str.split('-')
    acc = float(acc_str_splits[2])
    res1.append(acc)
res1 = np.reshape(res1, (len(upper_percent_list), len(beta_list)))
print(res1)
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1, projection='3d')

X = upper_percent_list
Y = beta_list
X, Y = np.meshgrid(X, Y)

Z1 = res1.T

ax1.set_xlabel('ρ(%)')
ax1.set_ylabel('β')
ax1.set_zlabel('Accuracy')
ax1.set_zlim(0.7, 0.9)

ax1.plot_surface(X, Y, Z1, cmap='autumn')

ax1.xaxis.set_tick_params(labelsize=10)
ax1.yaxis.set_tick_params(labelsize=10)
ax1.zaxis.set_tick_params(labelsize=10)

ax1.set_title('Accuracy-ρ-β')
plt.gca().invert_xaxis()
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
plt.show()
