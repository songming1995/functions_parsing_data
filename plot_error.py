import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# #  transform .txt to .csv
read_file = pd.read_csv (r'error03.txt')
read_file.columns = ['error']
read_file.to_csv (r'error03.csv', index=None)

filename = "error03.csv"
df = pd.read_csv(filename)

length_of_time_error = len(df)
print("The length of error is: ", length_of_time_error)
w_x, w_y, w_z, t_x, t_y, t_z = [], [], [], [], [], []
for i in range(len(df)):
    ff = df.loc[i, 'error']
    idx = []
    for j in range(len(ff)):
        if ff[j] == ' ':
            idx.append(j)
    w_x.append(float(ff[0:idx[0]]))
    w_y.append(float(ff[idx[0]+1:idx[1]]))
    w_z.append(float(ff[idx[1]+1:idx[2]]))
    t_x.append(float(ff[idx[2]+1:idx[3]]))
    t_y.append(float(ff[idx[3]+1:idx[4]]))
    t_z.append(float(ff[idx[4]+1:]))

#  transform .txt to .csv
#read_file = pd.read_csv (r'TrajectoryVariance03.txt')
read_file = pd.read_csv (r'SLAMVariance03.txt')
read_file.columns = ['sigma']
read_file.to_csv (r'TrajectoryVariance03.csv', index=None)

filename = "TrajectoryVariance03.csv"
df = pd.read_csv(filename)
pf = []

length_of_time = len(df)
print("The length of covariance is: ", length_of_time)
sigma_w_x, sigma_w_y, sigma_w_z, sigma_t_x, sigma_t_y, sigma_t_z = [], [], [], [], [], []
exception_idx = []
for i in range(len(df)):
    ff = df.loc[i, 'sigma']
    idx = []
    for j in range(len(ff)):
        if ff[j] == ' ':
            idx.append(j)
    sigma_w_x.append(float(ff[0:idx[0]]))
    sigma_w_y.append(float(ff[idx[0]+1:idx[1]]))
    sigma_w_z.append(float(ff[idx[1]+1:idx[2]]))
    sigma_t_x.append(float(ff[idx[2]+1:idx[3]]))
    sigma_t_y.append(float(ff[idx[3]+1:idx[4]]))
    sigma_t_z.append(float(ff[idx[4]+1:]))
    if not(np.isfinite(sigma_w_x[i]) and sigma_w_x[i]< 10 and \
        np.isfinite(sigma_w_y[i]) and sigma_w_y[i]< 10 and \
            np.isfinite(sigma_w_z[i]) and sigma_w_z[i]< 10 and \
            np.isfinite(sigma_t_x[i]) and sigma_t_x[i] < 10 and \
            np.isfinite(sigma_t_y[i]) and sigma_t_y[i]< 10 and \
            np.isfinite(sigma_t_z[i]) and sigma_t_z[i] < 10):
        exception_idx.append(i)
print("Print the frame index over the threshold:")
print(len(exception_idx), exception_idx)
time_step = np.arange(length_of_time)
normal_set = set(np.arange(length_of_time))-set(np.array(exception_idx))
normal_idx = np.array(list(normal_set))

# plot sigma_w_y
plot1 = plt.figure(1,figsize=(10 , 8))
plt.scatter(time_step[normal_idx],
            3*np.array(sigma_w_y)[normal_idx],  c='r', marker='o', s=2, label="+sigma_w_y")
plt.scatter(time_step[normal_idx],
            -3*np.array(sigma_w_y)[normal_idx],  c='b', marker='o', s=2, label="-sigma_w_y")
plt.scatter(np.arange(length_of_time_error), w_y,  c='g', marker='o', s=2, label="error_w_y")

for i in exception_idx:
    plt.vlines(i, min(w_y)/2, max(w_y)/2, colors='k', linestyles='dashed')
plt.xlabel('Time index')
plt.ylabel('Value')
plt.legend()
plt.savefig('w_y.pdf')

# plot sigma_tx
plot2 = plt.figure(2, figsize=(10 , 8))

plt.scatter(time_step[normal_idx],
            3*np.array(sigma_t_x)[normal_idx],  c='r', marker='o', s=2, label="+sigma_t_x")
plt.scatter(time_step[normal_idx],
            -3*np.array(sigma_t_x)[normal_idx],  c='b', marker='o', s=2, label="-sigma_t_x")
plt.scatter(np.arange(length_of_time_error), t_x,  c='g', marker='o', s=2, label="error_t_x")

for i in exception_idx:
    plt.vlines(i, min(t_x)/2, max(t_x)/2, colors='k', linestyles='dashed')

plt.xlabel('Time index')
plt.ylabel('Value')
plt.legend()
plt.savefig('t_x.pdf')

# plot sigma_ty
plot2 = plt.figure(3, figsize=(10 , 8))

plt.scatter(time_step[normal_idx],
            3*np.array(sigma_t_z)[normal_idx],  c='r', marker='o', s=2, label="+sigma_t_z")
plt.scatter(time_step[normal_idx],
            -3*np.array(sigma_t_z)[normal_idx],  c='b', marker='o', s=2, label="-sigma_t_z")
plt.scatter(np.arange(length_of_time_error), t_z,  c='g', marker='o', s=2, label="error_t_z")

for i in exception_idx:
    plt.vlines(i, min(t_z)/2, max(t_z)/2, colors='k', linestyles='dashed')
plt.xlabel('Time index')
plt.ylabel('Value')
plt.legend()
plt.savefig('t_z.pdf')

plt.show()


