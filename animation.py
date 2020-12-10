import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation


#  transform .txt to .csv
read_file = pd.read_csv (r'VO03.txt')
read_file.columns = ['V']
read_file.to_csv (r'VO03.csv', index=None)

read_file = pd.read_csv (r'GT03.txt')
read_file.columns = ['G']
read_file.to_csv (r'GT03.csv', index=None)

read_file = pd.read_csv (r'SLAM03.txt')
read_file.columns = ['S']
read_file.to_csv (r'SLAM03.csv', index=None)


filename = "VO03.csv"
df = pd.read_csv(filename)

length_of_time = len(df)
x_v, z_v = [], []
for i in range(len(df)):
    ff = df.loc[i, 'V']
    idx = []
    for j in range(len(ff)):
        if ff[j] == ' ':
            idx.append(j)

    x_v.append(float(ff[idx[2]+1:idx[3]]))
    z_v.append(float(ff[idx[10]+1:]))

filename = "GT03.csv"
df = pd.read_csv(filename)

length_of_time = len(df)
x_g, z_g = [], []
for i in range(len(df)):
    ff = df.loc[i, 'G']
    idx = []
    for j in range(len(ff)):
        if ff[j] == ' ':
            idx.append(j)

    x_g.append(float(ff[idx[2]+1:idx[3]]))
    z_g.append(float(ff[idx[10]+1:]))

filename = "SLAM03.csv"
df = pd.read_csv(filename)

length_of_time = len(df)
x_s, z_s = [], []
for i in range(len(df)):
    ff = df.loc[i, 'S']
    idx = []
    for j in range(len(ff)):
        if ff[j] == ' ':
            idx.append(j)

    x_s.append(float(ff[idx[2]+1:idx[3]]))
    z_s.append(float(ff[idx[10]+1:]))

# plt.scatter(x_v, z_v, s=1)
# plt.scatter(x_g, z_g, s=1)
# plt.scatter(x_s, z_s, s=1)
# plt.show()

# Set up formatting for the movie files
#  specify the directory: https://ffmpeg.org/download.html
plt.rcParams['animation.ffmpeg_path'] = '/home/songming/Downloads/ffmpeg-git-20201128-amd64-static/ffmpeg'
Writer = animation.writers['ffmpeg']
writer = Writer(fps=1, metadata=dict(artist='Me'), bitrate=-1)

fig, ax = plt.subplots(figsize=(8, 4))
ax.set(xlim=(-10, 500), ylim=(-10, 250))
line, = plt.plot([], [], '-', markersize=0.5)
plt.xlabel('X/m')
plt.ylabel('Z/m')
plt.title("Trajactory plot")
plotlays, plotcols = [3], ["green", "red", "blue"]
plotlays1, plotlabs = [3], ["VO03", "GT03", "SLAM03"]

lines = []

for index in range(3):
    lobj = plt.plot([], [], '-', markersize=0.5, label=plotlabs[index],  color=plotcols[index])[0]
    lines.append(lobj)

def init():
    for line in lines:
        line.set_data([],[])
    return lines

x_v_data, z_v_data = [], []
x_g_data, z_g_data = [], []
x_s_data, z_s_data = [], []

def animate(frame):
    x_v_data.append(x_v[frame])
    z_v_data.append(z_v[frame])

    x_g_data.append(x_g[frame])
    z_g_data.append(z_g[frame])

    x_s_data.append(x_s[frame])
    z_s_data.append(z_s[frame])

    xlist = [x_v_data, x_g_data, x_s_data]
    ylist = [z_v_data, z_g_data, z_s_data]

    for lnum,line in enumerate(lines):
        line.set_data(xlist[lnum], ylist[lnum]) # set data for each line separately.

    return lines

# modify interval: Delay between frames in milliseconds
anim = animation.FuncAnimation(fig, animate, frames=len(x_v),
                    init_func=init, blit=False, interval=100, repeat=False)

#plt.legend(prop={'size': 10}, markerscale=15)
#plt.show()

anim.save('traj.mp4', writer=writer)




