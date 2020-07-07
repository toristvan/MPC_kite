import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib._color_data as mcd
import math
import matplotlib.ticker as tck
from casadi import *

L = 300

x_ss = np.loadtxt("Plots/metadata/x_ss_data.dat")
u_ss = np.loadtxt("Plots/metadata/u_ss_data.dat") 
cost_ss = np.loadtxt("Plots/metadata/cost_ss_data.dat")
time_ss = np.loadtxt("Plots/metadata/time_ss_data.dat")

x_oc = np.loadtxt("Plots/metadata/x_oc_data.dat")
u_oc = np.loadtxt("Plots/metadata/u_oc_data.dat")
cost_oc = np.loadtxt("Plots/metadata/cost_oc_data.dat")
time_oc = np.loadtxt("Plots/metadata/time_oc_data.dat")

x_ie = np.loadtxt("Plots/metadata/x_ie_data.dat")
u_ie = np.loadtxt("Plots/metadata/u_ie_data.dat")
cost_ie = np.loadtxt("Plots/metadata/cost_ie_data.dat")
time_ie = np.loadtxt("Plots/metadata/time_ie_data.dat")

methods = ["Single shooting", "Orthogonal collocation", "Implicit Euler"]

x = [x_ss, x_oc, x_ie]
u = [u_ss, u_oc, u_ie]
cost = [cost_ss, cost_oc, cost_ie]
time = [time_ss, time_oc, time_ie]

#fix
dts = [2, 0.2, 0.2]
N_sims = [80, 200, 1000]

plotcolors = []
for color in mcd.XKCD_COLORS:
    plotcolors = plotcolors + [color]
color_factor=1

## PLOTTING
fig, ax = plt.subplots(2,2, figsize=(20,16))
fig.suptitle("Comparison of single shooting, orthogonal collocation and implicit euler")


#position
ax[0][0].set_xlabel("Horizontal [m]")
ax[0][0].set_ylabel("Vertical [m]")
#ax[0][0].set_xticks(np.linspace(0, int(N_sims[0]*dts[0])), int(N_sims[0]*dts[0]))
#ax[0][0].get_xaxis().set_major_formatter(tck.FuncFormatter(lambda x, p: format(int(x*dts[0]), ',')))


#control input 
ax[1][0].set_title("Control input")
ax[1][0].set_xlabel("Time") #figure out format
ax[1][0].set_ylabel("Force [N]")
#ax[0][0].set_xticks(np.linspace(0, int(N_sims[0]*dts[0])), int(N_sims[0]*dts[0]))
ax[1][0].get_xaxis().set_major_formatter(tck.FuncFormatter(lambda x, p: format(int(x*dts[1]), ',')))



#cost
ax[0][1].set_title("Mean cost")
ax[0][1].set_ylabel('Cost')
ax[0][1].set_xticks([])
#computation time
ax[1][1].set_title("Mean runtime")
ax[1][1].set_ylabel("Runtime [s]")
ax[1][1].set_xticks([])



for i in range(3):
    figk, axk = plt.subplots(2,2, figsize=(20,16))
    figk.suptitle(methods[i])

    #position
    axk[0][0].plot(L*sin(x[i][0].T)*sin(x[i][1].T), L*sin(x[i][0].T)*cos(x[i][1].T), color = plotcolors[i*color_factor])
    axk[0][0].set_title("Position of kite")
    axk[0][0].set_xlabel("Horizontal [m]")
    axk[0][0].set_ylabel("Vertical [m]")
    #axk[0][0].set_xticks(np.linspace(0, int(N_sims[i]*dts[i])), int(N_sims[i]*dts[i]))
    #axk[0][0].get_xaxis().set_major_formatter(tck.FuncFormatter(lambda x, p: format(int(x*dts[i]), ',')))



    ax[0][0].plot(L*sin(x[i][0].T)*sin(x[i][1].T), L*sin(x[i][0].T)*cos(x[i][1].T), label = methods[i], color = plotcolors[i*color_factor])

    #input
    axk[1][0].plot(u[i].T, color = plotcolors[i*color_factor])
    axk[1][0].set_title("Control input")
    axk[1][0].set_xlabel("Time") #figure out format
    axk[1][0].set_ylabel("Force [N]")
    #axk[1][0].set_xticks(np.linspace(0, int(N_sims[i]*dts[i])), int(N_sims[i]*dts[i]))
    axk[1][0].get_xaxis().set_major_formatter(tck.FuncFormatter(lambda x, p: format(int(x*dts[i]), ',')))
    
    #For when ss has larger dt
    scaled_u = []
    if i == 0:
        for inp in u[i]:
            for j in range(5):
                scaled_u.append(inp)
        scaled_u = np.array(scaled_u)
        ax[1][0].plot(scaled_u.T, label = methods[i], color = plotcolors[i*color_factor])
    else:
        ax[1][0].plot(u[i].T, label = methods[i], color = plotcolors[i*color_factor])

    #cost
    cost_mean = np.mean(cost[i])
    if i == 2:
        cost_mean = cost_mean/N_sims[i]
    
    axk[0][1].plot(cost[i].T)
    axk[0][1].axhline(cost_mean, label = "Mean cost", color = plotcolors[i*color_factor])
    axk[0][1].set_xlabel('Time [s]')
    axk[0][1].set_ylabel('Cost')
    axk[0][1].set_title("Value of cost function")
    #axk[0][1].set_xticks(np.linspace(0, int(N_sims[i]*dts[i])), int(N_sims[i]*dts[i]))
    axk[0][1].get_xaxis().set_major_formatter(tck.FuncFormatter(lambda x, p: format(int(x*dts[i]), ',')))

    axk[0][1].legend()


    ax[0][1].axhline(cost_mean, label = methods[i], color = plotcolors[i*color_factor])

    #computation time
    tsol_mean = np.mean(time[i])
    tsol_max = np.max(time[i])

    axk[1][1].plot(time[i])
    axk[1][1].set_title("Runtimes")
    axk[1][1].set_xlabel('Iterations')
    axk[1][1].set_ylabel('Runtime [s]')
    axk[1][1].axhline(tsol_mean, label='Mean runtime: ' + str('{0:.3g}'.format(tsol_mean)) + 's', color='k')
    axk[1][1].axhline(tsol_max, label='Max runtime: ' + str('{0:.3g}'.format(tsol_max)) + 's', color='r')
    axk[1][1].legend()

    ax[1][1].axhline(tsol_mean, label=methods[i] + str('{0:.3g}'.format(tsol_mean)) + 's', color=plotcolors[i*color_factor])

    figk.savefig("Plots/present/png/"+methods[i].replace(" ", "_") +".png")
    figk.savefig("Plots/present/eps/"+methods[i].replace(" ", "_") +".eps")

    


ax[0][0].legend()
ax[1][0].legend()
ax[0][1].legend()
ax[1][1].legend()

fig.savefig("Plots/present/png/all_methods.png")
fig.savefig("Plots/present/eps/all_methods.eps")
