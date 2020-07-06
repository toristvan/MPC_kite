import numpy as np
import matplotlib.pyplot as plt 
import matplotlib as mpl 
from casadi import *
from casadi.tools import *
from orthogonal_collocation import Orthogonal_collocation_MPC
#from orthogonal_collocation_testing import test_orth_col
#from datetime import datetime
import sys, os

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


def main():
    #blockPrint()
    #Nlpsol.doc("print_iteration")
    #test_orth_col(dt=0.2, Kmin=2, Kmax=2, Nmin=20, Nmax=20, figname="orth_col_all_test_wo_print")
    #Orthogonal_collocation_MPC()
    K=3
    N=50
    N_sim=200
    dt=0.2
    tau_col_str='legendre'
    L=300
    res_x_mpc, res_u_mpc, costs, solve_times = Orthogonal_collocation_MPC(K=K, N=N, N_sim=N_sim, dt=dt, collocation_tech=tau_col_str)

    fig, ax = plt.subplots(3,2, figsize=(21,15))
    
    ax[0][0].plot(L*sin(res_x_mpc[0].T)*sin(res_x_mpc[1].T), L*sin(res_x_mpc[0].T)*cos(res_x_mpc[1].T))
    ax[1][0].plot(res_u_mpc.T)

    tsol_mean=np.mean(solve_times)
    tsol_max=np.max(solve_times)
    ax[2][0].plot(solve_times)
    ax[2][0].set_xlabel('number of runs')
    ax[2][0].set_xlabel('time spent on runs [s]')
    ax[2][0].axhline(tsol_mean, label='mean runtime: ' + str('{0:.3g}'.format(tsol_mean)) + 's', color='k')
    ax[2][0].axhline(tsol_max, label='max runtime: ' + str('{0:.3g}'.format(tsol_max)) + 's', color='r')
    ax[2][0].legend()
    
    plt.show()



if __name__ == '__main__':
    main()