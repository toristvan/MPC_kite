import numpy as np
import matplotlib.pyplot as plt 
import matplotlib as mpl 
from casadi import *
from casadi.tools import *
#from orthogonal_collocation import Orthogonal_collocation_MPC
from orthogonal_collocation_testing import test_orth_col
#from datetime import datetime

def main():
    
    test_orth_col()

    '''
    L=300
    dt=0.35
    #before = datetime.now().timestamp()
    res_x_mpc, res_u_mpc, costs, solve_times = Orthogonal_collocation_MPC(dt=dt)
    #print(datetime.now().timestamp() - before)
    #print(solve_times)
    #print(costs)
    tsol_mean = np.mean(solve_times)
    tsol_max = np.max(solve_times)

    print("Mean runtime: ",tsol_mean ,"\nMax runtime: ", tsol_max)
    
    fig, ax = plt.subplots(3,3, figsize=(15,12))
    #plot position
    ax[0][0].plot(L*sin(res_x_mpc[0].T)*sin(res_x_mpc[1].T), L*sin(res_x_mpc[0].T)*cos(res_x_mpc[1].T))
    #plot angles towards each other
    ax[1][0].plot(res_x_mpc[1].T, res_x_mpc[0].T)
    # plot the input
    ax[2][0].plot(res_u_mpc.T)

    #plot angles over time
    ax[0][1].plot(res_x_mpc[0].T)
    ax[1][1].plot(res_x_mpc[1].T)
    ax[2][1].plot(res_x_mpc[2].T)

    #plot wind
    #ax[2][1].plot(t_k, v0_fcn(t_k))
    #plot cost
    ax[0][2].plot(costs.T)
    ax[0][2].set_xlabel('time')
    ax[0][2].set_ylabel('cost')

    #plot times
    ax[1][2].plot(solve_times)
    ax[1][2].set_xlabel('number of runs')
    ax[1][2].set_xlabel('time spent on runs [s]')
    ax[1][2].axhline(tsol_mean, label='mean runtime', color='k')
    ax[1][2].axhline(tsol_max, label='max runtime', color='r')
    ax[1][2].legend()


    #ax[0].plot(res_x_mpc[0].T, res_x_mpc[1].T)

    # Set labels
    ax[1][0].set_ylabel('theta')
    ax[1][0].set_xlabel('phi')

    ax[0][0].set_ylabel('height')
    ax[0][0].set_xlabel('horizontal position')

    ax[2][0].set_ylabel('inputs')
    ax[2][0].set_xlabel('time['+str(dt)+' sec]')

    ax[0][1].set_ylabel('theta')
    ax[0][1].set_xlabel('time['+str(dt)+' sec]')

    ax[1][1].set_ylabel('phi')
    ax[1][1].set_xlabel('time['+str(dt)+' sec]')

    ax[2][1].set_ylabel('psi')
    ax[2][1].set_xlabel('time['+str(dt)+' sec]')


    plt.show()
    '''


if __name__ == '__main__':
    main()