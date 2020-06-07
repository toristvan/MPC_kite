
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib as mpl 

from casadi import *
from casadi.tools import *


def implicit_euler_discretization():
    #Simulation parameters
    N_sim = 150
    dt = 0.2
    
    #System Parameters
    E0 = 5 # MX.sym("E0")
    vm = 10 # MX.sym("vm")
    vA = 0.5 #1 # MX.sym("vA")
    vf = 0.1 # MX.sym("vf")
    voff = np.pi # MX.sym("voff")
    c = 0.028 # MX.sym("c")
    beta = 0 # MX.sym("beta")
    rho = 1 # MX.sym("rho")
    L = 300 # MX.sym("l")
    A = 160 # MX.sym("A")
    
    #States and control variables
    nx = 3
    nu = 1
    
    x = SX.sym("x", nx, 1)
    u = SX.sym("u", nu, 1)
    u_old = SX.sym("u_old", nu, 1)
    t = SX.sym("t")
    
    #Equations
    v0 = vm + vA*sin(2*np.pi*vf*t + voff)
    v0_fcn = Function('v0_fcn', [t], [v0])
    E = E0 - c*(u**2)
    E_fcn = Function('E_fcn', [u], [E])
    va = v0_fcn(t)*E*cos(x[0])
    va_fcn = Function('va_fcn', [x, t], [va])
    PD = rho*(v0_fcn(t)**2)/2
    PD_fcn = Function('PD_fcn', [t], [PD])
    TF = PD_fcn(t)*A*(cos(x[0])**2)*(E_fcn(u) + 1)*np.sqrt(E_fcn(u)**2 + 1)*(cos(x[0])*cos(beta) + sin(x[0])*sin(beta)*sin(x[1]))
    tension = Function('tension', [x,u,t], [TF])
    
    xdot = vertcat((va_fcn(x, t)/L)*(cos(x[2]) - tan(x[0])/E_fcn(u)), 
    -va_fcn(x, t)*sin(x[2])/(L*sin(x[0])), 
    va_fcn(x,t)*u/L - cos(x[0])*(-va_fcn(x, t)*sin(x[2])/(L*sin(x[0]))))
    
    # System and numerical integration
    system = Function('sys', [x,u,t], [xdot])
    ode = {'x': x, 'ode': xdot, 'p': vertcat(u,t)}
    opts = {'tf': dt}
    ode_solver = integrator('F', 'idas', ode, opts)

    # Define the initial condition
    x_0 = np.array([np.pi/4,0,0]).reshape(nx,1)
    # Define the input (for the moment consider u = 0)
    u_k = np.array([[0]]).reshape(nu,1)
    
    res_x_sundials = [x_0]
    
        
    for i in range(N_sim):
        res_integrator = ode_solver(x0=x_0, p=u_k)
        x_next = res_integrator['xf']
        res_x_sundials.append(x_next)
        x_0 = x_next
    
    # Make an array from the list of arrays:
    res_x_sundials = np.concatenate(res_x_sundials,axis=1)
    
    
    fig, ax = plt.subplots(figsize=(10,6))
    
    # plot the states
    lines = ax.plot(res_x_sundials.T)
    
    
    # Set labels
    ax.set_ylabel('states')
    ax.set_xlabel('time')


