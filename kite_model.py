#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 13:19:00 2020

@author: raschid
"""
 
import numpy as np

from casadi import *
from casadi.tools import *

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# TODO Import here the discretization schemes later
from implicit_euler import *

mpl.rcParams['font.size'] = 14

class World(object):
    
    class Model(object):
        
        # general model attributes
        position = np.asarray([0,0,0])
        orientation = 0
        trajectory = []
        
        # constructor
        def __init__(self, position, orientation):
            self.position = position
            self.orientation = orientation
            self.trajectory.append(orientation)
            self.x_0 = self.position
            
    class Kite(Model):
        
        # TODO implement a constraint control for the states (angles)
        
        # states
        nx = 3
        
        def __init__(self, A=160, L=300, theta=0, phi=0, psi=0 ):
            self.A = A
            self.L = L
            self.trajectory = []
            self.update_trajectory(theta, phi, psi)
            self.x_0 = self.position.reshape(self.nx,1)
            
        def update_trajectory(self, theta, phi, psi):
            self.__update_position(theta, phi, psi)
            self.trajectory.append(self.position)
            
        def __update_position(self, theta, phi, psi):
            self.__update_states(theta, phi, psi)
            self.position = self.L * np.array([np.cos(self.theta), np.sin(self.phi)* np.sin(self.theta), -np.cos(self.phi) * np.sin(self.theta) ])
        
        def __update_states(self, theta, phi, psi):
            self.theta = theta
            self.phi = phi
            self.psi = psi
            
        
    class Boat(Model):
        
        # TODO maybe if necessary, add a boat model here
        
        pass
    
    class Physics(object):
        
        def __init__(self, kite, E0=5, vm=10, vA=0.5, vf=0.1, voff=np.pi, c=0.028, beta=0, rho=1):
            self.E0 = E0
            self.vm = vm
            self.vA = vA
            self.vf = vf 
            self.voff = voff
            self.c = c
            self.beta = beta
            self.rho = rho
            
            #States and control variables
            self.nx = 3
            self.nu = 1
            
            self.x = SX.sym("x", self.nx, 1)
            self.u = SX.sym("u", self.nu, 1)
            self.u_old = SX.sym("u_old", self.nu, 1)
            self.t = SX.sym("t")
            
            # Aird dynamic Equations
            self.v0 = self.vm + self.vA * sin(2*np.pi * self.vf * self.t + self.voff)
            self.v0_fcn = Function('v0_fcn', [self.t], [self.v0])
            self.E = self.E0 - self.c*(self.u**2)
            self.E_fcn = Function('E_fcn', [self.u], [self.E])
            self.va = self.v0_fcn(self.t)*self.E*cos(self.x[0])
            self.va_fcn = Function('va_fcn', [self.x, self.t], [self.va])
            self.PD = self.rho*(self.v0_fcn(self.t)**2)/2
            self.PD_fcn = Function('PD_fcn', [self.t], [self.PD])
            self.TF = self.PD_fcn(self.t)*kite.A*(cos(self.x[0])**2)*(self.E_fcn(self.u) + 1)*np.sqrt(self.E_fcn(self.u)**2 + 1)*(cos(self.x[0])*cos(self.beta) + sin(self.x[0])*sin(self.beta)*sin(self.x[1]))
            self.tension = Function('tension', [self.x,self.u,self.t], [self.TF])
            
            self.xdot = vertcat((self.va_fcn(self.x, self.t)/kite.L)*(cos(self.x[2]) - tan(self.x[0])/self.E_fcn(self.u)), 
                                -self.va_fcn(self.x, self.t)*sin(self.x[2])/(kite.L*sin(self.x[0])), 
                                self.va_fcn(self.x,self.t)*self.u/kite.L - cos(self.x[0])*(-self.va_fcn(self.x, self.t)*sin(self.x[2])/(kite.L*sin(self.x[0])))
                                )            
            
            #Constrains
            self.hmin = 100
            self.h = kite.L*sin(self.x[0])*cos(self.x[1])
            self.h_func = Function('h', [self.x], [self.h])
            
            #Cost function/Objective term
            self.wF = 1e-4
            self.wu = 0.5

            #cost_func = Function('cost', [TF, u_new, u],[-wF*TF + wu*(u_new - u)**2])
            self.stage_cost = -self.wF * self.tension(self.x, self.u, self.t) + self.wu * (self.u_old - self.u)**2
            self.stage_cost_fnc = Function('stage_cost', [self.x, self.u, self.u_old], [self.stage_cost])
            

    def simulate(self, kite, physics, u_k, dt=0.2, N_sim = 100):
        x_0 = kite.x_0
        t_k = np.array(np.linspace(dt, N_sim*dt, N_sim))
        res_x_sundials = [x_0]
        
        # System and numerical integration
        system = Function('sys', [physics.x, physics.u, physics.t], [physics.xdot])
        ode = {'x': physics.x, 'ode': physics.xdot, 'p': vertcat(physics.u, physics.t)}
        opts = {'tf': dt}
        ode_solver = integrator('F', 'idas', ode, opts)
        print(ode_solver)
        
        for i in range(N_sim):
            res_integrator = ode_solver(x0=x_0, p=vertcat(u_k,t_k[i]))
            x_next = res_integrator['xf']
            res_x_sundials.append(x_next)
            x_0 = x_next    
        # Make an array from the list of arrays:
        res_x_sundials = np.concatenate(res_x_sundials,axis=1) 
        self.plot(kite, res_x_sundials)
        
        return res_x_sundials
    
    def run_MPC(discretization='implicit_euler'):

        if discretization == 'orthogonal_collocation':
            #orthogonal_collocation_discretization()
            pass
        elif discretization == 'implicit_euler':
            #implicit_euler_discretization()
            pass
        elif discretization == 'single_shooting':
            #single_shooting_discretization()
            pass
        else :
            raise NameError('wrong discretization name')       
        
        
    def plot_kite_trajectory_from_states(self, kite, x):
        
        # TODO Here we should be able to choose the discretization scheme later
        
        N = x.shape[1]
        
        for i in range(N): 
            kite.update_trajectory(x[0,i], x[1,i], x[2,i])

        self.plot(kite, np.asarray(kite.trajectory).T)
        
    def plot_given_trajectory(self, kite, x):
    
        self.plot(kite, x) 
    
    def test_trajectory(self, N):
        x = np.linspace(-4*np.pi, 4*np.pi, N).reshape(1,N), 0*np.ones((1,N)), 0*np.ones((1,N))
        return x
    
    def test_states2(self, N):
        phi_t = np.linspace(- np.pi, np.pi, N)
        thetas =np.linspace(0, np.pi/2, N).reshape(1,N)
        theta_t = np.linspace(0, -np.pi/2, N)
        #thetas = 1* np.sin(theta_t).reshape(1,N)
        psis = 1* np.ones((1,N))
        phis = 1 * np.sin(phi_t).reshape(1,N)
        # states x (not position x)
        x = np.concatenate((thetas,phis,psis), axis=0)
        return x
    
    def spherical_lon_trajectory(self, kite, theta=np.pi/2):
        N = 100
        phi = np.linspace(0, 2*np.pi, N)
        r = kite.L
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = -r * np.cos(theta) * np.ones(N)
        
        x = x.reshape(1,N)
        y = y.reshape(1,N)
        z = z.reshape(1,N)
        
        X = np.concatenate((x,y,z), axis=0) 
        return X
    
    def spherical_lat_trajectory(self, kite, phi=np.pi/2):
        N = 100
        theta = np.linspace(-np.pi/2, np.pi/2, N)
        r = kite.L
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = -r * np.cos(theta) * np.ones(N)
        
        x = x.reshape(1,N)
        y = y.reshape(1,N)
        z = z.reshape(1,N)
        
        X = np.concatenate((x,y,z), axis=0) 
        return X
    
    def spherical_8_trajectory(self, kite, N):
        #N = 100
        t =  np.linspace(0, 2*np.pi, N)
        theta = np.pi/14* (np.sin(2*t) ) + np.pi/4
        
        phi = np.pi/4* np.sin(t)
        r = kite.L
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = -r * np.cos(theta) * np.ones(N)
        
        x = x.reshape(1,N)
        y = y.reshape(1,N)
        z = z.reshape(1,N)
        
        X = np.concatenate((x,y,z), axis=0) 
        return X
    
    def print_states(self,x):
                    
        fig, ax = plt.subplots(figsize=(10,6))    
        # plot the states
        lines = ax.plot(x.T)        
        # Set labels
        ax.set_ylabel('states')
        ax.set_xlabel('time')
        plt.show()
        
    def plot(self,kite, x):
        
        # TODO always make sure to work with shape = (3, NumberOfTimesteps)
        
        # inverse z axis, cause z-axis is pointing downwards with respect to gravity
        #x = x.T
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        plt.gca().invert_zaxis()
        ax.plot(x[0,:], x[1,:], x[2,:], label='kite trajectory')
        
        x_spher0 = self.spherical_lon_trajectory(kite, theta=np.deg2rad(90))
        ax.plot(*x_spher0, color='grey', alpha =0.2)
        x_spher1 = self.spherical_lon_trajectory(kite, theta=np.deg2rad(67.5))
        ax.plot(*x_spher1, color='grey', alpha =0.2)
        x_spher2 = self.spherical_lon_trajectory(kite, theta=np.deg2rad(45))
        ax.plot(*x_spher2, color='grey', alpha =0.2)
        x_spher3 = self.spherical_lon_trajectory(kite, theta=np.deg2rad(22.5))
        ax.plot(*x_spher3, color='grey', alpha =0.2)
        x_spher4 = self.spherical_lon_trajectory(kite, theta=np.deg2rad(5))
        ax.plot(*x_spher4, color='grey', alpha =0.2)
        
        
        x_spher0 = self.spherical_lat_trajectory(kite, phi=np.deg2rad(90))
        ax.plot(*x_spher0, color='grey', alpha =0.2)
        x_spher1 = self.spherical_lat_trajectory(kite, phi=np.deg2rad(0))
        ax.plot(*x_spher1, color='grey', alpha =0.2)
        
        ax.scatter(x[0,0], x[1,0], x[2,0], marker ='o')
        ax.scatter(0, 0, 0, marker ='o', color='k')
        #ax.scatter(x[0,-1], x[1,-1], x[2,-1], marker ='^')
        ax.legend()
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.grid(False)
        ax.set_zlim3d(0, -600)
        plt.show()
    
        

# ---- main -----
# the kite_model basically consists of 3 classes (world, kite, physics)
        
# create world, kite and physics instances
world = World()
kite = world.Kite(theta=np.pi/6, phi=np.pi/6, psi=np.pi/5)
physics = world.Physics(kite)

# test trajectory plot with some example states calculated by test_states2()
kite2 = world.Kite(theta=0, phi=0, psi=0)
x_test = world.test_states2(100)
world.plot_kite_trajectory_from_states(kite2, x_test)

# plot 8 trajectory as an example
x_8 = world.spherical_8_trajectory(kite, 100)
world.plot_given_trajectory(kite, x_8)

# run simulation
nu = 1
u_k = np.array([[0]]).reshape(nu,1)

x_sim = world.simulate(kite, physics, u_k, 2)
world.print_states(x_sim)

# run real MPC problem
#world.run_mpc(kite, physics, x_0, discretization='implicit_euler')




