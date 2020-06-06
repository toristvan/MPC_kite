#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 13:19:00 2020

@author: raschid
"""

import numpy as np

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

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
            
    class Kite(Model):
        
        # states
        theta = 0
        phi = 0
        psi = 0
        
        def __init__(self, A=160, L=300, theta=0, phi=0, psi=0 ):
            self.A = A
            self.L = L
            self.__init_trajectory(theta, phi, psi)
 
        def __init_trajectory(self, theta, phi, psi):
            self.trajectory = []
            self.update_trajectory(theta, phi, psi)
            
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
        # TODO
        pass
    
    class Physics(object):
        
        def __init__(self, E0 = 5, vm = 10, vA = 0.5, vf = 0.1, voff = np.pi, c = 0.028, beta = 0, rho = 1):
            self.E0 = E0
            self.vm = vm
            self.vA = vA
            self.vf = vf 
            self.voff = voff
            self.c = c
            self.beta = beta
            self.rho = rho


    def simulate(self, kite):

        thetas = np.linspace(-4 * np.pi, 4 * np.pi, 100)
        phis = 0 * np.ones(thetas.shape)
        psis = 0 * np.ones(thetas.shape)
        
        for theta, phi, psi in zip(thetas, phis, psis):    
            kite.update_trajectory(theta, phi, psi)
            
        self.plot(np.asarray(kite.trajectory)) 
        
    def plot(self, x):
        # TODO always make sure to work with shape = (3, NumberOfTimesteps)
        x = x.T
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        print(x.shape)
        ax.plot(x[0,:], x[1,:], x[2,:], label='kite trajectory')
        ax.legend()
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()
        
        

# main
world = World()
kite = world.Kite()
world.simulate(kite)   
        
        
        
        
        