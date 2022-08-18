#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 11:23:40 2022

@author: ryanmurphy
"""

import time

from firedrake import *
from ufl import tanh

# Load mesh and boundary definitions
mesh = Mesh('mesh/vertebraeFiredrake.msh')

# Define scalar (A) and vector function spaces (P)
A = FunctionSpace(mesh, "CG", 1)
B = FunctionSpace(mesh, "DG", 0)
P = VectorFunctionSpace(mesh, "CG", 1)

# Material properties
E = 10e9
nu = 0.3

# Lamé parameters
lmbda = Constant(E*nu/((1+nu)*(1-2*nu)))
mu = Constant(E/2/(1+nu))

# Mesh coordinates
x = SpatialCoordinate(A.mesh())

# Constitutive relations
def epsilon(v):
    return sym(nabla_grad(v))

def sigma(u):
    return lmbda*nabla_div(u)*Identity(3) + 2*mu*epsilon(u)   

# Pseudo-density functional
def stiffness(pseudoDensity):
    return 1e-3 + (1 - 1e-3)*pseudoDensity

# FEA     
def forwardProblem(tumourRadius):

    #Define scalar functional
    # 0 - tumour
    # 1 - bone   
    
    #Tumour center
    xCentre = Constant(0.0)
    yCentre = Constant(1.175)
    zCentre = Constant(0.0)
    
    tumourRadius = Constant(tumourRadius)
    pseudoDensityFunction = interpolate(0.5 * (tanh(1e8*(pow(x[0] - xCentre,2) + pow(x[1] - yCentre,2) + pow(x[2] - zCentre,2) - pow(tumourRadius,2))) + 1), A)
        
    #Define test and trial functions
    u = TrialFunction(P)
    v = TestFunction(P)
    
    #Define boundary conditions    
    bc = [DirichletBC(P, 0, [3])]
    
    #Half sine wave load sweep
    timeSteps = 5
    maxForce = 800 #N
    
    print("")
    print("######################################################################################")
    print("")
    
    #Linear elastic solver, gradually increments force using half sine wave
    for i in range(timeSteps):
        
        # Load vector
        load = maxForce*sin((i/(timeSteps-1))*(pi/2))
        f = Constant((0, -load, 0))
        
        # Weak variational form
        u = TrialFunction(P)
        v = TestFunction(P)
        a = stiffness(pseudoDensityFunction)*inner(sigma(u), epsilon(v))*dx
        L = inner(f,v)*ds(2)
        
        # Solve linear elastic equation
        uh = Function(P)
        solve(a == L, uh, bcs=bc)
        
        # Von-mises stress
        deviatoricStress = sigma(uh) - (1./3)*tr(sigma(uh))*Identity(3)
        vonMises = sqrt(3./2*inner(deviatoricStress, deviatoricStress))
        stresses = project(vonMises, B)
        
        print("Tumour radius (m)            : ", '%s' % float('%.4g' % tumourRadius))
        print("Load (N)                     : ", '%s' % float('%.4g' % load))
        print("Maximum von-mises stress (N) : ", '%s' % float('%.4g' % max(stresses.vector().get_local())))
        print("")
     
    return uh, pseudoDensityFunction, stresses


#Visualisation

#Compute the displacement for a range of tumour sizes
tumourMininumRadius = 0
tumourMaximumRadius = 0.01
steps = 2

t0 = time.time()

for i in range(steps):
    tumourRadius = i*(tumourMaximumRadius - tumourMininumRadius) / (steps - 1)
    uh, pseudoDensityFunction, stresses = forwardProblem(tumourRadius)
    
t1 = time.time()    
print("Time: ", round(t1-t0, 3))
    
#Visualisation
File('Firedrake/finalDisplacement.pvd').write(uh)
File('Firedrake/finalPseudoDensity.pvd').write(pseudoDensityFunction)
File('Firedrake/finalStress.pvd').write(stresses)







   
