#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 11:23:40 2022

@author: ryanmurphy
"""

# import libraries
import time

from fenics import *
from ufl import nabla_div
from mpi4py import MPI as mpi

# Load mesh
mesh = Mesh()
TMP_Mesh = HDF5File(mpi.COMM_WORLD, 'mesh/vertebraeFEniCS.hdf5', 'r')
TMP_Mesh.read(mesh, "Mesh", False)

# Load Boundary Definitions
subBoundaries = MeshFunction("size_t", mesh, 2)
tmpSubBoundaries = HDF5File(mpi.COMM_WORLD, 'mesh/vertebraeFEniCS.hdf5', 'r')
tmpSubBoundaries.read(subBoundaries, 'mesh/facet_region')
tmpSubBoundaries.close()

# Define integration domains
ds = Measure("ds")(subdomain_data=subBoundaries, domain=mesh)

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

#Constitutive relations
def epsilon(v):
    return sym(nabla_grad(v))

def sigma(u):
    return lmbda*nabla_div(u)*Identity(mesh.topology().dim()) + 2*mu*epsilon(u)   

# Pseudo-density functional
def stiffness(pseudoDensity):
    eps = 1e-3
    return eps + (1 - eps)*pseudoDensity

# FEA 
def forwardProblem(tumourRadius):
    
    #Define scalar functional
    # 0 - tumour
    # 1 - bone    
    pseudoDensityExpression = Expression('0.5 * (tanh(1e8*(pow(x[0] - xCentre,2) + pow(x[1] - yCentre,2) + pow(x[2] - zCentre,2) - pow(tumourRadius,2))) + 1)', \
                                          xCentre = Constant(0.0), \
                                          yCentre = Constant(1.175), \
                                          zCentre = Constant(0.0), \
                                          tumourRadius=Constant(tumourRadius), \
                                          degree=1)
    pseudoDensityFunction = project(pseudoDensityExpression, A)
    
    #Define boundary conditions    
    bc1 = DirichletBC(P.sub(0), 0, subBoundaries, 3)
    bc2 = DirichletBC(P.sub(1), 0, subBoundaries, 3)
    bc3 = DirichletBC(P.sub(2), 0, subBoundaries, 3)
    bc=[bc1,bc2,bc3]
    
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
        solve(a == L, uh, bc, solver_parameters={"linear_solver": "mumps"})
        
        # Von-mises stress
        deviatoricStress = sigma(uh) - (1./3)*tr(sigma(uh))*Identity(3)
        vonMises = sqrt(3./2*inner(deviatoricStress, deviatoricStress))
        stresses = project(vonMises, B)
        
        print("Tumour radius (m)            : ", '%s' % float('%.4g' % tumourRadius))
        print("Load (N)                     : ", '%s' % float('%.4g' % load))
        print("Maximum von-mises stress (N) : ", '%s' % float('%.4g' % max(stresses.vector().get_local())))
        print("")
     
    return uh, pseudoDensityFunction, stresses


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
File('FEniCS/finalDisplacement.pvd') << uh
File('FEniCS/finalPseudoDensity.pvd') << pseudoDensityFunction
File('FEniCS/finalStress.pvd') << stresses





   
