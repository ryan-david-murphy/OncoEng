#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 11:23:40 2022

@author: ryanmurphy
"""

# import libraries
import numpy as np
import time
import ufl

from dolfinx import fem
from dolfinx.io import VTKFile, XDMFFile
from dolfinx.io.gmshio import read_from_msh
from mpi4py import MPI
from petsc4py.PETSc import ScalarType

# Load mesh
mesh, cell_tags, facet_tags = read_from_msh("mesh/vertebraeFEniCSx.msh", MPI.COMM_WORLD, 0, gdim=3)

# Define both scalar (A) and vector function spaces (P)
A = fem.FunctionSpace(mesh, ("CG", 1))
B = fem.FunctionSpace(mesh, ("DG", 0))
P = fem.VectorFunctionSpace(mesh, ("CG", 1))

# Define integration domains
dx = ufl.Measure("dx", domain=mesh, subdomain_data=cell_tags)
ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)

# Material properties
E = 10e9
nu = 0.3

# Lamé parameters
lmbda = fem.Constant(mesh, E*nu/((1+nu)*(1-2*nu)))
mu = fem.Constant(mesh, E/2/(1+nu))

#Constitutive relations
def epsilon(u):
    return ufl.sym(ufl.grad(u))

def sigma(u):
    return lmbda * ufl.nabla_div(u) * ufl.Identity(u.geometric_dimension()) + 2*mu*epsilon(u)

# Pseudo-density functional
def stiffness(pseudoDensity):
    eps = 1e-3
    return eps + (1 - eps)*pseudoDensity

# FEA    
def forwardProblem(tumourRadius):
    
    #Define scalar functional
    # 0 - tumour
    # 1 - bone    
    pseudoDensityFunction = fem.Function(A)
    
    #Tumour center
    xCentre = 0.0
    yCentre = 1.175
    zCentre = 0.0
    pseudoDensityFunction.interpolate(lambda x: 0.5 * (np.tanh(1e8*(pow(x[0] - xCentre,2) + pow(x[1] - yCentre,2) + pow(x[2] - zCentre,2) - pow(tumourRadius,2))) + 1))    
    
    #Define boundary conditions    
    bcFacets = facet_tags.find(3)
    bcsDofs = fem.locate_dofs_topological(P, mesh.topology.dim-1, bcFacets)
    bc = fem.dirichletbc(ScalarType((0, 0, 0)), bcsDofs, P)
    
    #Half sine wave load sweep
    timeSteps = 5
    maxForce = 800 #N
    
    print("")
    print("######################################################################################")
    print("")
    
    #Linear elastic solver, gradually increments force using half sine wave
    for i in range(timeSteps):
        
        # Load vector
        load = maxForce*np.sin((i/(timeSteps-1))*(np.pi/2))
        forceVector = fem.Constant(mesh, ScalarType((0, -load, 0)))
        
        # Weak variational form
        u = ufl.TrialFunction(P)
        v = ufl.TestFunction(P)
        a = stiffness(pseudoDensityFunction)*ufl.inner(sigma(u), epsilon(v)) * ufl.dx
        L = ufl.dot(forceVector, v) * ds(2)
        
        # Solve linear elastic equation
        problem = fem.petsc.LinearProblem(a, L, bcs=[bc])
        uh = problem.solve()
        
        # Von-mises stress
        s = sigma(uh) - 1./3*ufl.tr(sigma(uh))*ufl.Identity(uh.geometric_dimension())
        von_Mises = ufl.sqrt(3./2*ufl.inner(s, s))
        stress_expr = fem.Expression(von_Mises, B.element.interpolation_points())
        stresses = fem.Function(B)
        stresses.interpolate(stress_expr)
        
        print("Tumour radius (m)            : ", '%s' % float('%.4g' % tumourRadius))
        print("Load (N)                     : ", '%s' % float('%.4g' % load))
        print("Maximum von-mises stress (N) : ", '%s' % float('%.4g' % max(stresses.vector.array)))
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
    
with VTKFile(mesh.comm, "FEniCSX/finalDisplacement.pvd", "w") as file:
    file.write_mesh(mesh)
    file.write_function([uh._cpp_object])
    
with VTKFile(mesh.comm, "FEniCSX/inalPseudoDensity.pvd", "w") as file:
    file.write_mesh(mesh)
    file.write_function([pseudoDensityFunction._cpp_object])
    
with VTKFile(mesh.comm, "FEniCSX/finalStress.pvd", "w") as file:
    file.write_mesh(mesh)
    file.write_function([stresses._cpp_object])
    








   
