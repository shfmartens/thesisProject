# thesisProject

This repository contains the code used for the generation and verification of periodic libration point orbits and their associated manifolds. For an extensive analysis of these structures, the reader is referred to the electronic version of my thesis on the [TU Delft education repository](https://repository.tudelft.nl/islandora/search/?collection=education) under the author Koen Langemeijer. 

## Getting Started

The set of functions require the [Tudat Bundle](https://github.com/Tudat/tudatBundle) to be installed. This repository resides in the tudatApplications folder of the tudatBundle.

### Functions

This project contains the source files for the generation of a family of periodic libration point orbits, using:
1. An initial guess derived from Richardson [richardsonThirdOrderApproximation.cpp]
2. Propagation of an initial state to an estimated (half)-period [propagateOrbit.cpp] by integrating the state derivative [stateDerivativeModel.cpp]
3. Refinement of the initial conditions until the required periodicity [applyDifferentialCorrection.cpp]

In addition, the precomputed initial conditions can be supplied to produce the associated hyperbolic manifolds [computeManifold.cpp].
