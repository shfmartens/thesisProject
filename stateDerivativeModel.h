#ifndef STATEDERIVATIVEMODEL_H
#define STATEDERIVATIVEMODEL_H

// Include-statements.
#include "stateDerivativeModel.cpp"
#include "Tudat/Astrodynamics/Gravitation/stateDerivativeCircularRestrictedThreeBodyProblem.h"

// Declare mass parameter.
extern double massParameter;
extern Eigen::Vector3d thrustVector;
extern double thrustAcceleration;

// Declare function.
Eigen::VectorXd computeStateDerivative( const double time, const Eigen::VectorXd cartesianState);

#endif // STATEDERIVATIVEMODEL_H
