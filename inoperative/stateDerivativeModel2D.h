#ifndef STATEDERIVATIVEMODEL_H
#define STATEDERIVATIVEMODEL_H

// Include-statements.
#include "stateDerivativeModel2D.cpp"
#include "Tudat/Astrodynamics/Gravitation/stateDerivativeCircularRestrictedThreeBodyProblem.h"

// Declare function.
Eigen::VectorXd computeStateDerivative( const double time, const Eigen::VectorXd cartesianState);

#endif // STATEDERIVATIVEMODEL_H
