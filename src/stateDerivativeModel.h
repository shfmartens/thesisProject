#ifndef TUDATBUNDLE_STATEDERIVATIVEMODEL_H
#define TUDATBUNDLE_STATEDERIVATIVEMODEL_H




#include "Tudat/Astrodynamics/Propagators/stateDerivativeCircularRestrictedThreeBodyProblem.h"

#include "stateDerivativeModel.cpp"


extern double massParameter;

Eigen::VectorXd computeStateDerivative( const double time, const Eigen::VectorXd cartesianState );



#endif  // TUDATBUNDLE_STATEDERIVATIVEMODEL_H
