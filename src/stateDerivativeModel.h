#ifndef TUDATBUNDLE_STATEDERIVATIVEMODEL_H
#define TUDATBUNDLE_STATEDERIVATIVEMODEL_H




#include "Tudat/Astrodynamics/Propagators/stateDerivativeCircularRestrictedThreeBodyProblem.h"

#include "stateDerivativeModel.cpp"


extern double massParameter;

Eigen::MatrixXd computeStateDerivative( const double time, const Eigen::MatrixXd& cartesianState );



#endif  // TUDATBUNDLE_STATEDERIVATIVEMODEL_H
