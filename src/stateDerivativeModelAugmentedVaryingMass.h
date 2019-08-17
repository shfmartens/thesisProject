#ifndef TUDATBUNDLE_STATEDERIVATIVEMODELAUGMENTEDVARYINGMASS_H
#define TUDATBUNDLE_STATEDERIVATIVEMODELAUGMENTEDVARYINGMASS_H




#include "Tudat/Astrodynamics/Propagators/stateDerivativeCircularRestrictedThreeBodyProblem.h"

Eigen::MatrixXd computeStateDerivativeAugmentedVaryingMass( const double time, const Eigen::MatrixXd& cartesianState );


#endif  // TUDATBUNDLE_STATEDERIVATIVEMODELAUGMENTEDVARYINGMASS_H
