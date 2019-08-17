#ifndef TUDATBUNDLE_STATEDERIVATIVEMODELAUGMENTEDTLT_H
#define TUDATBUNDLE_STATEDERIVATIVEMODELAUGMENTEDTLT_H




#include "Tudat/Astrodynamics/Propagators/stateDerivativeCircularRestrictedThreeBodyProblem.h"

Eigen::MatrixXd computeStateDerivativeAugmentedTLT( const double time, const Eigen::MatrixXd& cartesianState );


#endif  // TUDATBUNDLE_STATEDERIVATIVEMODELAUGMENTEDTLT_H
