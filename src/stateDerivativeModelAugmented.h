#ifndef TUDATBUNDLE_STATEDERIVATIVEMODELAUGMENTED_H
#define TUDATBUNDLE_STATEDERIVATIVEMODELAUGMENTED_H




#include "Tudat/Astrodynamics/Propagators/stateDerivativeCircularRestrictedThreeBodyProblem.h"
#include "computeManifoldsAugmented.h"

double determinePointingSign(const std::string thrustPointing);

Eigen::MatrixXd computeStateDerivativeAugmented( const double time, const Eigen::MatrixXd& cartesianState);

#endif  // TUDATBUNDLE_STATEDERIVATIVEMODELAUGMENTED_H
