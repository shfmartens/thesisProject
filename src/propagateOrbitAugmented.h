#ifndef TUDATBUNDLE_PROPAGATEORBITAUGMENTED_H
#define TUDATBUNDLE_PROPAGATEORBITAUGMENTED_H

#include <map>

#include <Eigen/Core>

#include "Tudat/Basics/basicTypedefs.h"

Eigen::MatrixXd getFullAugmentedInitialState( const Eigen::Vector6d& initialState, const Eigen::Vector1d& initialMass, const Eigen::Vector1d& stableInitialMass, int integrationDirection);

std::pair< Eigen::MatrixXd, double > propagateOrbitAugmented(
        const Eigen::MatrixXd& stateVectorInclSTM, double massParameter, double currentTime,
        int direction, std::string spacecraftName = "deepSpace",  std::string thrustPointing = "left", double initialStepSize = 1.0E-6, double maximumStepSize = 1.0E-5);

#endif  // TUDATBUNDLE_PROPAGATEORBITAUGMENTED_H
