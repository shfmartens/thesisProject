#ifndef TUDATBUNDLE_PROPAGATEORBITAUGMENTED_H
#define TUDATBUNDLE_PROPAGATEORBITAUGMENTED_H

#include <map>

#include <Eigen/Core>

#include "Tudat/Basics/basicTypedefs.h"

Eigen::MatrixXd getFullAugmentedInitialState( const Eigen::Vector6d& initialState, const Eigen::Vector1d& initialMass);

std::pair< Eigen::MatrixXd, double > propagateOrbitAugmented(
        const Eigen::MatrixXd& stateVectorInclSTM, double massParameter, double currentTime,
        int direction, std::string spacecraftName = "deepSpace",  std::string thrustPointing = "left", double initialStepSize = 1.0E-5, double maximumStepSize = 1.0E-4);

#endif  // TUDATBUNDLE_PROPAGATEORBITAUGMENTED_H
