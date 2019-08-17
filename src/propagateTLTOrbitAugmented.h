#ifndef TUDATBUNDLE_PROPAGATETLTORBITAUGMENTED_H
#define TUDATBUNDLE_PROPAGATETLTORBITAUGMENTED_H

#include <map>

#include <Eigen/Core>

#include "Tudat/Basics/basicTypedefs.h"


std::pair< Eigen::MatrixXd, double > propagateTLTOrbitAugmented(
        const Eigen::MatrixXd& stateVectorInclSTM, double massParameter, double currentTime,
        int direction, double initialStepSize = 1.0E-5, double maximumStepSize = 1.0E-4 );

std::pair< Eigen::MatrixXd, double >  propagateTLTOrbitAugmentedToFinalCondition(
        const Eigen::MatrixXd fullInitialState, const double massParameter, const double finalTime, int direction,
        std::map< double, Eigen::VectorXd >& stateHistory, const int saveFrequency = -1, const double initialTime = 0.0 );


#endif  // TUDATBUNDLE_PROPAGATETLTORBITAUGMENTED_H
