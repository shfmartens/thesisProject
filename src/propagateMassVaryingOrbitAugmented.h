#ifndef TUDATBUNDLE_PROPAGATEMASSVARYINGORBITAUGMENTED_H
#define TUDATBUNDLE_PROPAGATEMASSVARYINGORBITAUGMENTED_H

#include <map>

#include <Eigen/Core>

#include "Tudat/Basics/basicTypedefs.h"


std::pair< Eigen::MatrixXd, double > propagateMassVaryingOrbitAugmented(
        const Eigen::MatrixXd& stateVectorInclSTM, double massParameter, double currentTime,
        int direction, double initialStepSize = 1.0E-5, double maximumStepSize = 1.0E-4 );

std::pair< Eigen::MatrixXd, double >  propagateMassVaryingOrbitAugmentedToFinalCondition(
        const Eigen::MatrixXd fullInitialState, const double massParameter, const double finalTime, int direction,
        std::map< double, Eigen::VectorXd >& stateHistory, const int saveFrequency = -1, const double initialTime = 0.0 );


#endif  // TUDATBUNDLE_PROPAGATEMASSVARYINGORBITAUGMENTED_H
