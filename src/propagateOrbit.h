#ifndef TUDATBUNDLE_PROPAGATEORBIT_H
#define TUDATBUNDLE_PROPAGATEORBIT_H

#include <map>

#include <Eigen/Core>

#include "Tudat/Basics/basicTypedefs.h"

Eigen::MatrixXd getFullInitialState( const Eigen::Vector6d& initialState );

void writeStateHistoryToFile(
        const std::map< double, Eigen::Vector6d >& stateHistory,
        const int orbitId, const std::string orbitType, const int librationPointNr,
        const int saveEveryNthIntegrationStep, const bool completeInitialConditionsHaloFamily );

std::pair< Eigen::MatrixXd, double > propagateOrbit(
        const Eigen::MatrixXd& stateVectorInclSTM, double massParameter, double currentTime,
        int direction, double initialStepSize = 1.0E-5, double maximumStepSize = 1.0E-4 );

std::pair< Eigen::MatrixXd, double >  propagateOrbitToFinalCondition(
        const Eigen::MatrixXd fullInitialState, const double massParameter, const double finalTime, int direction,
        std::map< double, Eigen::Vector6d >& stateHistory, const int saveFrequency = -1, const double initialTime = 0.0 );

std::pair< Eigen::MatrixXd, double >  propagateOrbitToFinalSpatialCondition(
        const Eigen::MatrixXd fullInitialState, const double massParameter, const int stateIndex, int direction,
        std::map< double, Eigen::Vector6d >& stateHistory, const int saveFrequency, const double initialTime );


std::pair< Eigen::MatrixXd, double >  propagateOrbitWithStateTransitionMatrixToFinalCondition(
        const Eigen::MatrixXd fullInitialState, const double massParameter, const double finalTime, int direction,
        std::map< double, Eigen::MatrixXd >& stateTransitionMatrixHistory, const int saveFrequency = -1, const double initialTime = 0.0);

#endif  // TUDATBUNDLE_PROPAGATEORBIT_H
