#ifndef TUDATBUNDLE_PROPAGATEORBITAUGMENTED_H
#define TUDATBUNDLE_PROPAGATEORBITAUGMENTED_H

#include <map>

#include <Eigen/Core>

#include "Tudat/Basics/basicTypedefs.h"

Eigen::MatrixXd getFullInitialStateAugmented( const Eigen::VectorXd& initialState );

void writeStateHistoryAndStateVectorsToFile ( const std::map< double, Eigen::VectorXd >& stateHistory, const std::string orbitType, const Eigen::VectorXd stateVectors, const Eigen::VectorXd deviationVector, const Eigen::VectorXd deviationVectorFull,
                                              const int numberOfIterations, const int correctionLevel );

void writeStateHistoryToFileAugmented(
        const std::map< double, Eigen::VectorXd >& stateHistory, const double accelerationMagnitude, const double accelerationAngle, const double accelerationAngle2, const double initialHamiltonian,
        const int orbitId, const int librationPointNr, const std::string& orbitType,
        const int saveEveryNthIntegrationStep, const bool completeInitialConditionsHaloFamily );

std::pair< Eigen::MatrixXd, double > propagateOrbitAugmented(
        const Eigen::MatrixXd& stateVectorInclSTM, double massParameter, double currentTime,
        int direction, double initialStepSize = 1.0E-5, double maximumStepSize = 1.0E-4 );

std::pair< Eigen::MatrixXd, double >  propagateOrbitAugmentedToFinalCondition(
        const Eigen::MatrixXd fullInitialState, const double massParameter, const double finalTime, int direction,
        std::map< double, Eigen::VectorXd >& stateHistory, const int saveFrequency = -1, const double initialTime = 0.0 );

std::pair< Eigen::MatrixXd, double >  propagateOrbitAugmentedToFinalThetaCondition(
        const Eigen::MatrixXd fullInitialState, const double massParameter, int direction,
        std::map< double, Eigen::VectorXd >& stateHistoryMinimized, const int saveFrequency = -1, const double initialTime = 0.0 );

std::pair< Eigen::MatrixXd, double >  propagateOrbitAugmentedToFinalSpatialCondition(
        const Eigen::MatrixXd fullInitialState, const double massParameter, const int stateIndex, int direction,
        std::map< double, Eigen::VectorXd >& stateHistoryMinimized, const int saveFrequency, const double initialTime );

std::pair< Eigen::MatrixXd, double >  propagateOrbitAugmentedWithStateTransitionMatrixToFinalCondition(
        const Eigen::MatrixXd fullInitialState, const double massParameter, const double finalTime, int direction,
        std::map< double, Eigen::MatrixXd >& stateTransitionMatrixHistory, const int saveFrequency = -1, const double initialTime = 0.0);

#endif  // TUDATBUNDLE_PROPAGATEORBITAUGMENTED_H
