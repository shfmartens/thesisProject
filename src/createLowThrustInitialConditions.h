#ifndef TUDATBUNDLE_CREATELOWTHRUSTINITIALCONDITIONS_H
#define TUDATBUNDLE_CREATELOWTHRUSTINITIALCONDITIONS_H



#include <string>
#include <vector>

#include <Eigen/Core>

#include "Tudat/Basics/basicTypedefs.h"

#include <boost/bind.hpp>
#include <boost/function.hpp>

void writeFinalResultsToFiles( const int librationPointNr, const double accelerationMagnitude, const double accelerationAngle,
                               std::vector< Eigen::VectorXd > initialConditions,
                               std::vector< Eigen::VectorXd > differentialCorrections );

void appendResultsVectorAugmented(const double hamiltonian, const double orbitalPeriod, const Eigen::VectorXd& initialStateVector,
        const Eigen::MatrixXd& stateVectorInclSTM, std::vector< Eigen::VectorXd >& initialConditions );

void appendDifferentialCorrectionResultsVectorAugmented(
        const double hamiltonianHalfPeriod,  const Eigen::VectorXd& differentialCorrectionResult,
        std::vector< Eigen::VectorXd >& differentialCorrections );

double getDefaultArcLengthAugmented(
        const double distanceIncrement,
        const Eigen::Vector6d& currentState, const double periodIncrement, const int continuationIndex );

double computeHamiltonian ( const double massParameter, const Eigen::VectorXd stateVector );

Eigen::MatrixXd getCorrectedAugmentedInitialState( const Eigen::VectorXd& initialStateGuess, const double orbitalPeriod, const int orbitNumber,
                                          const int librationPointNr, const double massParameter,
                                          std::vector< Eigen::VectorXd >& initialConditions,
                                          std::vector< Eigen::VectorXd >& differentialCorrections,
                                          const double maxPositionDeviationFromPeriodicOrbit = 1.0e-11, const double maxVelocityDeviationFromPeriodicOrbit = 1.0e-11, const double maxPeriodDeviationFromPeriodicOrbit = 1.0e-08 );

Eigen::VectorXd getEarthMoonInitialGuessParameters ( const int librationPointNr, const std::string& orbitType, const double accelerationMagnitude, const double accelerationAngle, const double accelerationAngle2, const int continuationIndex, const int guessIteration );

Eigen::VectorXd getLowThrustInitialStateVectorGuess( const int librationPointNr, const std::string& orbitType, const double accelerationMagnitude, const double accelerationAngle, const double accelerationAngle2, const double initialMass, const int continuationIndex, const int guessIteration,
                                            const boost::function< Eigen::VectorXd( const int librationPointNr, const std::string& orbitType, const double accelerationMagnitude, const double accelerationAngle, const double accelerationAngle2, const int continuationIndex, const int guessIteration ) > getInitialGuessParameters = getEarthMoonInitialGuessParameters );

double getDefaultArcLengthAugmented(
        const double distanceIncrement,
        const Eigen::VectorXd& currentState,  const int continuationIndex );

bool checkTerminationAugmented( const std::vector< Eigen::VectorXd >& differentialCorrections,
                       const Eigen::MatrixXd& stateVectorInclSTM, const std::string orbitType, const int librationPointNr,
                       const double maxEigenvalueDeviation = 1.0e-3 );

void createLowThrustInitialConditions( const int librationPointNr, const std::string& orbitType, const int continuationIndex, const double accelerationMagnitude, const double accelerationAngle,
                                       const double accelerationAngle2, const double initialMass,
                              const double massParameter = tudat::gravitation::circular_restricted_three_body_problem::computeMassParameter(
            tudat::celestial_body_constants::EARTH_GRAVITATIONAL_PARAMETER,
            tudat::celestial_body_constants::MOON_GRAVITATIONAL_PARAMETER ),
                              const double maxPositionDeviationFromPeriodicOrbit = 1.0e-10, const double maxVelocityDeviationFromPeriodicOrbit = 1.0e-10, const double maxPeriodDeviationFromPeriodicOrbit = 1.0e-08,
                              const double maxEigenvalueDeviation = 1.0e-3,
                              const boost::function< double( const Eigen::VectorXd&, const int ) > pseudoArcLengthFunctionAugmented =
        boost::bind( &getDefaultArcLengthAugmented, 1.0E-4, _1, _2 ) );


#endif  // TUDATBUNDLE_CREATELOWTHRUSTINITIALCONDITIONS_H
