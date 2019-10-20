#ifndef TUDATBUNDLE_CREATELOWTHRUSTINITIALCONDITIONS_H
#define TUDATBUNDLE_CREATELOWTHRUSTINITIALCONDITIONS_H



#include <string>
#include <vector>

#include <Eigen/Core>

#include "Tudat/Basics/basicTypedefs.h"

#include <boost/bind.hpp>
#include <boost/function.hpp>

void writeFinalResultsToFiles( const int librationPointNr, const std::string& orbitType, const int continuationIndex, const double accelerationMagnitude, const double accelerationAngle, const double accelerationAngle2, const double familyHamiltonian, const int numberOfPatchPoints,
                               std::vector< Eigen::VectorXd > initialConditions,
                               std::vector< Eigen::VectorXd > differentialCorrections, std::vector< Eigen::VectorXd > statesContinuation );

void appendResultsVectorAugmented(const double hamiltonian, const double orbitalPeriod, const Eigen::VectorXd& initialStateVector,
        const Eigen::MatrixXd& stateVectorInclSTM, std::vector< Eigen::VectorXd >& initialConditions );

void appendDifferentialCorrectionResultsVectorAugmented(
        const double hamiltonianHalfPeriod,  const Eigen::VectorXd& differentialCorrectionResult,
        std::vector< Eigen::VectorXd >& differentialCorrections, const Eigen::VectorXd deviationsNorms );

void appendContinuationStatesVectorAugmented(const int orbitNumber, const int numberOfPatchPoints, const double hamiltonianInitialCondition, const double orbitalPeriod,
                                             const Eigen::VectorXd& differentialCorrectionResult, std::vector< Eigen::VectorXd >& statesContinuation);
double getDefaultArcLengthAugmented(
        const double distanceIncrement,
        const Eigen::Vector6d& currentState, const double periodIncrement, const int continuationIndex );

double computeHamiltonian ( const double massParameter, const Eigen::VectorXd stateVector );

Eigen::MatrixXd getCorrectedAugmentedInitialState( const Eigen::VectorXd& initialStateGuess, const double targetHamiltonian, const int orbitNumber,
                                          const int librationPointNr, const std::string& orbitType, const double massParameter, const int numberOfPatchPoints, const int numberOfCollocationPoints,  const bool hamiltonianConstraint,
                                          std::vector< Eigen::VectorXd >& initialConditions,
                                          std::vector< Eigen::VectorXd >& differentialCorrections,
                                          std::vector< Eigen::VectorXd >& statesContinuation,
                                          const double maxPositionDeviationFromPeriodicOrbit = 1.0e-12, const double maxVelocityDeviationFromPeriodicOrbit = 5.0e-12, const double maxPeriodDeviationFromPeriodicOrbit = 1.0e-12);

Eigen::VectorXd getEarthMoonInitialGuessParameters ( const int librationPointNr, const std::string& orbitType, const double accelerationMagnitude, const double accelerationAngle, const double accelerationAngle2, const int continuationIndex, const int guessIteration );

Eigen::VectorXd getLowThrustInitialStateVectorGuess( const int librationPointNr, const double ySign, const std::string& orbitType, const double accelerationMagnitude, const double accelerationAngle, const double accelerationAngle2, const double initialMass, const int continuationIndex, const int numberOfPatchPoints, const int guessIteration,
                                            const boost::function< Eigen::VectorXd( const int librationPointNr, const std::string& orbitType, const double accelerationMagnitude, const double accelerationAngle, const double accelerationAngle2, const int continuationIndex, const int guessIteration ) > getInitialGuessParameters = getEarthMoonInitialGuessParameters );

double getDefaultArcLengthAugmented(
        const double distanceIncrement,
        const Eigen::VectorXd& currentState,  const int continuationIndex  );

bool checkTerminationAugmented( const std::vector< Eigen::VectorXd >& differentialCorrections,
                       const Eigen::MatrixXd& stateVectorInclSTM, const std::string orbitType, const int librationPointNr,
                       const double maxEigenvalueDeviation = 1.0e-3 );

Eigen::MatrixXd getCollocatedAugmentedInitialState( const Eigen::VectorXd& initialStateGuess, const int orbitNumber,
                                          const int librationPointNr, const std::string& orbitType, const int continuationIndex, const Eigen::VectorXd previousDesignVector, bool& continuationDirectionReversed, const double massParameter, const int numberOfPatchPoints, int& numberOfCollocationPoints,
                                          std::vector< Eigen::VectorXd >& initialConditions,
                                          std::vector< Eigen::VectorXd >& differentialCorrections,
                                          std::vector< Eigen::VectorXd >& statesContinuation,
                                                   const double maxPositionDeviationFromPeriodicOrbit, double maxVelocityDeviationFromPeriodicOrbit, const double maxPeriodDeviationFromPeriodicOrbit, const bool initialSolutionFromTextFile );

Eigen::VectorXd computeHamiltonianVaryingStateIncrement(const Eigen::VectorXd initialStateVector, const int numberOfCollocationPoints, const double massParameter);

void createLowThrustInitialConditions( const int librationPointNr, const double ySign, const std::string& orbitType, const int continuationIndex, const double accelerationMagnitude, const double accelerationAngle,
                                       const double accelerationAngle2, const double initialMass, const double familyHamiltonian, const bool startContinuationFromTextFile,
                              const double massParameter = tudat::gravitation::circular_restricted_three_body_problem::computeMassParameter(
            tudat::celestial_body_constants::EARTH_GRAVITATIONAL_PARAMETER,
            tudat::celestial_body_constants::MOON_GRAVITATIONAL_PARAMETER ),
                                       const int numberOfPatchPoints = 5, //19 daarna
                                       const int initialNumberOfCollocationPoints = 5,
                              const double maxPositionDeviationFromPeriodicOrbit = 1.0e-12, const double maxVelocityDeviationFromPeriodicOrbit = 5.0e-12, const double maxPeriodDeviationFromPeriodicOrbit = 1.0e-12,
                              const double maxEigenvalueDeviation = 1.0e-3,
                              const boost::function< double( const Eigen::VectorXd&, const int ) > pseudoArcLengthFunctionAugmented =
        boost::bind( &getDefaultArcLengthAugmented, 1.0E-4, _1, _2 ) );


#endif  // TUDATBUNDLE_CREATELOWTHRUSTINITIALCONDITIONS_H
