#ifndef TUDATBUNDLE_CREATEINITIALCONDITIONS_H
#define TUDATBUNDLE_CREATEINITIALCONDITIONS_H



#include <string>
#include <vector>

#include <Eigen/Core>

#include "Tudat/Basics/basicTypedefs.h"

void appendResultsVector(
        const double jacobiEnergy, const double orbitalPeriod, const Eigen::VectorXd& initialStateVector,
        const Eigen::VectorXd& stateVectorInclSTM, std::vector< Eigen::VectorXd >& initialConditions );

void appendDifferentialCorrectionResultsVector(
        const double jacobiEnergyHalfPeriod,  const Eigen::VectorXd& differentialCorrectionResult,
        std::vector< Eigen::VectorXd >& differentialCorrections );

Eigen::Vector7d getInitialStateVectorGuess( const int librationPointNr, const std::string& orbitType, const int guessIteration );

Eigen::MatrixXd getCorrectedInitialState( const Eigen::Vector6d& initialStateGuess, const double orbitalPeriod, const int orbitNumber,
                                          const int librationPointNr, std::string orbitType, const double massParameter,
                                          std::vector< Eigen::VectorXd >& initialConditions,
                                          std::vector< Eigen::VectorXd >& differentialCorrections,
                                          const double maxPositionDeviationFromPeriodicOrbit = 1.0e-12, const double maxVelocityDeviationFromPeriodicOrbit = 1.0e-12 );

void writeFinalResultsToFiles( const int librationPointNr, const std::string orbitType,
                               std::vector< Eigen::VectorXd > initialConditions,
                               std::vector< Eigen::VectorXd > differentialCorrections );

bool checkTermination( const std::vector< Eigen::VectorXd >& differentialCorrections,
                       const Eigen::MatrixXd& stateVectorInclSTM, const std::string orbitType, const int librationPointNr,
                       const double maxEigenvalueDeviation = 1.0e-3 );

void createInitialConditions( const int librationPointNr, const std::string& orbitType,
                              const double massParameter = tudat::gravitation::circular_restricted_three_body_problem::computeMassParameter(
            tudat::celestial_body_constants::EARTH_GRAVITATIONAL_PARAMETER,
            tudat::celestial_body_constants::MOON_GRAVITATIONAL_PARAMETER ),
                              const double maxPositionDeviationFromPeriodicOrbit = 1.0e-12, const double maxVelocityDeviationFromPeriodicOrbit = 1.0e-12,
                              const double maxEigenvalueDeviation = 1.0e-3 );


#endif  // TUDATBUNDLE_CREATEINITIALCONDITIONS_H
