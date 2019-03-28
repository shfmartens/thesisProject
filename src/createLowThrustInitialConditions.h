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
        const Eigen::Vector6d& currentState );

double computeHamiltonian ( const double massParameter, const Eigen::VectorXd stateVector );

Eigen::MatrixXd getCorrectedInitialState( const Eigen::Vector6d& initialStateGuess, const double orbitalPeriod, const int orbitNumber,
                                          const int librationPointNr, const double accelerationMagnitude, const double accelerationAngle,
                                          const double accelerationAngle2, const double initialMass, const double massParameter,
                                          std::vector< Eigen::VectorXd >& initialConditions,
                                          std::vector< Eigen::VectorXd >& differentialCorrections,
                                          const double maxPositionDeviationFromPeriodicOrbit = 1.0e-12, const double maxVelocityDeviationFromPeriodicOrbit = 1.0e-12 );


void createLowThrustInitialConditions( const int librationPointNr, const double accelerationMagnitude, const double accelerationAngle,
                                       const double accelerationAngle2, const double initialMass,
                              const double massParameter = tudat::gravitation::circular_restricted_three_body_problem::computeMassParameter(
            tudat::celestial_body_constants::EARTH_GRAVITATIONAL_PARAMETER,
            tudat::celestial_body_constants::MOON_GRAVITATIONAL_PARAMETER ),
                              const double maxPositionDeviationFromPeriodicOrbit = 1.0e-12, const double maxVelocityDeviationFromPeriodicOrbit = 1.0e-12,
                              const double maxEigenvalueDeviation = 1.0e-3,
                              const boost::function< double( const Eigen::Vector6d& ) > pseudoArcLengthFunction =
        boost::bind( &getDefaultArcLengthAugmented, 1.0E-4, _1 ) );


#endif  // TUDATBUNDLE_CREATELOWTHRUSTINITIALCONDITIONS_H
