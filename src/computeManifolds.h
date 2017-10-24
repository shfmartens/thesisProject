#ifndef TUDATBUNDLE_COMPUTEMANIFOLDS_H
#define TUDATBUNDLE_COMPUTEMANIFOLDS_H


#include <string>
#include <vector>

#include <Eigen/Core>

#include "Tudat/Basics/basicTypedefs.h"

void determineStableUnstableEigenvectors( Eigen::MatrixXd& monodromyMatrix, Eigen::Vector6d& stableEigenvector,
                                          Eigen::Vector6d& unstableEigenvector,
                                          const double maxEigenvalueDeviation = 1.0E-3 );

double determineEigenvectorSign( Eigen::Vector6d& eigenvector );

bool checkJacobiOnManifoldOutsideBounds( Eigen::MatrixXd& stateVectorInclSTM, double& referenceJacobiEnergy,
                                         const double massParameter = tudat::gravitation::circular_restricted_three_body_problem::computeMassParameter(tudat::celestial_body_constants::EARTH_GRAVITATIONAL_PARAMETER, tudat::celestial_body_constants::MOON_GRAVITATIONAL_PARAMETER ),
                                         const double maxJacobiEnergyDeviation = 1.0e-12 );

void reduceOvershootAtPoincareSectionU1U4( std::pair< Eigen::MatrixXd, double >& stateVectorInclSTMAndTime,
                                           std::pair< Eigen::MatrixXd, double >& previousStateVectorInclSTMAndTime,
                                           Eigen::MatrixXd& stateVectorInclSTM, double& currentTime, double& ySign,
                                           int& integrationDirection, const double& massParameter = tudat::gravitation::circular_restricted_three_body_problem::computeMassParameter(tudat::celestial_body_constants::EARTH_GRAVITATIONAL_PARAMETER, tudat::celestial_body_constants::MOON_GRAVITATIONAL_PARAMETER ) );

void reduceOvershootAtPoincareSectionU2U3( std::pair< Eigen::MatrixXd, double >& stateVectorInclSTMAndTime,
                                           std::pair< Eigen::MatrixXd, double >& previousStateVectorInclSTMAndTime,
                                           Eigen::MatrixXd& stateVectorInclSTM, double& currentTime, double& xDiffSign,
                                           int& integrationDirection, const double& massParameter = tudat::gravitation::circular_restricted_three_body_problem::computeMassParameter(tudat::celestial_body_constants::EARTH_GRAVITATIONAL_PARAMETER, tudat::celestial_body_constants::MOON_GRAVITATIONAL_PARAMETER ) );

void writeManifoldStateHistoryToFile( std::map< int, std::map< int, std::map< double, Eigen::Vector6d > > >& manifoldStateHistory,
                                      const int& orbitNumber, const int& librationPointNr, const std::string& orbitType );

void writeEigenvectorStateHistoryToFile( std::map< int, std::map< int, std::pair< Eigen::Vector6d, Eigen::Vector6d > > >& eigenvectorStateHistory,
                                         const int& orbitNumber, const int& librationPointNr,
                                         const std::string& orbitType );

void computeManifolds( const Eigen::Vector6d initialStateVector, const double orbitalPeriod, const int orbitNumber,
                       const int librationPointNr, const std::string orbitType,
                       const double massParameter = tudat::gravitation::circular_restricted_three_body_problem::computeMassParameter(tudat::celestial_body_constants::EARTH_GRAVITATIONAL_PARAMETER, tudat::celestial_body_constants::MOON_GRAVITATIONAL_PARAMETER ),
                       const double eigenvectorDisplacementFromOrbit = 1.0E-6,
                       const int numberOfTrajectoriesPerManifold = 100, const int saveFrequency = 1000,
                       const bool saveEigenvectors = true,
                       const double maximumIntegrationTimeManifoldTrajectories = 50.0,
                       const double maxEigenvalueDeviation = 1.0E-3 );

#endif  // TUDATBUNDLE_COMPUTEMANIFOLDS_H
