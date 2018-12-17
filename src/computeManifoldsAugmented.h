#ifndef TUDATBUNDLE_COMPUTEMANIFOLDSAUGMENTED_H
#define TUDATBUNDLE_COMPUTEMANIFOLDSAUGMENTED_H


#include <string>
#include <vector>
#include <map>

#include <Eigen/Core>
#include "computeManifolds.h"

#include "Tudat/Basics/basicTypedefs.h"
#include "Tudat/Astrodynamics/Gravitation/librationPoint.h"
#include "Tudat/Astrodynamics/BasicAstrodynamics/celestialBodyConstants.h"

Eigen::MatrixXd retrieveSpacecraftProperties( const std::string spacecraftName);

double computeIntegralOfMotion (const Eigen::VectorXd currentStateVector, const std::string spacecraftName, const std::string thrustPointing, const double massParameter = tudat::gravitation::circular_restricted_three_body_problem::computeMassParameter(tudat::celestial_body_constants::EARTH_GRAVITATIONAL_PARAMETER, tudat::celestial_body_constants::MOON_GRAVITATIONAL_PARAMETER ), const double currentTime = 0.0);

bool checkIoMOnManifoldAugmentedOutsideBounds( Eigen::VectorXd currentStateVector, const double referenceIoM,
                                         const double massParameter, const std::string spacecraftName, const std::string thrustPointing, const double currentTime = 0.0, const double maxIoMDeviation = 1.0E-11 );

void reduceOvershootAtPoincareSectionU1U4Augmented( std::pair< Eigen::MatrixXd, double >& stateVectorInclSTMAndTime,
                                           std::pair< Eigen::MatrixXd, double >& previousStateVectorInclSTMAndTime,
                                           Eigen::MatrixXd& stateVectorInclSTM, double& currentTime, double& ySign,
                                           int& integrationDirection, const double& massParameter = tudat::gravitation::circular_restricted_three_body_problem::computeMassParameter(tudat::celestial_body_constants::EARTH_GRAVITATIONAL_PARAMETER, tudat::celestial_body_constants::MOON_GRAVITATIONAL_PARAMETER ), std::string spacecraftName = "deepSpace", std::string thrustPointing = "left");

void reduceOvershootAtPoincareSectionU2U3Augmented( std::pair< Eigen::MatrixXd, double >& stateVectorInclSTMAndTime,
                                           std::pair< Eigen::MatrixXd, double >& previousStateVectorInclSTMAndTime,
                                           Eigen::MatrixXd& stateVectorInclSTM, double& currentTime, double& xDiffSign,
                                           int& integrationDirection, const double& massParameter = tudat::gravitation::circular_restricted_three_body_problem::computeMassParameter(tudat::celestial_body_constants::EARTH_GRAVITATIONAL_PARAMETER, tudat::celestial_body_constants::MOON_GRAVITATIONAL_PARAMETER ), std::string spacecraftName = "deepSpace", std::string thrustPointing = "left" );

void reduceOverShootInitialMass(std::pair< Eigen::MatrixXd, double >& stateVectorInclSTMAndTime,
                                std::pair< Eigen::MatrixXd, double >& previousStateVectorInclSTMAndTime,
                             Eigen::MatrixXd& stateVectorInclSTM, double& currentTime,
                                int& integrationDirection, const double& massParameter = tudat::gravitation::circular_restricted_three_body_problem::computeMassParameter(tudat::celestial_body_constants::EARTH_GRAVITATIONAL_PARAMETER, tudat::celestial_body_constants::MOON_GRAVITATIONAL_PARAMETER ), std::string spacecraftName = "deepSpace", std::string thrustPointing = "left");

void writeAugmentedManifoldStateHistoryToFile( std::map< int, std::map< int, std::map< double, Eigen::Vector7d > > >& manifoldStateHistory,
                                      const int& orbitNumber, const int& librationPointNr, const std::string& orbitType, const std::string spacecraftName, const std::string thrustPointing );

void computeManifoldsAugmented( const Eigen::Vector6d initialStateVector, const double orbitalPeriod, const int orbitNumber,
                       const int librationPointNr, const std::string orbitType, const std::string spacecraftName, const std::string thrustPointing,
                       const double massParameter = tudat::gravitation::circular_restricted_three_body_problem::computeMassParameter(tudat::celestial_body_constants::EARTH_GRAVITATIONAL_PARAMETER, tudat::celestial_body_constants::MOON_GRAVITATIONAL_PARAMETER ),
                       const double eigenvectorDisplacementFromOrbit = 1.0E-6,
                       const int numberOfTrajectoriesPerManifold = 50, const int saveFrequency = 1000,
                       const bool saveEigenvectors = true,
                       const double maximumIntegrationTimeManifoldTrajectories = 50.0,
                       const double maxEigenvalueDeviation = 1.0E-3 );

#endif  // TUDATBUNDLE_COMPUTEMANIFOLDSAUGMENTED_H
