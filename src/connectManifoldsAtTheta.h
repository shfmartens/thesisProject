#ifndef TUDATBUNDLE_CONNECTMANIFOLDSATTHETA_H
#define TUDATBUNDLE_CONNECTMANIFOLDSATTHETA_H


#include <string>
#include <vector>

#include <Eigen/Core>

#include "Tudat/Basics/basicTypedefs.h"

Eigen::VectorXd readInitialConditionsFromFile(const int librationPointNr, const std::string orbitType,
                                              int orbitIdOne, int orbitIdTwo, const double massParameter);

bool checkJacobiOnManifoldOutsideBounds( Eigen::VectorXd currentStateVector, const double referenceJacobiEnergy,
                                         const double massParameter, const double maxJacobiEnergyDeviation = 1.0E-12 );

void computeManifoldStatesAtTheta( std::map< int, std::map< double, Eigen::Vector6d > >& manifoldStateHistory,
                                   Eigen::VectorXd initialStateVector, double orbitalPeriod, int librationPointNr,
                                   const double massParameter, int displacementFromOrbitSign, int integrationTimeDirection,
                                   double thetaStoppingAngle, const int numberOfTrajectoriesPerManifold,
                                   const int saveFrequency = 1000,
                                   const double eigenvectorDisplacementFromOrbit = 1.0E-6,
                                   const double maximumIntegrationTimeManifoldTrajectories = 50.0,
                                   const double maxEigenvalueDeviation = 1.0E-3, const std::string orbitType = "vertical");

Eigen::VectorXd refineOrbitJacobiEnergy( const int librationPointNr, const std::string orbitType, const double desiredJacobiEnergy,
                                         Eigen::VectorXd initialStateVector1, double orbitalPeriod1,
                                         Eigen::VectorXd initialStateVector2, double orbitalPeriod2,
                                         const double massParameter,
                                         const double maxPositionDeviationFromPeriodicOrbit = 1.0E-12,
                                         const double maxVelocityDeviationFromPeriodicOrbit = 1.0E-12,
                                         const double maxJacobiEnergyDeviation = 1.0E-12 );

void writePoincareSectionToFile( std::map< int, std::map< double, Eigen::Vector6d > >& manifoldStateHistory,
                                 int librationPointNr, std::string orbitType, double desiredJacobiEnergy,
                                 int displacementFromOrbitSign, int integrationTimeDirection, double thetaStoppingAngle,
                                 int numberOfTrajectoriesPerManifold );

Eigen::MatrixXd findMinimumImpulseManifoldConnection( std::map< int, std::map< double, Eigen::Vector6d > >& stableManifoldStateHistoryAtTheta,
                                                      std::map< int, std::map< double, Eigen::Vector6d > >& unstableManifoldStateHistoryAtTheta,
                                                      const int numberOfTrajectoriesPerManifold, const double maximumVelocityDiscrepancy = 0.5 );

void writeManifoldStateHistoryAtThetaToFile( std::map< int, std::map< double, Eigen::Vector6d > >& manifoldStateHistory,
                                             int librationPointNr, std::string orbitType, double desiredJacobiEnergy,
                                             int displacementFromOrbitSign, int integrationTimeDirection, double thetaStoppingAngle);

Eigen::MatrixXd connectManifoldsAtTheta( const std::string orbitType = "vertical", const double thetaStoppingAngle = -90.0,
                                         const int numberOfTrajectoriesPerManifold = 100, const double desiredJacobiEnergy = 3.1,
                                         const int saveFrequency = 1000,
                                         const double massParameter = tudat::gravitation::circular_restricted_three_body_problem::computeMassParameter(
                                                            tudat::celestial_body_constants::EARTH_GRAVITATIONAL_PARAMETER,
                                                            tudat::celestial_body_constants::MOON_GRAVITATIONAL_PARAMETER ) );

#endif //TUDATBUNDLE_REFINEORBITCLEVEL_H
