#ifndef TUDATBUNDLE_COMPUTEMANIFOLDSAUGMENTED_H
#define TUDATBUNDLE_COMPUTEMANIFOLDSAUGMENTED_H


#include <string>
#include <vector>
#include <map>

#include <Eigen/Core>

#include "Tudat/Basics/basicTypedefs.h"


void computeManifoldsAugmented( const Eigen::Vector6d initialStateVector, const double orbitalPeriod, const int orbitNumber,
                       const int librationPointNr, const std::string orbitType,
                       const double massParameter = tudat::gravitation::circular_restricted_three_body_problem::computeMassParameter(tudat::celestial_body_constants::EARTH_GRAVITATIONAL_PARAMETER, tudat::celestial_body_constants::MOON_GRAVITATIONAL_PARAMETER ),
                       const double eigenvectorDisplacementFromOrbit = 1.0E-6,
                       const int numberOfTrajectoriesPerManifold = 100, const int saveFrequency = 1000,
                       const bool saveEigenvectors = true,
                       const double maximumIntegrationTimeManifoldTrajectories = 50.0,
                       const double maxEigenvalueDeviation = 1.0E-3 );

#endif  // TUDATBUNDLE_COMPUTEMANIFOLDSAUGMENTED_H
