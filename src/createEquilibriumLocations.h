#ifndef TUDATBUNDLE_CREATEEQUILIBRIUMLOCATIONS_H
#define TUDATBUNDLE_CREATEEQUILIBRIUMLOCATIONS_H

#include <map>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "Tudat/Basics/basicTypedefs.h"

#include <boost/function.hpp>

void writeResultsToFile (const int librationPointNr, const std::map< double, Eigen::Vector2d > equilibriaCatalog );


void createEquilibriumLocations (const int librationPointNr, const double thrustAcceleration, const double massParameter = tudat::gravitation::circular_restricted_three_body_problem::computeMassParameter(
            tudat::celestial_body_constants::EARTH_GRAVITATIONAL_PARAMETER,tudat::celestial_body_constants::MOON_GRAVITATIONAL_PARAMETER), const double maxDeviationFromSolution = 1.0e-15, const int maxIterations = 1000);

#endif  // TUDATBUNDLE_CREATEEQUILIBRIUMCONDITIONS_H
