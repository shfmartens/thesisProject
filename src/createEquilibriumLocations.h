#ifndef TUDATBUNDLE_CREATEEQUILIBRIUMLOCATIONS_H
#define TUDATBUNDLE_CREATEEQUILIBRIUMLOCATIONS_H

#include <map>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "Tudat/Basics/basicTypedefs.h"

#include <boost/function.hpp>

Eigen::Vector6d computeDeviationAfterPropagation(const Eigen::Vector3d equilibriumLocationWithIterations, const double accelerationMagnitude, const double accelerationAngle, const double massParameter, const double finalTime);

void writeResultsToFile (const int librationPointNr, const double parameterOfInterest, const std::string parameterSpecification, const double seedAngle, const double continuationDirection, std::map< double, std::map <double, Eigen::Vector3d > > equilibriaCatalog, std::map <double, std::map <double, Eigen::MatrixXd > > stabilityCatalog,
                          std::map <double, Eigen::Vector3d > deviationCatalog);


Eigen::MatrixXd computeEquilibriaStability(Eigen::Vector2d equilibriumLocation, const double alpha, const double accelerationMagnitude, const double massParameter);

double newtonRapshonRootFinding(const int librationPointNr, const double accelerationMagnitude,  const double massParameter, const double maxDeviationFromSolution = 1.0E-13, const double maxNumberOfIterations = 80000, const double relaxationParameter = 0.2 );

void equilibriaValidation(Eigen::Vector2d equilibriumLocation, double acceleration, double alpha, double massParameter);

Eigen::Vector2d computeSeedSolution(const int librationPointNr, const double thrustAcceleration, const double seedAngle, const double maxDeviationFromSolution, bool& seedExistence );

Eigen::Vector2d createEquilibriumLocations (const int librationPointNr, const double thrustAcceleration, const double accelerationAngle, const std::string parameterSpecification, const double ySign, const double massParameter = tudat::gravitation::circular_restricted_three_body_problem::computeMassParameter(
            tudat::celestial_body_constants::EARTH_GRAVITATIONAL_PARAMETER,tudat::celestial_body_constants::MOON_GRAVITATIONAL_PARAMETER), const double maxDeviationFromSolution = 1.0E-13, const int maxIterations = 1000000, const int saveFrequency = 1, const double stepSize = 0.0001, const double relaxationParameter = 0.2);

#endif  // TUDATBUNDLE_CREATEEQUILIBRIUMLOCATIONS_H
