#ifndef TUDATBUNDLE_APPLYMULTIVARIATEROOTFINDING_H
#define TUDATBUNDLE_APPLYMULTIVARIATEROOTFINDING_H



#include "Eigen/Core"

Eigen::Vector2d computeDeviation(const int librationPointNr, const Eigen::Vector2d currentLocation, const double alpha, const double massParameter);

Eigen::Vector2d applyMultivariateRootFinding( const int librationPointNr, const Eigen::Vector2d initialEquilibrium,
                                              const double alpha, const double thrustAcceleration, const double massParameter, double maxDeviationFromEquilibrium = 1.0e-12,
                                             const int maxNumberOfIterations = 10000 );

#endif  // TUDATBUNDLE_APPLYMULTIVARIATEROOTFINDING_H
