#ifndef TUDATBUNDLE_APPLYMULTIVARIATEROOTFINDING_H
#define TUDATBUNDLE_APPLYMULTIVARIATEROOTFINDING_H



#include "Eigen/Core"

Eigen::MatrixXd computeJacobian(Eigen::Vector2d initialGuess, double acceleration, double alpha, double massParameter);

Eigen::Vector2d computeDeviation(Eigen::Vector2d equilibriumGuess, double acceleration, double alpha, bool& iterationsReached, double massParameter);

Eigen::Vector3d applyMultivariateRootFinding( const int librationPointNr, const Eigen::Vector2d initialEquilibrium,
                                              const double alpha, const double thrustAcceleration, bool& iterationsReached, const double massParameter, double maxDeviationFromEquilibrium = 1.0e-13,
                                             const int maxNumberOfIterations = 100 );

#endif  // TUDATBUNDLE_APPLYMULTIVARIATEROOTFINDING_H
