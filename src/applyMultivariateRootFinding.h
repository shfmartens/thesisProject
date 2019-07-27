#ifndef TUDATBUNDLE_APPLYMULTIVARIATEROOTFINDING_H
#define TUDATBUNDLE_APPLYMULTIVARIATEROOTFINDING_H



#include "Eigen/Core"

Eigen::MatrixXd computeJacobian(Eigen::Vector2d currentGuess, const double massParameter);

Eigen::Vector2d computeConstraintVector(Eigen::Vector2d currentGuess, const double thrustAcceleration, const double alpha, const double massParameter);

Eigen::Vector3d applyMultivariateRootFinding( const Eigen::Vector2d initialEquilibrium,
                                              const double thrustAcceleration, const double alpha, const double massParameter, double relaxationParameter, const double maxDeviationFromEquilibrium = 1.0e-13,
                                             const int maxNumberOfIterations = 100000 );

#endif  // TUDATBUNDLE_APPLYMULTIVARIATEROOTFINDING_H
