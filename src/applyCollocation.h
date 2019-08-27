#ifndef TUDATBUNDLE_APPLYCOLLOCATION_H
#define TUDATBUNDLE_APPLYCOLLOCATION_H


#include "Eigen/Core"
#include <map>


Eigen::VectorXd applyCollocation(const Eigen::VectorXd initialCollocationGuess, const double massParameter, const int numberOfCollocationPoints, Eigen::VectorXd& collocatedGuess,
                                                         const double maxPositionDeviationFromPeriodicOrbit, const double maxVelocityDeviationFromPeriodicOrbit, const double maxPeriodDeviationFromPeriodicOrbit, const int maxNumberOfIterations = 10 );



#endif  // TUDATBUNDLE_APPLYCOLLOCATION_H
