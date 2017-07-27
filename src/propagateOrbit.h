#ifndef TUDATBUNDLE_PROPAGATEORBIT_H
#define TUDATBUNDLE_PROPAGATEORBIT_H



#include <Eigen/Core>

#include "propagateOrbit.cpp"


Eigen::VectorXd propagateOrbit( Eigen::VectorXd stateVectorInclSTM, double massParameter, double currentTime,
                                int direction, double initialStepSize, double maximumStepSize );



#endif  // TUDATBUNDLE_PROPAGATEORBIT_H
