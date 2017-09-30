#ifndef TUDATBUNDLE_PROPAGATEORBIT_H
#define TUDATBUNDLE_PROPAGATEORBIT_H



#include <Eigen/Core>

#include "propagateOrbit.cpp"


std::pair< Eigen::MatrixXd, double > propagateOrbit(
        Eigen::VectorXd stateVectorInclSTM, double massParameter, double currentTime,
                                int direction, double initialStepSize, double maximumStepSize );


std::pair< Eigen::MatrixXd, double >  propagateOrbitToFinalCondition(
        const Eigen::MatrixXd fullInitialState, const double massParameter, const double finalTime, int direction,
        const double initialTime = 0.0 );

#endif  // TUDATBUNDLE_PROPAGATEORBIT_H
