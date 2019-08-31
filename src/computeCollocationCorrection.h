#ifndef TUDATBUNDLE_COMPUTECOLLOCATIONCORRECTION_H
#define TUDATBUNDLE_COMPUTECOLLOCATIONCORRECTION_H

#include "Tudat/Basics/basicTypedefs.h"

#include <Eigen/Core>
#include <string>
#include <vector>


Eigen::VectorXcd computeComplexStateDerivative(const Eigen::VectorXcd singleOddState, Eigen::VectorXd thrustAndMassParameters);

Eigen::VectorXd computeDerivativesUsingComplexStep(Eigen::VectorXcd designVector, double currentTime, Eigen::VectorXd thrustAndMassParameters);

Eigen::VectorXd computeCollocationCorrection(const Eigen::MatrixXd collocationDefectVector, const Eigen::MatrixXd collocationDesignVectorconst, Eigen::VectorXd timeIntervals, Eigen::VectorXd thrustAndMassParameters, const int numberOfCollocationPoints);



#endif  // TUDATBUNDLE_COMPUTECOLLOCATIONCORRECTION_H
