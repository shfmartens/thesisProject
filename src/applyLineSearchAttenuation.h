#ifndef TUDATBUNDLE_APPLYLINESEARCHATTENUATION_H
#define TUDATBUNDLE_APPLYLINESEARCHATTENUATION_H

#include "Tudat/Basics/basicTypedefs.h"

#include <Eigen/Core>
#include <string>
#include <vector>


void rearrangeTemporaryVectors(const Eigen::VectorXd temporaryDesignVector, const Eigen::VectorXd temporaryDerivativeVector, Eigen::MatrixXd& oddStates, Eigen::MatrixXd& oddStatesDerivatives, const int numberOfCollocationPoints);

Eigen::VectorXd computeDesignVectorDerivatives(const Eigen::VectorXd temporaryDesignVector, const Eigen::VectorXd thrustAndMassParameters, const int numberOfCollocationPoints  );

void applyLineSearchAttenuation(const Eigen::VectorXd collocationCorrectionVector,  Eigen::MatrixXd& collocationDefectVector,  Eigen::MatrixXd& collocationDesignVector, const Eigen::VectorXd timeIntervals, const Eigen::VectorXd thrustAndMassParameters, const int numberOfCollocationPoints, const int continuationIndex, const Eigen::VectorXd phaseConstraintVector, const int orbitNumber);




#endif  // TUDATBUNDLE_APPLYLINESEARCHATTENUATION_H
