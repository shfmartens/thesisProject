#ifndef TUDATBUNDLE_COMPUTECOLLOCATIONCORRECTION_H
#define TUDATBUNDLE_COMPUTECOLLOCATIONCORRECTION_H

#include "Tudat/Basics/basicTypedefs.h"

#include <Eigen/Core>
#include <string>
#include <vector>


Eigen::VectorXcd computeComplexStateDerivative(const Eigen::VectorXcd singleOddState, Eigen::VectorXd thrustAndMassParameters);

Eigen::VectorXd computeDerivativesUsingComplexStep(Eigen::VectorXcd designVector, std::complex<double> currentTime, Eigen::VectorXd thrustAndMassParameters, const double epsilon);

Eigen::VectorXd computePeriodicityDerivativeUsingComplexStep(Eigen::VectorXcd initialState, Eigen::VectorXcd finalState, const double epsilon);

void recomputeTimeProperties(const Eigen::MatrixXd temporaryDesignVector, double& initialTime, Eigen::VectorXd& timeIntervals, const int numberOfCollocationPoints);

double computePhasePeriodicityDerivativeUsingComplexStep(const Eigen::VectorXcd columnInitialState, const Eigen::VectorXd phaseConstraintVector, const double epsilon);

std::complex<double> computeComplexJacobi(const Eigen::VectorXcd currentState, const double massParameter);

double computeHamiltonianDerivativeUsingComplexStep( const Eigen::VectorXcd currentState, const Eigen::VectorXd thrustAndMassParameters, const double HamiltonianTarget, const double epsilon, const double massParameter );


Eigen::VectorXd computeCollocationCorrection(const Eigen::MatrixXd collocationDefectVector, const Eigen::MatrixXd collocationDesignVectorconst, Eigen::VectorXd timeIntervals, Eigen::VectorXd thrustAndMassParameters, const int numberOfCollocationPoints, const int continuationIndex, const Eigen::MatrixXd phaseConstraintVector);



#endif  // TUDATBUNDLE_COMPUTECOLLOCATIONCORRECTION_H
