#ifndef TUDATBUNDLE_FLOQUETAPPROXIMATION_H
#define TUDATBUNDLE_FLOQUETAPPROXIMATION_H



#include <string>
#include <Eigen/Core>

void computeMotionDecomposition(const int librationPointNr, const std::string orbitType, Eigen::MatrixXd statePropagationMatrix, Eigen::MatrixXd stateTransitionMatrix, Eigen::VectorXd initialPerturbationVector, const double perturbationTime, const double numericalThreshold );

Eigen::VectorXd computeVelocityCorrection(const int librationPointNr, const std::string orbitType, Eigen::MatrixXd statePropagationMatrix, Eigen::MatrixXd stateTransitionMatrix, Eigen::VectorXd initialPerturbationVector, const double perturbationTime, const double numericalThreshold = 1.0E-13 );


Eigen::VectorXd floquetApproximation( int librationPointNr, std::string orbitType,
                                                   double amplitude, double thrustMagnitude, double accelerationAngle, double accelerationAngle2, const double initialMass, const int numberOfPatchPoints, const double correctionTime = 5.0, const double maxEigenValueDeviation = 1.0E-6 );



#endif  // TUDATBUNDLE_FLOQUETAPPROXIMATION_H
