#ifndef TUDATBUNDLE_FLOQUETAPPROXIMATION_H
#define TUDATBUNDLE_FLOQUETAPPROXIMATION_H



#include <string>
#include <Eigen/Core>

//Eigen::VectorXd floquetCorrection (Eigen::MatrixXcd perturbationComponents);


Eigen::VectorXd floquetApproximation( int librationPointNr, std::string orbitType,
                                                   double amplitude, double thrustMagnitude, double accelerationAngle, double accelerationAngle2, const double initialMass, const int numberOfPatchPoints, const double maxEigenValueDeviation = 1.0E-6 );



#endif  // TUDATBUNDLE_FLOQUETAPPROXIMATION_H
