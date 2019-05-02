#ifndef TUDATBUNDLE_MORIMOTOFIRSTORDERAPPROXIMATION_H
#define TUDATBUNDLE_MORIMOTOFIRSTORDERAPPROXIMATION_H



#include <string>

Eigen::VectorXd  computeOffsets( const Eigen::Vector2d equilibriumLocation, const double minimumCenterEigenValue, const int stabilityType, const double amplitude, const double massParameter, const double currentTime, const int timeParameter );

void computeCenterEigenValues( const Eigen::MatrixXd statePropagationMatrix, double& minimumCenterEigenvalue, int& stabilityType, const double maxEigenvalueDeviation = 1.0e-3 );

Eigen::VectorXd morimotoFirstOrderApproximation( int librationPointNr,
                                                   double amplitude, double thrustMagnitude, double accelerationAngle, double accelerationAngle2, const double initialMass, const int numberOfPatchPoints );



#endif  // TUDATBUNDLE_MORIMOTOFIRSTORDERAPPROXIMATION_H
