#ifndef TUDATBUNDLE_RICHARDSONTHIRDORDERAPPROXIMATIONMULTIPLE_H
#define TUDATBUNDLE_RICHARDSONTHIRDORDERAPPROXIMATIONMULTIPLE_H



#include <string>


Eigen::VectorXd richardsonThirdOrderApproximationMultiple( std::string orbitType, int librationPointNr,
                                                   double amplitude, double accelerationMagnitude, double thrustAngle1, double thrustAngle2, double initialMass, int numberOfPatchPoints, double n= 1.0 );



#endif  // TUDATBUNDLE_RICHARDSONTHIRDORDERAPPROXIMATIONMULTIPLE_H
