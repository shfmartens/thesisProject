#ifndef TUDATBUNDLE_RICHARDSONTHIRDORDERAPPROXIMATION_H
#define TUDATBUNDLE_RICHARDSONTHIRDORDERAPPROXIMATION_H



#include <string>


Eigen::VectorXd richardsonThirdOrderApproximation( std::string orbitType, int librationPointNr,
                                                   double amplitude, double n= 1.0 );



#endif  // TUDATBUNDLE_RICHARDSONTHIRDORDERAPPROXIMATION_H
