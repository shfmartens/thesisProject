//
// Created by Koen Langemeijer on 25/07/2017.
//

#ifndef TUDATBUNDLE_RICHARDSONTHIRDORDERAPPROXIMATION_H
#define TUDATBUNDLE_RICHARDSONTHIRDORDERAPPROXIMATION_H



#include "richardsonThirdOrderApproximation.cpp"


Eigen::VectorXd richardsonThirdOrderApproximation(std::string orbitType, int lagrangePointNr, double amplitude);



#endif //TUDATBUNDLE_RICHARDSONTHIRDORDERAPPROXIMATION_H
