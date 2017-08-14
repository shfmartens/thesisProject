#ifndef TUDATBUNDLE_COMPUTEDIFFERENTIALCORRECTION_H
#define TUDATBUNDLE_COMPUTEDIFFERENTIALCORRECTION_H



#include "computeDifferentialCorrection.cpp"


Eigen::VectorXd computeDifferentialCorrection( Eigen::VectorXd cartesianState, std::string orbitType );



#endif  // TUDATBUNDLE_COMPUTEDIFFERENTIALCORRECTION_H
