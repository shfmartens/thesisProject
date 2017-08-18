#ifndef TUDATBUNDLE_COMPUTEDIFFERENTIALCORRECTION_H
#define TUDATBUNDLE_COMPUTEDIFFERENTIALCORRECTION_H



#include "computeDifferentialCorrection.cpp"


Eigen::VectorXd computeDifferentialCorrection( int librationPointNr, std::string orbitType,
                                               Eigen::VectorXd cartesianState );



#endif  // TUDATBUNDLE_COMPUTEDIFFERENTIALCORRECTION_H
