#ifndef TUDATBUNDLE_COMPUTEDIFFERENTIALCORRECTION_H
#define TUDATBUNDLE_COMPUTEDIFFERENTIALCORRECTION_H



#include "computeDifferentialCorrection.cpp"


Eigen::VectorXd computeDifferentialCorrection( int librationPointNr, std::string orbitType,
                                               Eigen::MatrixXd cartesianStateWithStm, bool xPositionFixed );



#endif  // TUDATBUNDLE_COMPUTEDIFFERENTIALCORRECTION_H
