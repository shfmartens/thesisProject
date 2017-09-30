#ifndef TUDATBUNDLE_CHECKEIGENVALUES_H
#define TUDATBUNDLE_CHECKEIGENVALUES_H



#include "checkEigenvalues.cpp"


bool checkEigenvalues( Eigen::MatrixXd stateVectorInclSTM, double maxEigenvalueDeviation,
                       bool moduleOneInsteadOfRealOne );



#endif  // TUDATBUNDLE_CHECKEIGENVALUES_H
