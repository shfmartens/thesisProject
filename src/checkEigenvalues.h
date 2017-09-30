#ifndef TUDATBUNDLE_CHECKEIGENVALUES_H
#define TUDATBUNDLE_CHECKEIGENVALUES_H



#include "Eigen/Core"


bool checkEigenvalues( Eigen::MatrixXd stateVectorInclSTM, double maxEigenvalueDeviation = 1.0E-3,
                       bool moduleOneInsteadOfRealOne = false );



#endif  // TUDATBUNDLE_CHECKEIGENVALUES_H
