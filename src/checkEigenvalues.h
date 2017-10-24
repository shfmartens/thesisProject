#ifndef TUDATBUNDLE_CHECKEIGENVALUES_H
#define TUDATBUNDLE_CHECKEIGENVALUES_H



#include "Eigen/Core"


bool checkEigenvalues( const Eigen::MatrixXd& stateVectorInclSTM, const double maxEigenvalueDeviation = 1.0E-3,
                      const bool moduleOneInsteadOfRealOne = false );



#endif  // TUDATBUNDLE_CHECKEIGENVALUES_H
