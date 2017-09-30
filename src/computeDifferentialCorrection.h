#ifndef TUDATBUNDLE_COMPUTEDIFFERENTIALCORRECTION_H
#define TUDATBUNDLE_COMPUTEDIFFERENTIALCORRECTION_H

#include <Eigen/Core>

Eigen::VectorXd computeDifferentialCorrection( int librationPointNr, std::string orbitType,
                                               Eigen::MatrixXd cartesianStateWithStm, bool xPositionFixed = false );



#endif  // TUDATBUNDLE_COMPUTEDIFFERENTIALCORRECTION_H
