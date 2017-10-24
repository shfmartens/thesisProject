#ifndef TUDATBUNDLE_COMPUTEDIFFERENTIALCORRECTION_H
#define TUDATBUNDLE_COMPUTEDIFFERENTIALCORRECTION_H

#include <Eigen/Core>

Eigen::VectorXd computeDifferentialCorrection( const int librationPointNr, const std::string& orbitType,
                                               const Eigen::MatrixXd& cartesianStateWithStm, const bool xPositionFixed = false );



#endif  // TUDATBUNDLE_COMPUTEDIFFERENTIALCORRECTION_H
