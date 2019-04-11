#ifndef TUDATBUNDLE_COMPUTEDIFFERENTIALCORRECTIONAUGMENTED_H
#define TUDATBUNDLE_COMPUTEDIFFERENTIALCORRECTIONAUGMENTED_H

#include <Eigen/Core>

Eigen::VectorXd computeDifferentialCorrectionAugmented( const Eigen::MatrixXd& cartesianStateWithStm, const Eigen::VectorXd& deviationVector);



#endif  // TUDATBUNDLE_COMPUTEDIFFERENTIALCORRECTIONAUGMENTED_H
