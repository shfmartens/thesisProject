#ifndef TUDATBUNDLE_COMPUTEDIFFERENTIALCORRECTIONAUGMENTED_H
#define TUDATBUNDLE_COMPUTEDIFFERENTIALCORRECTIONAUGMENTED_H

#include <Eigen/Core>

Eigen::VectorXd computeDifferentialCorrectionAugmented( const Eigen::MatrixXd& cartesianStateWithStm, const Eigen::VectorXd& deviationVector, const bool symmetryDependence = true , const int stateIndex = 1 );



#endif  // TUDATBUNDLE_COMPUTEDIFFERENTIALCORRECTIONAUGMENTED_H
