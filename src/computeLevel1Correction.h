#ifndef TUDATBUNDLE_COMPUTELEVEL1CORRECTION_H
#define TUDATBUNDLE_COMPUTELEVEL1CORRECTION_H

#include <Eigen/Core>

Eigen::VectorXd computeLevel1Correction( const Eigen::VectorXd deviationVector, const Eigen::MatrixXd propagatedStatesInclSTM, const int numberOfPatchPoints);

#endif  // TUDATBUNDLE_COMPUTELEVEL1CORRECTION_H
