#ifndef TUDATBUNDLE_COMPUTELEVEL1CORRECTION_H
#define TUDATBUNDLE_COMPUTELEVEL1CORRECTION_H

#include <Eigen/Core>

Eigen::VectorXd LocalLevelITLTUpdate(const Eigen::VectorXd localStateVector, const Eigen::VectorXd localDefectVector, const Eigen::MatrixXd localPropagatedStateInclSTMThrust, const Eigen::MatrixXd localPropagatedStateInclSTMCoast );

Eigen::VectorXd computeLevel1Correction( const Eigen::VectorXd deviationVector, const Eigen::MatrixXd propagatedStatesInclSTM, const Eigen::VectorXd initialGuess, const int numberOfPatchPoints);

#endif  // TUDATBUNDLE_COMPUTELEVEL1CORRECTION_H
