#ifndef TUDATBUNDLE_COMPUTELEVEL2CORRECTION_H
#define TUDATBUNDLE_COMPUTELEVEL2CORRECTION_H

#include <Eigen/Core>

Eigen::MatrixXd computeWeightedLyapunovExponent(const Eigen::MatrixXd stateTransitionMatrixPO, const Eigen::MatrixXd stateTransitionMatrixFP, const double previousTime,const double nextTime );


Eigen::VectorXd computeLevel2Correction( const Eigen::VectorXd deviationVector, const Eigen::MatrixXd forwardPropagatedStatesInclSTM, const Eigen::VectorXd initialGuess, const Eigen::VectorXd unitOffsetVector, const int numberOfPatchPoints, const double massParameter );



#endif  // TUDATBUNDLE_COMPUTELEVEL2CORRECTION_H
