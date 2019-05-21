#ifndef TUDATBUNDLE_COMPUTELEVEL2CORRECTION_H
#define TUDATBUNDLE_COMPUTELEVEL2CORRECTION_H

#include <Eigen/Core>

Eigen::VectorXd computeLevel2Correction( const Eigen::VectorXd deviationVector, const Eigen::MatrixXd propagatedStatesInclSTM, const Eigen::VectorXd initialGuess, const int numberOfPatchPoints);

#endif  // TUDATBUNDLE_COMPUTELEVEL2CORRECTION_H
