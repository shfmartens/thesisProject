#ifndef TUDATBUNDLE_COMPUTELEVEL2TLTCORRECTION_H
#define TUDATBUNDLE_COMPUTELEVEL2TLTCORRECTION_H

#include <Eigen/Core>

Eigen::VectorXd computeLevel2TLTCorrection( const Eigen::VectorXd deviationVector, const Eigen::MatrixXd propagatedStatesInclSTMThrust, const Eigen::MatrixXd propagatedStatesInclSTMCoast, const Eigen::VectorXd initialGuess, const int numberOfPatchPoints, double massParameter );

#endif  // TUDATBUNDLE_COMPUTELEVEL2TLTCORRECTION_H
