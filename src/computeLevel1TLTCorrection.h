#ifndef TUDATBUNDLE_COMPUTELEVEL1TLTCORRECTION_H
#define TUDATBUNDLE_COMPUTELEVEL1TLTCORRECTION_H

#include <Eigen/Core>

void   computeLevelIDeviations(Eigen::VectorXd localStateVector,Eigen::MatrixXd& localPropagatedStateInclSTMThrust, Eigen::MatrixXd& localPropagatedStateInclSTMCoast, const double initialTimeThrust, const double finalTimeThrust, const double finalTimeCoast, const double massParameter );

Eigen::VectorXd LocalLevelITLTUpdate(const Eigen::VectorXd localStateVector, const Eigen::VectorXd localDefectVector, const Eigen::MatrixXd localPropagatedStateInclSTMThrust, const Eigen::MatrixXd localPropagatedStateInclSTMCoast, double& finalTimeThrust );

Eigen::VectorXd computeLevel1TLTCorrection( const Eigen::VectorXd deviationVector, const Eigen::MatrixXd propagatedStatesInclSTMThrustCoast, const Eigen::MatrixXd propagatedStatesInclSTM, const Eigen::VectorXd initialGuess, const int numberOfPatchPoints, double massParameter, int& numberOfLevelICorrections);

#endif  // TUDATBUNDLE_COMPUTELEVEL1TLTCORRECTION_H
