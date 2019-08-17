#ifndef TUDATBUNDLE_COMPUTELEVEL1MASSREFINEMENTCORRECTION_H
#define TUDATBUNDLE_COMPUTELEVEL1MASSREFINEMENTCORRECTION_H

#include <Eigen/Core>

Eigen::VectorXd computeLevel1MassRefinementCorrection( const Eigen::VectorXd deviationVector, const Eigen::MatrixXd propagatedStatesInclSTM, const Eigen::VectorXd initialGuess, const int numberOfPatchPoints, double massParameter, int& numberOfLevelICorrections);

#endif  // TUDATBUNDLE_COMPUTELEVEL1MASSREFINEMENTCORRECTION_H
