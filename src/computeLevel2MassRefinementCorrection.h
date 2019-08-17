#ifndef TUDATBUNDLE_COMPUTELEVEL2MASSREFINEMENTCORRECTION_H
#define TUDATBUNDLE_COMPUTELEVEL2MASSREFINEMENTCORRECTION_H

#include <Eigen/Core>

Eigen::VectorXd computeLevel2MassRefinementCorrection( const Eigen::VectorXd deviationVector, const Eigen::MatrixXd propagatedStatesInclSTM, const Eigen::VectorXd initialGuess, const int numberOfPatchPoints, double massParameter );

#endif  // TUDATBUNDLE_COMPUTELEVEL2MASSREFINEMENTCORRECTION_H
