#ifndef TUDATBUNDLE_COMPUTELEVEL2CORRECTION_H
#define TUDATBUNDLE_COMPUTELEVEL2CORRECTION_H

#include <Eigen/Core>

Eigen::MatrixXd computeHamiltonianPositionPartials(const double massParameter, const Eigen::VectorXd stateVector );

Eigen::MatrixXd computeLLECorrection(const Eigen::MatrixXd pastStateTransitionMatrix, const::Eigen::MatrixXd futurestateTransitionMatrix, const double pastTime, const double futureTime, const bool exteriorPoint);

Eigen::VectorXd computeLevel2Correction( const Eigen::VectorXd deviationVector, const Eigen::MatrixXd forwardPropagatedStatesInclSTM, const Eigen::VectorXd initialGuess, const int numberOfPatchPoints, const double massParameter, const bool hamiltonianConstraint, Eigen::VectorXd hamiltonianDeviationVector );



#endif  // TUDATBUNDLE_COMPUTELEVEL2CORRECTION_H
