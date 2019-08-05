#ifndef TUDATBUNDLE_APPLYPREDICTONCORRECTION_H
#define TUDATBUNDLE_APPLYPREDICTONCORRECTION_H



#include "Eigen/Core"

Eigen::VectorXd computeDeviationVector(const Eigen::VectorXd targetVector, const double targetTime, const Eigen::VectorXd finalVector, const double finalTime);

Eigen::VectorXd computeDeviationsFromPeriodicOrbit(const Eigen::VectorXd deviationVector, const int numberOfPatchPoints);

Eigen::VectorXd computeLevel1Correction( const Eigen::VectorXd deviationVector, const Eigen::MatrixXd propagatedStatesInclSTM, const int numberOfPatchPoints);

Eigen::VectorXd computeLevel2Correction( const Eigen::VectorXd deviationVector, const Eigen::MatrixXd forwardPropagatedStatesInclSTM, const Eigen::VectorXd initialGuess, const int numberOfPatchPoints, const double massParameter, const bool hamiltonianConstraint, Eigen::VectorXd hamiltonianDeviationVector );

Eigen::VectorXd applyPredictionCorrection( const int librationPointNr,
                                             const Eigen::VectorXd& initialStateVector,
                                            const double targetHamiltonian,
                                             const double massParameter, const int numberOfPatchPoints,
                                             const bool hamiltonianConstraint,
                                             double maxPositionDeviationFromPeriodicOrbit,
                                             double maxVelocityDeviationFromPeriodicOrbit, const double maxPeriodDeviationFromPeriodicOrbit,
                                             const int maxNumberOfIterations = 20 );


#endif  // TUDATBUNDLE_APPLYPREDICTONCORRECTION_H
