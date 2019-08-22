#ifndef TUDATBUNDLE_APPLYTWOLEVELTARGETERLOWTHRUST_H
#define TUDATBUNDLE_APPLYTWOLEVELTARGETERLOWTHRUST_H


#include "Eigen/Core"
#include <map>

void writeTLTDataToFile(const int librationPointNr, const double accelerationMagnitude, const double alpha, const double amplitude, const int numberOfPatchPoints, const double correctionTime,
                              std::map< double, Eigen::VectorXd > stateHistory, const Eigen::VectorXd stateVector, Eigen::VectorXd deviations, const Eigen::MatrixXd propagatedStatesInclSTMThrust, const Eigen::MatrixXd propagatedStatesInclSTMCoast,
                              const int cycleNumber, const int correctorLevel, const int numberOfCorrections, const double correctionDuration );

Eigen::VectorXd rewriteInputGuess(const Eigen::VectorXd initialStateVector, const int numberOfPatchPoints);

void computeTwoLevelDeviations(Eigen::VectorXd inputStateVector, const int numberOfPatchPoints, Eigen::MatrixXd& propagatedStatesInclSTMCoast, Eigen::MatrixXd& propagatedStatesInclSTMThrust, Eigen::VectorXd& defectVector, std::map< double, Eigen::VectorXd >& stateHistory, const double massParameter);

Eigen::VectorXd computeTLTDeviationNorms (const Eigen::VectorXd defectVector, const int numberOfPatchPoints );


Eigen::VectorXd applyTwoLevelTargeterLowThrust( const int librationPointNr,
                                             const Eigen::VectorXd& initialStateVector,
                                             const double massParameter, const int numberOfPatchPoints,
                                             double maxPositionDeviationFromPeriodicOrbit,
                                             double maxVelocityDeviationFromPeriodicOrbit, const double maxPeriodDeviationFromPeriodicOrbit,
                                             const int maxNumberOfIterations = 20 );


#endif  // TUDATBUNDLE_applyTwoLevelTargeterLowThrust_H
