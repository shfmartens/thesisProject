#ifndef TUDATBUNDLE_APPLYTWOLEVELTARGETERLOWTHRUST_H
#define TUDATBUNDLE_APPLYTWOLEVELTARGETERLOWTHRUST_H


#include "Eigen/Core"
#include <map>

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
