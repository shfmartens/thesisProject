#ifndef TUDATBUNDLE_APPLYPREDICTONCORRECTION_H
#define TUDATBUNDLE_APPLYPREDICTONCORRECTION_H


#include "Eigen/Core"
#include <map>

void shiftConvergedTrajectoryGuess(int librationPointNr,Eigen::VectorXd currentTrajectoryGuess, Eigen::VectorXd inputTrajectoryGuess, const Eigen::VectorXd offsetUnitVector, Eigen::VectorXd& convergedTrajectoryGuess, double massParameter, const int numberOfPatchPoints);


void writeCorrectorDataToFile(const int librationPointNr, const double accelerationMagnitude, const double alpha, const double amplitude, const int numberOfPatchPoints, const double correctionTime, const std::map< double, Eigen::VectorXd > stateHistory, const Eigen::VectorXd stateVector, Eigen::VectorXd deviations,
                              const Eigen::MatrixXd propagatedStatesInclSTM, const int cycleNumber, const int correctorLevel, const int numberOfCorrections, const double correctionDuration );

Eigen::VectorXd computeDeviationNorms (const Eigen::VectorXd defectVector, const int numberOfPatchPoints );

void computeOrbitDeviations(Eigen::VectorXd inputStateVector, const int numberOfPatchPoints, Eigen::MatrixXd& propagatedStatesInclSTM, Eigen::VectorXd& defectVector,   std::map< double, Eigen::VectorXd >& stateHistory,const double massParameter  );

Eigen::VectorXd applyPredictionCorrection( const int librationPointNr,
                                             const Eigen::VectorXd& initialStateVector,
                                             const double massParameter, const int numberOfPatchPoints,
                                             double maxPositionDeviationFromPeriodicOrbit,
                                             double maxVelocityDeviationFromPeriodicOrbit, const double maxPeriodDeviationFromPeriodicOrbit,
                                             const int maxNumberOfIterations = 10 );


#endif  // TUDATBUNDLE_APPLYPREDICTONCORRECTION_H
