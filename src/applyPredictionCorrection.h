#ifndef TUDATBUNDLE_APPLYPREDICTONCORRECTION_H
#define TUDATBUNDLE_APPLYPREDICTONCORRECTION_H


#include "Eigen/Core"

Eigen::VectorXd computeDeviationNorms (const Eigen::VectorXd defectVector, const int numberOfPatchPoints );

void computeOrbitDeviations(Eigen::VectorXd inputStateVector, const int numberOfPatchPoints, Eigen::MatrixXd& propagatedStatesInclSTM, Eigen::VectorXd& defectVector, const double massParameter  );

Eigen::VectorXd applyPredictionCorrection( const int librationPointNr,
                                             const Eigen::VectorXd& initialStateVector,
                                             const double massParameter, const int numberOfPatchPoints,
                                             double maxPositionDeviationFromPeriodicOrbit,
                                             double maxVelocityDeviationFromPeriodicOrbit, const double maxPeriodDeviationFromPeriodicOrbit,
                                             const int maxNumberOfIterations = 20 );


#endif  // TUDATBUNDLE_APPLYPREDICTONCORRECTION_H
