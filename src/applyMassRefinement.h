#ifndef TUDATBUNDLE_APPLYMASSREFINEMENT_H
#define TUDATBUNDLE_APPLYMASSREFINEMENT_H


#include "Eigen/Core"
#include <map>

void writeMassRefinementDataToFile(const int librationPointNr, const double accelerationMagnitude, const double alpha, const double amplitude, const int numberOfPatchPoints, const double correctionTime, const std::map< double, Eigen::VectorXd > stateHistory, const Eigen::VectorXd stateVector, Eigen::VectorXd deviations,
                              const Eigen::MatrixXd propagatedStatesInclSTM, const int cycleNumber, const int correctorLevel, const int numberOfCorrections, const double correctionDuration );

void computeMassVaryingDeviations(Eigen::VectorXd inputStateVector, const int numberOfPatchPoints, Eigen::MatrixXd& propagatedStatesInclSTM, Eigen::VectorXd& defectVector, std::map< double, Eigen::VectorXd >& stateHistory, const double massParameter);

Eigen::VectorXd computeMassVaryingDeviationNorms(const Eigen::VectorXd defectVector, const int numberOfPatchPoints );

Eigen::VectorXd applyMassRefinement( const int librationPointNr,
                                             const Eigen::VectorXd& initialStateVector,
                                             const double massParameter, const int numberOfPatchPoints,
                                             double maxPositionDeviationFromPeriodicOrbit,
                                             double maxVelocityDeviationFromPeriodicOrbit, const double maxPeriodDeviationFromPeriodicOrbit,
                                             const int maxNumberOfIterations = 20 );


#endif  // TUDATBUNDLE_APPLYMASSREFINEMENT_H
