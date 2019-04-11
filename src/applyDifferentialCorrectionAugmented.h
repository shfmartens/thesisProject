#ifndef TUDATBUNDLE_APPLYDIFFERENTIALCORRECTIONAUGMENTED_H
#define TUDATBUNDLE_APPLYDIFFERENTIALCORRECTIONAUGMENTED_H



#include "Eigen/Core"

void computeDifferenceStateDerivatives ( const Eigen::MatrixXd fullStateVectorAugmented, const Eigen::MatrixXd fullStateVectorBallistic, const double finalTime);

int computePositionMinimumDeviation (const Eigen::MatrixXd initialStateVectorInclSTM, const double massParameter, const double orbitalPeriod, const bool symmetryDependence );

Eigen::VectorXd computeDeviationVector (const Eigen::VectorXd& initialStateVector, const double targetPeriod,
                                        const Eigen::VectorXd& targetStateVector, const double currentPeriod);

Eigen::VectorXd applyDifferentialCorrectionAugmented( const int librationPointNr,
                                             const Eigen::VectorXd& initialStateVector,
                                             double orbitalPeriod, const double massParameter,
                                             double maxPositionDeviationFromPeriodicOrbit,
                                             double maxVelocityDeviationFromPeriodicOrbit, const double maxPeriodDeviationFromPeriodicOrbit, const bool symmetryDependence,
                                             const int maxNumberOfIterations = 1000 );


#endif  // TUDATBUNDLE_APPLYDIFFERENTIALCORRECTIONAUGMENTED_H
