#ifndef TUDATBUNDLE_APPLYDIFFERENTIALCORRECTIONAUGMENTED_H
#define TUDATBUNDLE_APPLYDIFFERENTIALCORRECTIONAUGMENTED_H



#include "Eigen/Core"

Eigen::VectorXd computeDeviationsFromPeriodicOrbit(const Eigen::VectorXd deviationVector, const int numberOfPatchPoints);

Eigen::VectorXd applyMultipleShooting( const int librationPointNr,
                                             const Eigen::VectorXd& initialStateVector,
                                             const double massParameter, const int numberOfPatchPoints,
                                             double maxPositionDeviationFromPeriodicOrbit,
                                             double maxVelocityDeviationFromPeriodicOrbit, const double maxPeriodDeviationFromPeriodicOrbit,
                                             const int maxNumberOfIterations = 20 );


#endif  // TUDATBUNDLE_APPLYDIFFERENTIALCORRECTIONAUGMENTED_H
