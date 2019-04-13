#ifndef TUDATBUNDLE_APPLYDIFFERENTIALCORRECTIONAUGMENTED_H
#define TUDATBUNDLE_APPLYDIFFERENTIALCORRECTIONAUGMENTED_H



#include "Eigen/Core"

Eigen::VectorXd computeDeviationVector (const Eigen::VectorXd& initialStateVector, const double targetPeriod,
                                        const Eigen::VectorXd& targetStateVector, const double currentPeriod);

Eigen::VectorXd applyDifferentialCorrectionAugmented( const int librationPointNr,
                                             const Eigen::VectorXd& initialStateVector,
                                             double orbitalPeriod, const double massParameter,
                                             double maxPositionDeviationFromPeriodicOrbit,
                                             double maxVelocityDeviationFromPeriodicOrbit, const double maxPeriodDeviationFromPeriodicOrbit,
                                             const int maxNumberOfIterations = 20 );


#endif  // TUDATBUNDLE_APPLYDIFFERENTIALCORRECTIONAUGMENTED_H
