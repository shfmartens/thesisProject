#ifndef TUDATBUNDLE_APPLYDIFFERENTIALCORRECTIONAUGMENTED_H
#define TUDATBUNDLE_APPLYDIFFERENTIALCORRECTIONAUGMENTED_H



#include "Eigen/Core"

Eigen::VectorXd computeDeviationVector (const Eigen::VectorXd& initialStateVector, const double initialPeriod,
                                        const Eigen::VectorXd& targetStateVector, const double targetPeriod);

Eigen::VectorXd applyDifferentialCorrectionAugmented( const int librationPointNr,
                                             const Eigen::VectorXd& initialStateVector,
                                             double orbitalPeriod, const double massParameter,
                                             double maxPositionDeviationFromPeriodicOrbit,
                                             double maxVelocityDeviationFromPeriodicOrbit,
                                             const int maxNumberOfIterations = 1000 );


#endif  // TUDATBUNDLE_APPLYDIFFERENTIALCORRECTIONAUGMENTED_H
