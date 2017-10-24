#ifndef TUDATBUNDLE_APPLYDIFFERENTIALCORRECTION_H
#define TUDATBUNDLE_APPLYDIFFERENTIALCORRECTION_H



#include "Eigen/Core"


Eigen::VectorXd applyDifferentialCorrection( const int librationPointNr, const std::string& orbitType,
                                             const Eigen::VectorXd& initialStateVector,
                                             double orbitalPeriod, const double massParameter,
                                             double maxPositionDeviationFromPeriodicOrbit,
                                             double maxVelocityDeviationFromPeriodicOrbit,
                                             const int maxNumberOfIterations = 1000 );


#endif  // TUDATBUNDLE_APPLYDIFFERENTIALCORRECTION_H
