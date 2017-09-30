#ifndef TUDATBUNDLE_APPLYDIFFERENTIALCORRECTION_H
#define TUDATBUNDLE_APPLYDIFFERENTIALCORRECTION_H



#include "Eigen/Core"


Eigen::VectorXd applyDifferentialCorrection( int librationPointNr, std::string orbitType,
                                             Eigen::VectorXd initialStateVector,
                                             double orbitalPeriod, const double massParameter,
                                             double maxPositionDeviationFromPeriodicOrbit,
                                             double maxVelocityDeviationFromPeriodicOrbit,
                                             int maxNumberOfIterations = 1000 );


#endif  // TUDATBUNDLE_APPLYDIFFERENTIALCORRECTION_H
