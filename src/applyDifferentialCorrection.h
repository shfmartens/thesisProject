#ifndef TUDATBUNDLE_APPLYDIFFERENTIALCORRECTION_H
#define TUDATBUNDLE_APPLYDIFFERENTIALCORRECTION_H



#include "applyDifferentialCorrection.cpp"


Eigen::VectorXd applyDifferentialCorrection( int librationPointNr, std::string orbitType,
                                             Eigen::VectorXd initialStateVector,
                                             double orbitalPeriod, const double massParameter,
                                             double maxPositionDeviationFromPeriodicOrbit,
                                             double maxVelocityDeviationFromPeriodicOrbit );


#endif  // TUDATBUNDLE_APPLYDIFFERENTIALCORRECTION_H
