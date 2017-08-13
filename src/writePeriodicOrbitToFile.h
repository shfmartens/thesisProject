#ifndef TUDATBUNDLE_WRITEPERIODICORBITTOFILE_H
#define TUDATBUNDLE_WRITEPERIODICORBITTOFILE_H



#include "writePeriodicOrbitToFile.cpp"


Eigen::VectorXd writePeriodicOrbitToFile( Eigen::VectorXd initialStateVector, int lagrangePointNr, std::string orbitType,
                                          int orbitId, double orbitalPeriod, const double massParameter,
                                          bool completeInitialConditionsHaloFamily, int saveEveryNthIntegrationStep );



#endif  // TUDATBUNDLE_WRITEPERIODICORBITTOFILE_H
