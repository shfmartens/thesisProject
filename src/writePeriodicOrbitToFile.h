#ifndef TUDATBUNDLE_WRITEPERIODICORBITTOFILE_H
#define TUDATBUNDLE_WRITEPERIODICORBITTOFILE_H



#include "writePeriodicOrbitToFile.cpp"


Eigen::MatrixXd writePeriodicOrbitToFile( Eigen::VectorXd initialStateVector, int lagrangePointNr, std::string orbitType,
                                          int orbitId, double orbitalPeriod, const double massParameter,
                                          bool completeInitialConditionsHaloFamily = false, int saveEveryNthIntegrationStep = 1000 );



#endif  // TUDATBUNDLE_WRITEPERIODICORBITTOFILE_H
