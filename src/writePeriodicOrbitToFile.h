//
// Created by Koen Langemeijer on 13/07/2017.
//

#ifndef TUDATBUNDLE_WRITEPERIODICORBITTOFILE_H
#define TUDATBUNDLE_WRITEPERIODICORBITTOFILE_H



#include "writePeriodicOrbitToFile.cpp"


Eigen::VectorXd writePeriodicOrbitToFile( Eigen::VectorXd initialStateVector, int lagrangePointNr, string orbitType,
                                          int orbitId, double orbitalPeriod, const double massParameter,
                                          int saveEveryNthIntegrationStep);



#endif //TUDATBUNDLE_WRITEPERIODICORBITTOFILE_H
