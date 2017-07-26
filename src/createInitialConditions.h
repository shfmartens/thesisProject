//
// Created by Koen Langemeijer on 13/07/2017.
//

#ifndef TUDATBUNDLE_CREATEINITIALCONDITIONS_H
#define TUDATBUNDLE_CREATEINITIALCONDITIONS_H



#include "createInitialConditions.cpp"


void createInitialConditions( int librationPointNr, string orbitType,
                              const double primaryGravitationalParameter, const double secondaryGravitationalParameter,
                              double maxPositionDeviationFromPeriodicOrbit, double maxVelocityDeviationFromPeriodicOrbit,
                              double maxDeviationEigenvalue);



#endif //TUDATBUNDLE_CREATEINITIALCONDITIONS_H
