#ifndef TUDATBUNDLE_CREATEINITIALCONDITIONS_H
#define TUDATBUNDLE_CREATEINITIALCONDITIONS_H



#include "createInitialConditions.cpp"


void createInitialConditions( int librationPointNr, std::string orbitType,
                              const double primaryGravitationalParameter, const double secondaryGravitationalParameter,
                              double maxPositionDeviationFromPeriodicOrbit, double maxVelocityDeviationFromPeriodicOrbit,
                              double maxDeviationEigenvalue );



#endif  // TUDATBUNDLE_CREATEINITIALCONDITIONS_H
