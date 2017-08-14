#ifndef TUDATBUNDLE_CREATEINITIALCONDITIONSAXIALFAMILY_H
#define TUDATBUNDLE_CREATEINITIALCONDITIONSAXIALFAMILY_H



#include "createInitialConditionsAxialFamily.cpp"


void createInitialConditionsAxialFamily( Eigen::VectorXd initialStateVector1, Eigen::VectorXd initialStateVector2,
                                         double orbitalPeriod1, double orbitalPeriod2, int librationPointNr,
                                         const double primaryGravitationalParameter, const double secondaryGravitationalParameter,
                                         double maxPositionDeviationFromPeriodicOrbit, double maxVelocityDeviationFromPeriodicOrbit,
                                         double maxEigenvalueDeviation );



#endif //TUDATBUNDLE_CREATEINITIALCONDITIONSAXIALFAMILY_H
