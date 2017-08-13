#ifndef TUDATBUNDLE_COMPLETEINITIALCONDITIONSHALOFAMILY_H
#define TUDATBUNDLE_COMPLETEINITIALCONDITIONSHALOFAMILY_H



#include "completeInitialConditionsHaloFamily.cpp"


void completeInitialConditionsHaloFamily( Eigen::VectorXd initialStateVector1, Eigen::VectorXd initialStateVector2,
                                          double orbitalPeriod1, double orbitalPeriod2, int librationPointNr,
                                          const double primaryGravitationalParameter, const double secondaryGravitationalParameter,
                                          double maxPositionDeviationFromPeriodicOrbit, double maxVelocityDeviationFromPeriodicOrbit,
                                          double maxEigenvalueDeviation );



#endif //TUDATBUNDLE_COMPLETEINITIALCONDITIONSHALOFAMILY_H
