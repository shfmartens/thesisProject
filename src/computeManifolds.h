#ifndef TUDATBUNDLE_COMPUTEMANIFOLDS_H
#define TUDATBUNDLE_COMPUTEMANIFOLDS_H



#include "computeManifolds.cpp"


void computeManifolds( string orbit_type, string selected_orbit, Eigen::VectorXd initialStateVector,
                       const double primaryGravitationalParameter, const double secondaryGravitationalParameter,
                       double displacementFromOrbit, double maxDeviationFromPeriodicOrbit,
                       double integrationStopTime, int numberOfOrbits, int saveEveryNthStep );



#endif  // TUDATBUNDLE_COMPUTEMANIFOLDS_H