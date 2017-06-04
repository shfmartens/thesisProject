//
// Created by Koen Langemeijer on 26/05/2017.
//

#ifndef TUDATBUNDLE_MAININITIALIZATION_H
#define TUDATBUNDLE_MAININITIALIZATION_H


#include "computeManifolds.cpp"
void computeManifolds( string orbit_type, string selected_orbit, Eigen::VectorXd initialStateVector,
                       const double primaryGravitationalParameter, const double secondaryGravitationalParameter,
                       double displacementFromOrbit, double maxDeviationFromPeriodicOrbit,
                       double integrationStopTime, int numberOfOrbits, int saveEveryNthStep );

#endif //TUDATBUNDLE_MAININITIALIZATION_H