#ifndef TUDATBUNDLE_COMPUTEMANIFOLDS_H
#define TUDATBUNDLE_COMPUTEMANIFOLDS_H


#include <string>

#include <Eigen/Core>


void computeManifolds( Eigen::VectorXd initialStateVector, double orbitalPeriod, int librationPointNr,
                       std::string orbitType, int orbitId,
                       const double primaryGravitationalParameter, const double secondaryGravitationalParameter,
                       double displacementFromOrbit, int numberOfManifoldOrbits, int saveEveryNthIntegrationStep,
                       double maximumIntegrationTimeManifoldOrbits, double maxEigenvalueDeviation,
                       double positionBoundarySurroundingPrimaries );


#endif  // TUDATBUNDLE_COMPUTEMANIFOLDS_H
