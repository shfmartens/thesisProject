#ifndef TUDATBUNDLE_REFINEORBITCLEVEL_H
#define TUDATBUNDLE_REFINEORBITCLEVEL_H



#include "refineOrbitJacobiEnergy.cpp"


Eigen::MatrixXd connectManifolds(std::string orbitType, double thetaStoppingAngle, int numberOfManifoldOrbits,
                                 double desiredJacobiEnergy, double maxPositionDiscrepancy );


#endif //TUDATBUNDLE_REFINEORBITCLEVEL_H
