//
// Created by Koen Langemeijer on 13/07/2017.
//

#ifndef TUDATBUNDLE_APPLYDIFFERENTIALCORRECTION_H
#define TUDATBUNDLE_APPLYDIFFERENTIALCORRECTION_H



#include "applyDifferentialCorrection.cpp"


Eigen::VectorXd applyDifferentialCorrection( Eigen::VectorXd initialStateVector, string orbitType, const double massParameter,
                                             double maxPositionDeviationFromPeriodicOrbit, double maxVelocityDeviationFromPeriodicOrbit);



#endif //TUDATBUNDLE_APPLYDIFFERENTIALCORRECTION_H
