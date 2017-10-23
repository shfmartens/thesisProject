#ifndef TUDATBUNDLE_REFINEORBITCLEVEL_H
#define TUDATBUNDLE_REFINEORBITCLEVEL_H



#include "connectManifoldsAtTheta.cpp"


void connectManifoldsAtTheta( const std::string orbitType = "vertical", const double thetaStoppingAngle = -90.0,
                              const int numberOfTrajectoriesPerManifold = 100, const double desiredJacobiEnergy = 3.1,
                              const int saveFrequency = 1000,
                              const double massParameter = tudat::gravitation::circular_restricted_three_body_problem::computeMassParameter(
                                                 tudat::celestial_body_constants::EARTH_GRAVITATIONAL_PARAMETER,
                                                 tudat::celestial_body_constants::MOON_GRAVITATIONAL_PARAMETER ) );

#endif //TUDATBUNDLE_REFINEORBITCLEVEL_H
