#ifndef TUDATBUNDLE_CREATEINITIALCONDITIONS_H
#define TUDATBUNDLE_CREATEINITIALCONDITIONS_H



#include <string>

void createInitialConditions( int librationPointNr, std::string orbitType,
                              const double massParameter = tudat::gravitation::circular_restricted_three_body_problem::computeMassParameter(
            tudat::celestial_body_constants::EARTH_GRAVITATIONAL_PARAMETER,
            tudat::celestial_body_constants::MOON_GRAVITATIONAL_PARAMETER ),
                              double maxPositionDeviationFromPeriodicOrbit = 1.0e-12, double maxVelocityDeviationFromPeriodicOrbit = 1.0e-12,
                              double maxEigenvalueDeviation = 1.0e-3 );


#endif  // TUDATBUNDLE_CREATEINITIALCONDITIONS_H
