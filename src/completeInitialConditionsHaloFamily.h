#ifndef TUDATBUNDLE_COMPLETEINITIALCONDITIONSHALOFAMILY_H
#define TUDATBUNDLE_COMPLETEINITIALCONDITIONSHALOFAMILY_H



#include <Eigen/Core>


void completeInitialConditionsHaloFamily( Eigen::VectorXd initialStateVector1, Eigen::VectorXd initialStateVector2,
                                          double orbitalPeriod1, double orbitalPeriod2, int librationPointNr,
                                          const double primaryGravitationalParameter = tudat::celestial_body_constants::EARTH_GRAVITATIONAL_PARAMETER,
                                          const double secondaryGravitationalParameter = tudat::celestial_body_constants::MOON_GRAVITATIONAL_PARAMETER,
                                          double maxPositionDeviationFromPeriodicOrbit = 1.0e-12, double maxVelocityDeviationFromPeriodicOrbit = 1.0e-12,
                                          double maxEigenvalueDeviation = 1.0e-3 );



#endif //TUDATBUNDLE_COMPLETEINITIALCONDITIONSHALOFAMILY_H
