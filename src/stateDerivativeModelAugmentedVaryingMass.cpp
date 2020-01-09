#include "Tudat/Basics/utilityMacros.h"
#include "Tudat/Astrodynamics/Gravitation/librationPoint.h"
#include "Tudat/Astrodynamics/BasicAstrodynamics/celestialBodyConstants.h"
#include "Tudat/Astrodynamics/BasicAstrodynamics/physicalConstants.h"
#include "Tudat/Astrodynamics/BasicAstrodynamics/astrodynamicsFunctions.h"


#include "stateDerivativeModelAugmentedVaryingMass.h"
#include <iostream>


Eigen::MatrixXd computeStateDerivativeAugmentedVaryingMass( const double time, const Eigen::MatrixXd& cartesianState )
{

    // ==== Declare essential model parameters and derivatives === //

    // Time is not directly used in the function.
    TUDAT_UNUSED_PARAMETER( time );

    // Declare mass parameter.
    extern double massParameter;

    // Compute mass Rate
    double characteristicLength = 384400.0 * 1000.0; // Distance between Earth and Moon [m]
    double orbitalPeriod = tudat::basic_astrodynamics::computeKeplerOrbitalPeriod( characteristicLength, tudat::celestial_body_constants::EARTH_GRAVITATIONAL_PARAMETER, tudat::celestial_body_constants::MOON_GRAVITATIONAL_PARAMETER / tudat::physical_constants::GRAVITATIONAL_CONSTANT) ;
    double characteristicTime = orbitalPeriod /  ( 2.0 * tudat::mathematical_constants::PI); // inverted mean motion in [s]
    double seaGravitationalAcceleration = tudat::physical_constants::SEA_LEVEL_GRAVITATIONAL_ACCELERATION; // TUDAT
    double specificImpulse = 3000.0; // Recheck spacecraft properties

    double massRate = ( -1.0 * cartesianState(6,0) * characteristicLength ) / (specificImpulse * seaGravitationalAcceleration * characteristicTime );

    // === Declare state derivative vector with same length as the state. ==
    Eigen::MatrixXd stateDerivative = Eigen::MatrixXd::Zero( 10, 11 );

    // === Compute state derivatives of the state vector entries. ==

    // set position derivatives equal to velocities
    stateDerivative.block( 0, 0, 3, 1 ) = cartesianState.block( 3, 0, 3, 1 );

    double xPositionScaledSquared = (cartesianState(0)+massParameter) * (cartesianState(0)+massParameter);
    double xPositionScaledSquared2 = (1.0-massParameter-cartesianState(0)) * (1.0-massParameter-cartesianState(0));
    double yPositionScaledSquared = (cartesianState(1) * cartesianState(1) );
    double zPositionScaledSquared = (cartesianState(2) * cartesianState(2) );

    // Compute distances to primaries.
    double distanceToPrimaryBody   = sqrt(xPositionScaledSquared     + yPositionScaledSquared + zPositionScaledSquared);
    double distanceToSecondaryBody = sqrt(xPositionScaledSquared2 + yPositionScaledSquared + zPositionScaledSquared);

    double distanceToPrimaryCubed = distanceToPrimaryBody * distanceToPrimaryBody * distanceToPrimaryBody;
    double distanceToSecondaryCubed = distanceToSecondaryBody * distanceToSecondaryBody * distanceToSecondaryBody;

    double distanceToPrimaryToFifthPower = distanceToPrimaryCubed * distanceToPrimaryBody * distanceToPrimaryBody;
    double distanceToSecondaryToFifthPower = distanceToSecondaryCubed * distanceToSecondaryBody * distanceToSecondaryBody;


    // Set the derivative of the velocities to the accelerations including the low-thrust terms
    double termRelatedToPrimaryBody   = (1.0-massParameter)/distanceToPrimaryCubed;
    double termRelatedToSecondaryBody = massParameter      /distanceToSecondaryCubed;
    double alpha = cartesianState(7) * tudat::mathematical_constants::PI / 180.0;
    double beta = cartesianState(8) * tudat::mathematical_constants::PI / 180.0;

    stateDerivative( 3, 0 ) = -termRelatedToPrimaryBody*(massParameter+cartesianState(0)) + termRelatedToSecondaryBody*(1.0-massParameter-cartesianState(0)) + cartesianState(0) + 2.0*cartesianState(4) + (cartesianState(6) / cartesianState(9)) * std::cos( alpha ) * std::cos( beta );
    stateDerivative( 4, 0 ) = -termRelatedToPrimaryBody*cartesianState(1)                 - termRelatedToSecondaryBody*cartesianState(1)                     + cartesianState(1) - 2.0*cartesianState(3) + (cartesianState(6) / cartesianState(9)) * std::sin( alpha ) * std::cos( beta );
    stateDerivative( 5, 0 ) = -termRelatedToPrimaryBody*cartesianState(2)                 - termRelatedToSecondaryBody*cartesianState(2) + (cartesianState(6) / cartesianState(9)) * std::sin( beta ) ;

    // set the dervative of the thrust parameters equal to zero, since thrust magnitude and direction is constant
    stateDerivative( 6, 0 ) = 0.0;
    stateDerivative( 7, 0 ) = 0.0;
    stateDerivative( 8, 0 ) = 0.0;

    // set the derivative of the mass equal to the earlier computed massRate
    stateDerivative( 9, 0 ) = massRate;

    // Compute partial derivatives of the potential.
    double Uxx = (3.0*(1.0-massParameter)*xPositionScaledSquared          )/distanceToPrimaryToFifthPower+ (3.0*massParameter*xPositionScaledSquared2           )/distanceToSecondaryToFifthPower - (1.0-massParameter)/distanceToPrimaryCubed - massParameter/distanceToSecondaryCubed + 1.0;
    double Uxy = (3.0*(1.0-massParameter)*(cartesianState(0)+massParameter)*cartesianState(1))/distanceToPrimaryToFifthPower- (3.0*massParameter*(1.0-massParameter-cartesianState(0))*cartesianState(1))/distanceToSecondaryToFifthPower;
    double Uxz = (3.0*(1.0-massParameter)*(cartesianState(0)+massParameter)*cartesianState(2))/distanceToPrimaryToFifthPower- (3.0*massParameter*(1.0-massParameter-cartesianState(0))*cartesianState(2))/distanceToSecondaryToFifthPower;
    double Uyx = Uxy;
    double Uyy = (3.0*(1.0-massParameter)*yPositionScaledSquared          )/distanceToPrimaryToFifthPower+ (3.0*massParameter*yPositionScaledSquared            )/distanceToSecondaryToFifthPower - (1.0-massParameter)/distanceToPrimaryCubed - massParameter/distanceToSecondaryCubed + 1.0 ;
    double Uyz = (3.0*(1.0-massParameter)*cartesianState(1)*cartesianState(2)                )/distanceToPrimaryToFifthPower+ (3.0*massParameter*cartesianState(1)*cartesianState(2)                    )/distanceToSecondaryToFifthPower;
    double Uzx = Uxz;
    double Uzy = Uyz;
    double Uzz = (3.0*(1.0-massParameter)*zPositionScaledSquared                         )/distanceToPrimaryToFifthPower+ (3.0*massParameter*zPositionScaledSquared                             )/distanceToSecondaryToFifthPower - (1.0-massParameter)/distanceToPrimaryCubed - massParameter/distanceToSecondaryCubed ;

    // Compute partials w.r.t the thrust magnitude
    double xAccelerationPartialThrust = ( 1.0 / cartesianState(9) ) * std::cos( alpha ) * std::cos( beta );
    double yAccelerationPartialThrust = ( 1.0 / cartesianState(9) ) * std::sin( alpha ) * std::cos( beta );
    double zAccelerationPartialThrust = ( 1.0 / cartesianState(9) ) * std::sin( beta );
    double massRatePartialThrust = ( -1.0 * characteristicLength ) / (specificImpulse * seaGravitationalAcceleration * characteristicTime );

    // Compute partials w.r.t the alpha
    double xAccelerationPartialAlpha = -1.0 * (cartesianState(6) / cartesianState(9)) * std::sin( alpha ) * std::cos( beta );
    double yAccelerationPartialAlpha = (cartesianState(6) / cartesianState(9)) * std::cos( alpha ) * std::cos( beta );

    // Compute partials w.r.t the beta
    double xAccelerationPartialBeta = -1.0 * (cartesianState(6) / cartesianState(9)) * std::cos( alpha ) * std::sin( beta );
    double yAccelerationPartialBeta = -1.0 * (cartesianState(6) / cartesianState(9)) * std::sin( alpha ) * std::sin( beta );
    double zAccelerationPartialBeta = (cartesianState(6) / cartesianState(9)) * std::cos( beta );

    // Compute partials w.r.t the mass
    double xAccelerationPartialMass = -1.0 * (cartesianState(6) / ( cartesianState(9) * cartesianState(9) ) ) * std::cos( alpha ) * std::cos( beta );
    double yAccelerationPartialMass = -1.0 * (cartesianState(6) / ( cartesianState(9) * cartesianState(9) ) ) * std::sin( alpha ) * std::cos( beta );
    double zAccelerationPartialMass = -1.0 * (cartesianState(6) / ( cartesianState(9) * cartesianState(9) ) ) * std::sin( beta );




    // Create the STM-derivative matrix
    Eigen::MatrixXd stmDerivativeFunctionAugmented (10,10);
    stmDerivativeFunctionAugmented << 0.0, 0.0, 0.0,  1.0, 0.0, 0.0,  0.0,                        0.0,                           0.0,                         0.0,
                                      0.0, 0.0, 0.0,  0.0, 1.0, 0.0,  0.0,                        0.0,                           0.0,                         0.0,
                                      0.0, 0.0, 0.0,  0.0, 0.0, 1.0,  0.0,                        0.0,                           0.0,                         0.0,
                                      Uxx, Uxy, Uxz,  0.0, 2.0, 0.0,  xAccelerationPartialThrust, xAccelerationPartialAlpha,     xAccelerationPartialBeta,    xAccelerationPartialMass,
                                      Uyx, Uyy, Uyz, -2.0, 0.0, 0.0,  yAccelerationPartialThrust, yAccelerationPartialAlpha,     yAccelerationPartialBeta,    yAccelerationPartialMass,
                                      Uzx, Uzy, Uzz,  0.0, 0.0, 0.0,  zAccelerationPartialThrust, 0.0,                           zAccelerationPartialBeta,    zAccelerationPartialMass,
                                      0.0, 0.0, 0.0,  0.0, 0.0,  0.0, 0.0,                        0.0,                           0.0,                         0.0,
                                      0.0, 0.0, 0.0,  0.0, 0.0,  0.0, 0.0,                        0.0,                           0.0,                         0.0,
                                      0.0, 0.0, 0.0,  0.0, 0.0,  0.0, 0.0,                        0.0,                           0.0,                         0.0,
                                      0.0, 0.0, 0.0,  0.0, 0.0,  0.0, massRatePartialThrust,      0.0,                           0.0,                         0.0;

    // Differentiate the STM.
    stateDerivative.block( 0, 1, 10, 10 ) = stmDerivativeFunctionAugmented * cartesianState.block( 0, 1, 10, 10 );


    return stateDerivative;
}
