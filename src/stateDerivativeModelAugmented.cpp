#include "Tudat/Basics/utilityMacros.h"
#include "Tudat/Astrodynamics/Gravitation/librationPoint.h"

#include <cmath>
#include "computeManifoldsAugmented.h"
#include "stateDerivativeModelAugmented.h"
#include <iostream>

double determinePointingSign(const std::string thrustPointing) {
    if (thrustPointing == "left" ) {
        double pointingSign = 1.0;
        return pointingSign;
    } if (thrustPointing == "right") {
        double pointingSign = -1.0;
        return pointingSign;
    }
}

Eigen::MatrixXd computeStateDerivativeAugmented( const double time, const Eigen::MatrixXd& cartesianState )
{
    // Time is not directly used in the function.
    TUDAT_UNUSED_PARAMETER( time );

    // Declare mass parameter, spacecraft name and thrust law restrictions
    extern double massParameter;
    extern std::string spacecraftName;
    extern std::string thrustPointing;
    //std::string spacecraftName = "deepSpace";
    //std::string thrustPointing = "left";
    double pointingSign = determinePointingSign(thrustPointing);
    //double pointingSign = 1.0;

    // Retrieve spacecraft properties and thrust law restrictions
    Eigen::MatrixXd characteristics = retrieveSpacecraftProperties( spacecraftName );
    double thrustMagnitude = characteristics(0);
    double massRate        = characteristics(2);

    // Set the right thrusting
    if (thrustPointing == "left" || thrustPointing == "right"){

        // Declare state derivative vector with same length as the state.
        Eigen::MatrixXd stateDerivative = Eigen::MatrixXd::Zero( 5, 6 );

        // Set the derivative of the position equal to the velocities.
        stateDerivative.block( 0, 0, 2, 1 ) = cartesianState.block( 2, 0, 2, 1 );

        double xPositionScaledSquared = (cartesianState(0)+massParameter) * (cartesianState(0)+massParameter);
        double xPositionScaledSquared2 = (1.0-massParameter-cartesianState(0)) * (1.0-massParameter-cartesianState(0));
        double yPositionScaledSquared = (cartesianState(1) * cartesianState(1) );

        // Compute distances to primaries.
        double distanceToPrimaryBody   = sqrt(xPositionScaledSquared     + yPositionScaledSquared);
        double distanceToSecondaryBody = sqrt(xPositionScaledSquared2 + yPositionScaledSquared);

        double distanceToPrimaryCubed = distanceToPrimaryBody * distanceToPrimaryBody * distanceToPrimaryBody;
        double distanceToSecondaryCubed = distanceToSecondaryBody * distanceToSecondaryBody * distanceToSecondaryBody;

        double distanceToPrimaryToFifthPower = distanceToPrimaryCubed * distanceToPrimaryBody * distanceToPrimaryBody;
        double distanceToSecondaryToFifthPower = distanceToSecondaryCubed * distanceToSecondaryBody * distanceToSecondaryBody;

        // Set the derivative of the velocities to the accelerations.
        double termRelatedToPrimaryBody   = (1.0-massParameter)/distanceToPrimaryCubed;
        double termRelatedToSecondaryBody = massParameter      /distanceToSecondaryCubed;
        double velocityMagnitude = sqrt( cartesianState(2) * cartesianState(2) + cartesianState(3) * cartesianState(3) );
        double xTermRelatedToThrust = (thrustMagnitude / cartesianState(4)) * (pointingSign * -1.0 * cartesianState(3) ) / velocityMagnitude;
        double yTermRelatedToThrust = (thrustMagnitude / cartesianState(4)) * (pointingSign * cartesianState(2) ) / velocityMagnitude;
        stateDerivative( 2, 0 ) = -termRelatedToPrimaryBody*(massParameter+cartesianState(0)) + termRelatedToSecondaryBody*(1.0-massParameter-cartesianState(0)) + cartesianState(0) + 2.0*cartesianState(4) + xTermRelatedToThrust;
        stateDerivative( 3, 0 ) = -termRelatedToPrimaryBody*cartesianState(1)                 - termRelatedToSecondaryBody*cartesianState(1)                     + cartesianState(1) - 2.0*cartesianState(3) + yTermRelatedToThrust;

        //Set the derivate of the mass to the mass flow rate.
        stateDerivative( 4, 0 ) = massRate;

        // Compute partial derivatives of the potential.
        double Uxx = (3.0*(1.0-massParameter)*xPositionScaledSquared          )/distanceToPrimaryToFifthPower+ (3.0*massParameter*xPositionScaledSquared2           )/distanceToSecondaryToFifthPower - (1.0-massParameter)/distanceToPrimaryCubed - massParameter/distanceToSecondaryCubed + 1.0;
        double Uxy = (3.0*(1.0-massParameter)*(cartesianState(0)+massParameter)*cartesianState(1))/distanceToPrimaryToFifthPower- (3.0*massParameter*(1.0-massParameter-cartesianState(0))*cartesianState(1))/distanceToSecondaryToFifthPower;
        double Uyx = Uxy;
        double Uyy = (3.0*(1.0-massParameter)*yPositionScaledSquared                         )/distanceToPrimaryToFifthPower+ (3.0*massParameter*yPositionScaledSquared                             )/distanceToSecondaryToFifthPower - (1.0-massParameter)/distanceToPrimaryCubed - massParameter/distanceToSecondaryCubed + 1.0 ;

        //Compute the acceleration derivatives with respect to the velocities and mass
        double partialXaccXvel =  (thrustMagnitude / cartesianState(4)) * ((pointingSign * cartesianState(3) * cartesianState(2) ) / (velocityMagnitude * sqrt( velocityMagnitude ) ) ) ;
        double partialXaccYvel =  2.0 +  (thrustMagnitude / cartesianState(4)) * ((-1.0 * pointingSign / velocityMagnitude) + (pointingSign * cartesianState(3) * cartesianState(3) / (velocityMagnitude * sqrt(velocityMagnitude) ) ) );
        double partialYaccXvel =  -2.0 + (thrustMagnitude / cartesianState(4)) * (( 1.0 * pointingSign / velocityMagnitude) + ( -1.0 * pointingSign * cartesianState(2) * cartesianState(2) / (velocityMagnitude * sqrt(velocityMagnitude) ) ) ) ;
        double partialYaccYvel =  (thrustMagnitude / cartesianState(4)) * ((pointingSign * -1.0 * cartesianState(2) * cartesianState(3) ) / (velocityMagnitude * sqrt( velocityMagnitude ) ) ) ;

        double partialXaccMass =  (-thrustMagnitude / (cartesianState(4) * cartesianState(4))) * xTermRelatedToThrust;
        double partialYaccMass =  (-thrustMagnitude / (cartesianState(4) * cartesianState(4))) * yTermRelatedToThrust;
        // Create the STM-derivative matrix
        Eigen::MatrixXd stmDerivativeFunction (5,5);
        stmDerivativeFunction << 0.0, 0.0, 1.0,             0.0,             0.0,
                                 0.0, 0.0, 0.0,             1.0,             0.0,
                                 Uxx, Uxy, partialXaccXvel, partialXaccYvel, partialXaccMass,
                                 Uyx, Uyy, partialYaccXvel, partialYaccYvel, partialYaccMass,
                                 0.0, 0.0, 0.0,             0.0,             0.0;

        // Differentiate the STM.
        stateDerivative.block( 0, 1, 5, 5 ) = stmDerivativeFunction * cartesianState.block( 0, 1, 5, 5 );

        std::cout << "======================================================" << std::endl;
        std::cout << "Spacecraft Name : " << spacecraftName     << std::endl;
        std::cout << "Thrust Pointing : " << thrustPointing     << std::endl;
        std::cout << "Pointing Sign   : " << pointingSign       << std::endl;
        std::cout << "thrust Magnitude: " << thrustMagnitude    << std::endl;
        std::cout << "Mass rate       : " << massRate           << std::endl;
        std::cout << "State Derivative: " << stateDerivative    << std::endl;
        std::cout << "======================================================" << std::endl;

        return stateDerivative;

    } else {
    double alpha = std::stod(thrustPointing);

    // Declare state derivative vector with same length as the state.
    Eigen::MatrixXd stateDerivative = Eigen::MatrixXd::Zero( 5, 6 );

    // Set the derivative of the position equal to the velocities.
    stateDerivative.block( 0, 0, 2, 1 ) = cartesianState.block( 2, 0, 2, 1 );

    double xPositionScaledSquared = (cartesianState(0)+massParameter) * (cartesianState(0)+massParameter);
    double xPositionScaledSquared2 = (1.0-massParameter-cartesianState(0)) * (1.0-massParameter-cartesianState(0));
    double yPositionScaledSquared = (cartesianState(1) * cartesianState(1) );

    // Compute distances to primaries.
    double distanceToPrimaryBody   = sqrt(xPositionScaledSquared     + yPositionScaledSquared);
    double distanceToSecondaryBody = sqrt(xPositionScaledSquared2 + yPositionScaledSquared);

    double distanceToPrimaryCubed = distanceToPrimaryBody * distanceToPrimaryBody * distanceToPrimaryBody;
    double distanceToSecondaryCubed = distanceToSecondaryBody * distanceToSecondaryBody * distanceToSecondaryBody;

    double distanceToPrimaryToFifthPower = distanceToPrimaryCubed * distanceToPrimaryBody * distanceToPrimaryBody;
    double distanceToSecondaryToFifthPower = distanceToSecondaryCubed * distanceToSecondaryBody * distanceToSecondaryBody;

    // Set the derivative of the velocities to the accelerations.
    double termRelatedToPrimaryBody   = (1.0-massParameter)/distanceToPrimaryCubed;
    double termRelatedToSecondaryBody = massParameter      /distanceToSecondaryCubed;

    double xTermRelatedToThrust = (thrustMagnitude / cartesianState(4)) * cos( alpha * 2.0 * tudat::mathematical_constants::PI / 180.0 );
    double yTermRelatedToThrust = (thrustMagnitude / cartesianState(4)) * sin( alpha * 2.0 * tudat::mathematical_constants::PI / 180.0 );
    stateDerivative( 2, 0 ) = -termRelatedToPrimaryBody*(massParameter+cartesianState(0)) + termRelatedToSecondaryBody*(1.0-massParameter-cartesianState(0)) + cartesianState(0) + 2.0*cartesianState(4) + xTermRelatedToThrust;
    stateDerivative( 3, 0 ) = -termRelatedToPrimaryBody*cartesianState(1)                 - termRelatedToSecondaryBody*cartesianState(1)                     + cartesianState(1) - 2.0*cartesianState(3) + yTermRelatedToThrust;

    //Set the derivate of the mass to the mass flow rate.
    stateDerivative( 4, 0 ) = 0.0;

    // Compute partial derivatives of the potential.
    double Uxx = (3.0*(1.0-massParameter)*xPositionScaledSquared          )/distanceToPrimaryToFifthPower+ (3.0*massParameter*xPositionScaledSquared2           )/distanceToSecondaryToFifthPower - (1.0-massParameter)/distanceToPrimaryCubed - massParameter/distanceToSecondaryCubed + 1.0;
    double Uxy = (3.0*(1.0-massParameter)*(cartesianState(0)+massParameter)*cartesianState(1))/distanceToPrimaryToFifthPower- (3.0*massParameter*(1.0-massParameter-cartesianState(0))*cartesianState(1))/distanceToSecondaryToFifthPower;
    double Uyx = Uxy;
    double Uyy = (3.0*(1.0-massParameter)*yPositionScaledSquared                         )/distanceToPrimaryToFifthPower+ (3.0*massParameter*yPositionScaledSquared                             )/distanceToSecondaryToFifthPower - (1.0-massParameter)/distanceToPrimaryCubed - massParameter/distanceToSecondaryCubed + 1.0 ;

    //Compute the acceleration derivatives with respect to the velocities and mass
    double partialXaccXvel =   0.0  ;
    double partialXaccYvel =   2.0  ;
    double partialYaccXvel =  -2.0  ;
    double partialYaccYvel =   0.0  ;

    double partialXaccMass =  0.0; // since mass is assumed constant for now
    double partialYaccMass =  0.0; // since mass is assumed constant for now, if not, variable thrust magnitude and therefore f and mdot vary throughout time

    // Create the STM-derivative matrix
    Eigen::MatrixXd stmDerivativeFunction (5,5);
    stmDerivativeFunction << 0.0, 0.0, 1.0,             0.0,             0.0,
                             0.0, 0.0, 0.0,             1.0,             0.0,
                             Uxx, Uxy, partialXaccXvel, partialXaccYvel, partialXaccMass,
                             Uyx, Uyy, partialYaccXvel, partialYaccYvel, partialYaccMass,
                             0.0, 0.0, 0.0,             0.0,             0.0;

    // Differentiate the STM.
    stateDerivative.block( 0, 1, 5, 5 ) = stmDerivativeFunction * cartesianState.block( 0, 1, 5, 5 );

    return stateDerivative;

    }
}
