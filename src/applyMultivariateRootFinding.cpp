#include <cmath>
#include <iostream>
#include <Eigen/Core>

#include "Tudat/Astrodynamics/BasicAstrodynamics/celestialBodyConstants.h"
#include "Tudat/Astrodynamics/Gravitation/librationPoint.h"
#include "Tudat/Astrodynamics/Gravitation/jacobiEnergy.h"

#include "applyDifferentialCorrection.h"
#include "computeDifferentialCorrection.h"
#include "propagateOrbit.h"

Eigen::Vector2d computeDeviation(const int librationPointNr, const Eigen::Vector2d currentLocation, const double alpha, const double thrustAcceleration, const double massParameter){

Eigen::Vector2d currentDeviation;

//Compute Distances from primaries to satellite
double xDistancePrimarySquared = (currentLocation(0)+massParameter)*(currentLocation(0)+massParameter);
double yDistancePrimarySquared = currentLocation(1)*currentLocation(1);
double xDistanceSecondarySquared = (currentLocation(0)-1.0+massParameter)*(currentLocation(0)-1.0+massParameter);
double yDistanceSecondarySquared = currentLocation(1)*currentLocation(1);

double distanceToPrimary = sqrt(xDistancePrimarySquared + yDistancePrimarySquared);
double distanceToPrimaryCubed = distanceToPrimary * distanceToPrimary * distanceToPrimary;
double distanceToSecondary = sqrt(xDistanceSecondarySquared + yDistanceSecondarySquared);
double distanceToSecondaryCubed = distanceToSecondary * distanceToSecondary * distanceToSecondary;

// Compute potential terms
double termRelatedToPrimary = (1.0 - massParameter)/distanceToPrimaryCubed;
double termRelatedToSecondary = massParameter/distanceToSecondaryCubed;
double recurringTerm = (1.0-termRelatedToPrimary-termRelatedToSecondary);
double recurringTerm2 = (-termRelatedToPrimary-termRelatedToSecondary);

currentDeviation(0) = currentLocation(0)*recurringTerm
        +massParameter*recurringTerm2+termRelatedToSecondary
        +thrustAcceleration*std::cos(alpha * tudat::mathematical_constants::PI /180.0);

currentDeviation(1) = currentLocation(1)*recurringTerm
        +thrustAcceleration*std::sin(alpha * tudat::mathematical_constants::PI /180.0);

return currentDeviation;
}

Eigen::MatrixXd computeJacobian (const int librationPointNr, const Eigen::Vector2d currentLocation, const double massParameter){

    Eigen::MatrixXd inverseJacobian = Eigen::MatrixXd::Zero(2,2);
    Eigen::MatrixXd Jacobian = Eigen::MatrixXd::Zero(2,2);

    //Compute Distances from primaries to satellite
    double xDistancePrimarySquared = (currentLocation(0)+massParameter)*(currentLocation(0)+massParameter);
    double yDistancePrimarySquared = currentLocation(1)*currentLocation(1);
    double xDistanceSecondarySquared = (currentLocation(0)-1.0+massParameter)*(currentLocation(0)-1.0+massParameter);
    double yDistanceSecondarySquared = currentLocation(1)*currentLocation(1);

    double distanceToPrimary = sqrt(xDistancePrimarySquared + yDistancePrimarySquared);
    double distanceToPrimaryCubed = distanceToPrimary * distanceToPrimary * distanceToPrimary;
    double distanceToPrimaryToTheFifth = distanceToPrimaryCubed * distanceToPrimary * distanceToPrimary;
    double distanceToSecondary = sqrt(xDistanceSecondarySquared + yDistanceSecondarySquared);
    double distanceToSecondaryCubed = distanceToSecondary * distanceToSecondary * distanceToSecondary;
    double distanceToSecondaryToTheFifth = distanceToSecondaryCubed * distanceToSecondary * distanceToSecondary;

    double termRelatedToPrimary = (1.0 - massParameter)/distanceToPrimaryCubed;
    double termRelatedToSecondary = massParameter/distanceToSecondaryCubed;
    double xDerivativePrimaryTerm = (-3.0*(1.0-massParameter)*(currentLocation(0)+massParameter))/(distanceToPrimaryToTheFifth);
    double yDerivativePrimaryTerm = (-3.0*(1.0-massParameter)*currentLocation(1))/(distanceToPrimaryToTheFifth);
    double xDerivativeSecondaryTerm = (-3.0*massParameter*(currentLocation(0)-1.0+massParameter))/(distanceToSecondaryToTheFifth);
    double yDerivativeSecondaryTerm = (-3.0*massParameter*currentLocation(1))/(distanceToSecondaryToTheFifth);


    double xEquilibriumPartialX = (1.0 -termRelatedToPrimary - termRelatedToSecondary)
            + currentLocation(0) * (-xDerivativePrimaryTerm - xDerivativeSecondaryTerm)
            + massParameter * ( -xDerivativePrimaryTerm - xDerivativeSecondaryTerm) + xDerivativeSecondaryTerm;
    double xEquilibriumPartialY = currentLocation(0)*(-yDerivativePrimaryTerm - yDerivativeSecondaryTerm)
            + massParameter * (-yDerivativePrimaryTerm - yDerivativeSecondaryTerm) + yDerivativeSecondaryTerm;
    double yEquilibriumPartialX = currentLocation(1) * (-xDerivativePrimaryTerm - xDerivativeSecondaryTerm);
    double yEquilibriumPartialY = (1.0 -termRelatedToPrimary - termRelatedToSecondary)
            + currentLocation(1)*(-yDerivativePrimaryTerm - yDerivativeSecondaryTerm);
    double inverseDeterminant = 1.0 /(xEquilibriumPartialX*yEquilibriumPartialY-xEquilibriumPartialY*yEquilibriumPartialX);
    Eigen::MatrixXd inverseJacobianMatrix (2,2);
    inverseJacobianMatrix << yEquilibriumPartialY, -1.0* xEquilibriumPartialY,
                             -1.0*yEquilibriumPartialX, xEquilibriumPartialX;

    inverseJacobian = inverseDeterminant * inverseJacobianMatrix;

    return inverseJacobian;
}

Eigen::Vector2d applyMultivariateRootFinding( const int librationPointNr, const Eigen::Vector2d initialEquilibrium,
                                              const double alpha, const double thrustAcceleration, const double massParameter, double maxDeviationFromEquilibrium,
                                              const int maxNumberOfIterations) {

    Eigen::Vector2d currentGuess = initialEquilibrium;
    Eigen::Vector2d deviationFromEquilibrium = computeDeviation(librationPointNr, initialEquilibrium, alpha, thrustAcceleration, massParameter );
    double deviationNormFromEquilibrium = sqrt(deviationFromEquilibrium(0)*deviationFromEquilibrium(0) + deviationFromEquilibrium(1)*deviationFromEquilibrium(1));
    Eigen::MatrixXd inverseJacobian = Eigen::MatrixXd::Zero(2,2);
    Eigen::Vector2d updateVector;
    Eigen::Vector2d updatedGuess;
    int numberOfIterations = 1;
    Eigen::Vector2d currentDeviation;

    while (deviationNormFromEquilibrium > maxDeviationFromEquilibrium ) {

        if (numberOfIterations > maxNumberOfIterations) {
            std::cout << "Maximum number of iterations exceeded" << std::endl;
            return Eigen::VectorXd::Ones(2);
        }

        currentDeviation = computeDeviation(librationPointNr, currentGuess, alpha, thrustAcceleration, massParameter);
        inverseJacobian  = computeJacobian(librationPointNr, currentGuess, massParameter);
        updateVector     = -1.0*inverseJacobian*currentDeviation;
        updatedGuess     = currentGuess + updateVector;

        //std::cout << "Inverse Jacobian " << inverseJacobian << std::endl;

        //Update the equilibrium solution
        deviationFromEquilibrium = computeDeviation(librationPointNr, updatedGuess, alpha, thrustAcceleration, massParameter);
        deviationNormFromEquilibrium = sqrt(deviationFromEquilibrium(0)*deviationFromEquilibrium(0) + deviationFromEquilibrium(1)*deviationFromEquilibrium(1));
        //std::cout << "Position deviation from equilibrium: " << deviationNormFromEquilibrium << std::endl;

        numberOfIterations += 1;
        currentGuess = updatedGuess;       

    }

    return currentGuess;


}
