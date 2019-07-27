#include <cmath>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>


#include "Tudat/Astrodynamics/BasicAstrodynamics/celestialBodyConstants.h"
#include "Tudat/Astrodynamics/Gravitation/librationPoint.h"
#include "Tudat/Astrodynamics/Gravitation/jacobiEnergy.h"

#include "applyDifferentialCorrection.h"
#include "computeDifferentialCorrection.h"
#include "propagateOrbit.h"
#include "createEquilibriumLocations.h"

Eigen::MatrixXd computeJacobian(Eigen::Vector2d initialGuess, double acceleration, double alpha, double massParameter)
{

Eigen::MatrixXd jacobian(2,2);
jacobian.setZero();

// Compute r13 and r23 and their derivatives
double xDistance1 = initialGuess(0) + massParameter;
double xDistance2 = initialGuess(0) - 1 + massParameter;
double yDistance =  initialGuess(1);

double xDistance1Squared = xDistance1 * xDistance1;
double xDistance2Squared = xDistance2 * xDistance2;
double yDistanceSquared = yDistance * yDistance;

double distanceToPrimary = sqrt(xDistance1Squared + yDistanceSquared);
double distanceToSecondary = sqrt(xDistance2Squared + yDistanceSquared);

double distanceToPrimaryCubed = distanceToPrimary * distanceToPrimary * distanceToPrimary;
double distanceToSecondaryCubed = distanceToSecondary * distanceToSecondary * distanceToSecondary;

double distanceToPrimaryFifth = distanceToPrimaryCubed * distanceToPrimary * distanceToPrimary;
double distanceToSecondaryFifth = distanceToSecondaryCubed * distanceToSecondary * distanceToSecondary;

double termRelatedToPrimary = (1.0 - massParameter) / distanceToPrimaryCubed;
double termRelatedToSecondary = massParameter / distanceToSecondaryCubed;

double termRelatedToPrimaryDerivative = -3.0* ( (1.0 - massParameter) / distanceToPrimaryFifth );
double termRelatedToSecondaryDerivative = -3.0 * ( ( massParameter / distanceToSecondaryFifth ) );

// Compute the Jacobian components
double XPositionXDerivative = (1.0 - termRelatedToPrimary - termRelatedToSecondary)
                              + initialGuess(0)* (-termRelatedToPrimaryDerivative * xDistance1 - termRelatedToSecondaryDerivative * xDistance2 )
                              + massParameter * ( -termRelatedToPrimaryDerivative * xDistance1 - termRelatedToSecondaryDerivative * xDistance2 )
                              + termRelatedToSecondaryDerivative * xDistance2;

double XPositionYDerivative = initialGuess(0)* (termRelatedToPrimaryDerivative * yDistance + termRelatedToSecondaryDerivative * yDistance )
                              + massParameter * ( termRelatedToPrimaryDerivative * yDistance + termRelatedToSecondaryDerivative * yDistance )
                              + termRelatedToSecondaryDerivative * yDistance;

double YPositionXDerivative = initialGuess(1)* (-termRelatedToPrimaryDerivative * xDistance1 - termRelatedToSecondaryDerivative * xDistance2 );

double YPositionYDerivative = (1.0 - termRelatedToPrimary - termRelatedToSecondary) + initialGuess(1)* (-termRelatedToPrimaryDerivative * yDistance - termRelatedToSecondaryDerivative *yDistance );



//std::cout << "initialGuess: \n" << initialGuess << std::endl
//          << "XPositionXDerivative: \n" << XPositionXDerivative << std::endl
//          << "XPositionYDerivative: \n" << XPositionYDerivative << std::endl
//          << "YPositionXDerivative: \n" << YPositionXDerivative << std::endl
//          << "YPositionYDerivative: \n" << YPositionYDerivative << std::endl;



jacobian << XPositionXDerivative, XPositionYDerivative,
            YPositionXDerivative, YPositionYDerivative;

return jacobian;

}

Eigen::Vector2d computeDeviation(Eigen::Vector2d equilibriumGuess, double acceleration, double alpha, double massParameter)
{
    Eigen::Vector2d deviationVector;
    deviationVector.setZero();

    // Compute r13 and r23
    double xDistance1 = equilibriumGuess(0) + massParameter;
    double xDistance2 = equilibriumGuess(0) - 1 + massParameter;
    double yDistance = equilibriumGuess(1);

    double xDistance1Squared = xDistance1 * xDistance1;
    double xDistance2Squared = xDistance2 * xDistance2;
    double yDistanceSquared = yDistance * yDistance;

    double distanceToPrimary = sqrt(xDistance1Squared + yDistanceSquared);
    double distanceToSecondary = sqrt(xDistance2Squared + yDistanceSquared);

    double distanceToPrimaryCubed = distanceToPrimary * distanceToPrimary * distanceToPrimary;
    double distanceToSecondaryCubed = distanceToSecondary * distanceToSecondary * distanceToSecondary;

    double termRelatedToPrimary = (1.0 - massParameter) / distanceToPrimaryCubed;
    double termRelatedToSecondary = massParameter / distanceToSecondaryCubed;

    deviationVector(0) = equilibriumGuess(0) * ( 1.0 - termRelatedToPrimary - termRelatedToSecondary)
                        + massParameter * ( -termRelatedToPrimary - termRelatedToSecondary ) + termRelatedToSecondary
                        + acceleration * std::cos( alpha * tudat::mathematical_constants::PI/180.0);

    deviationVector(1) = equilibriumGuess(1) * ( 1.0 - termRelatedToPrimary - termRelatedToSecondary)
                        + acceleration * std::sin( alpha * tudat::mathematical_constants::PI/180.0);

    return deviationVector;

}

Eigen::Vector3d applyMultivariateRootFinding( const int librationPointNr, const Eigen::Vector2d initialEquilibrium,
                                              const double alpha, const double thrustAcceleration, bool& iterationsReached, const double massParameter, double maxDeviationFromEquilibrium,
                                              const int maxNumberOfIterations) {

    Eigen::Vector2d initialGuess;
    Eigen::Vector2d correctedGuess;
    Eigen::Vector3d convergedGuess;
    Eigen::Vector2d deviationVector;
    Eigen::Vector2d constraintVector;

    initialGuess.setZero();
    correctedGuess.setZero();
    convergedGuess.setZero();
    deviationVector.setZero();

    initialGuess = initialEquilibrium;
    deviationVector = computeDeviation(initialEquilibrium, thrustAcceleration, alpha, massParameter);
    constraintVector = -1.0*deviationVector;

    int numberOfIterations = 0;
    while (deviationVector.norm() > maxDeviationFromEquilibrium)
    {

        if (numberOfIterations > maxNumberOfIterations)
        {
            // std::cout << "Maxmimum number of iterations exceeded" << std::endl;
            iterationsReached = true;
            return Eigen::Vector3d::Zero(3);
        }

        Eigen::MatrixXd updateMatrix(2,2);
        updateMatrix.setZero();
        updateMatrix = computeJacobian(initialGuess, thrustAcceleration, alpha, massParameter);

        correctedGuess = initialGuess + updateMatrix.inverse() * constraintVector;


        deviationVector = computeDeviation(correctedGuess, thrustAcceleration, alpha, massParameter);
        constraintVector = -1.0 * deviationVector;


        initialGuess = correctedGuess;
        numberOfIterations++;

    }

    convergedGuess.segment(0,2) = correctedGuess;
    convergedGuess(2) = numberOfIterations;
    iterationsReached = false;

    return convergedGuess;

}
