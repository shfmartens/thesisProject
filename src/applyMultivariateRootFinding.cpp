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

Eigen::MatrixXd computeJacobian(Eigen::Vector2d currentGuess, const double massParameter)
{
    Eigen::MatrixXd jacobianMatrix(2,2);
    jacobianMatrix.setZero();

    double xDistancePrimary = currentGuess(0) + massParameter;
    double xDistanceSecondary = currentGuess(0) -1.0 + massParameter;
    double yDistance = currentGuess(1);

    double r13 = sqrt(xDistancePrimary*xDistancePrimary + yDistance*yDistance);
    double r23 = sqrt(xDistanceSecondary*xDistanceSecondary+ yDistance*yDistance);

    double r13Cubed = r13*r13*r13;
    double r23Cubed = r23*r23*r23;

    double r13Fifth = r13Cubed*r13*r13;
    double r23Fifth = r23Cubed*r23*r23;

    double primaryTerm = (1.0 - massParameter)/r13Cubed;
    double secondaryTerm = massParameter/r23Cubed;

    double primaryTermDerivative = 3.0 * (1.0-massParameter)/r13Fifth;
    double secondaryTermDerivative = 3.0 * (massParameter)/r23Fifth;

    double partialf1partialX = (1.0 - primaryTerm - secondaryTerm)
            + (currentGuess(0) + massParameter) * (primaryTermDerivative * xDistancePrimary + secondaryTermDerivative * xDistanceSecondary)
            - secondaryTermDerivative * xDistanceSecondary;

    double partialf1partialY =  (currentGuess(0) + massParameter) * (primaryTermDerivative * yDistance + secondaryTermDerivative * yDistance)
            - secondaryTermDerivative * yDistance;

    double partialf2partialX = currentGuess(1)*(primaryTermDerivative*xDistancePrimary + secondaryTermDerivative*xDistanceSecondary);

    double partialf2partialY =  (1.0 - primaryTerm - secondaryTerm) +
                                + currentGuess(1)*(primaryTermDerivative*yDistance + secondaryTermDerivative*yDistance);

    jacobianMatrix(0.0) = partialf1partialX;
    jacobianMatrix(0,1) = partialf1partialY;
    jacobianMatrix(1,0) = partialf2partialX;
    jacobianMatrix(1,1) = partialf2partialY;

    return jacobianMatrix;


}

Eigen::Vector2d computeConstraintVector(Eigen::Vector2d currentGuess, const double thrustAcceleration, const double alpha, const double massParameter)
{
    Eigen::Vector2d constraintVector;
    constraintVector.setZero();

    double xDistancePrimary = currentGuess(0) + massParameter;
    double xDistanceSecondary = currentGuess(0) -1.0 + massParameter;
    double yDistance = currentGuess(1);

    double r13 = sqrt(xDistancePrimary*xDistancePrimary + yDistance*yDistance);
    double r23 = sqrt(xDistanceSecondary*xDistanceSecondary+ yDistance*yDistance);

    double r13Cubed = r13*r13*r13;
    double r23Cubed = r23*r23*r23;

    double primaryTerm = (1.0 - massParameter)/r13Cubed;
    double secondaryTerm = massParameter/r23Cubed;


    double f1 = currentGuess(0)*(1.0 - primaryTerm - secondaryTerm ) + massParameter * (-primaryTerm - secondaryTerm)
                + secondaryTerm + thrustAcceleration * std::cos(alpha * tudat::mathematical_constants::PI/180.0);
    double f2 = currentGuess(1)*(1.0 - primaryTerm - secondaryTerm ) + thrustAcceleration * std::sin(alpha * tudat::mathematical_constants::PI/180.0);

    constraintVector(0) = -f1;
    constraintVector(1) = -f2;

    return constraintVector;

}

Eigen::Vector3d applyMultivariateRootFinding( const Eigen::Vector2d initialEquilibrium,
                                              const double thrustAcceleration, const double alpha, const double massParameter, const double relaxationParameter, const double maxDeviationFromEquilibrium,
                                              const int maxNumberOfIterations)
{

    Eigen::Vector3d ConvergedGuessWithIterations;

    Eigen::Vector2d correctedGuess;
    correctedGuess.setZero();

    Eigen::Vector2d constraintVector = computeConstraintVector(initialEquilibrium, thrustAcceleration, alpha, massParameter);
    Eigen::Vector2d currentGuess = initialEquilibrium;

    ConvergedGuessWithIterations.setZero();

    int numberOfIterations = 0;

    while( constraintVector.norm() > maxDeviationFromEquilibrium )
    {
        if (numberOfIterations > maxNumberOfIterations)
        {
            std::cout << "MaxNumberOfIterations Reached, MV rootfinder did not converge within maxNumberOfIterations!" << std::endl;
            ConvergedGuessWithIterations.segment(0,2) = Eigen::Vector2d::Zero();
            ConvergedGuessWithIterations(2) = maxNumberOfIterations+999;
            return ConvergedGuessWithIterations;
        }

        Eigen::MatrixXd updateMatrix = computeJacobian( currentGuess, massParameter);



        Eigen::Vector2d correction = updateMatrix.inverse() * constraintVector;

        correctedGuess = currentGuess + relaxationParameter * correction;

//        if (std::abs(alpha) < 0.2 && numberOfIterations < 3)
//        {
//            std::cout << "currentGuess: \n" << currentGuess << std::endl
//                      << "\nupdateMatrix \n" << updateMatrix << std::endl
//                      << "updateMatrix.inverse(): \n" << updateMatrix.inverse() << std::endl
//                      << "correctionApplied: \n " << correction << std::endl;
//        }

        constraintVector = computeConstraintVector(correctedGuess, thrustAcceleration, alpha, massParameter);

        currentGuess = correctedGuess;

        numberOfIterations++;
    }

//    if (std::abs(alpha) < 0.2 )
//    {

//                  std::cout << "\ncorrectedGuess: \n" << correctedGuess << std::endl
//                  << "iterations: " << numberOfIterations << std::endl;
//    }

    ConvergedGuessWithIterations.segment(0,2) = currentGuess;
    ConvergedGuessWithIterations(2) = numberOfIterations;

    return ConvergedGuessWithIterations;


}
