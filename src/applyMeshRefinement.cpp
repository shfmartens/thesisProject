#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/LU>
#include <iostream>
#include <math.h>

#include "Tudat/Astrodynamics/Gravitation/librationPoint.h"
#include "Tudat/Astrodynamics/BasicAstrodynamics/celestialBodyConstants.h"
#include "stateDerivativeModelAugmented.h"
#include "propagateOrbitAugmented.h"
#include <Eigen/Eigenvalues>
#include "applyCollocation.h"
#include "applyLineSearchAttenuation.h"
#include "applyMeshRefinement.h"

Eigen::VectorXd computeProcedureTimeShifts(Eigen::VectorXd collocationDesignVectorInitial, Eigen::VectorXd collocationDesignVectorFinal, const int numberOfCollocationPoints)
{
    // initialize outputVector //
    Eigen::VectorXd outputVector(numberOfCollocationPoints); outputVector.setZero();
    Eigen::VectorXd differenceVector = collocationDesignVectorFinal - collocationDesignVectorInitial;


    for(int i = 0; i < numberOfCollocationPoints-1;i++)
    {
        Eigen::VectorXd localDifferenceVector = differenceVector.segment(19*i,26);
        outputVector(i) = localDifferenceVector(6);
        outputVector(i+1) = localDifferenceVector(25);

    }

    return outputVector;

}

Eigen::VectorXd computeStateViaPolynomialInterpolation(const Eigen::MatrixXd segmentOddStates, const Eigen::MatrixXd segmentOddStateDerivatives, const double deltaTime, const double interpolationTime)
{
    // Define relevant variables
    Eigen::VectorXd outputVector(6);                                             outputVector.setZero();
    Eigen::MatrixXd oddTimesMatrix(8,8);                                         oddTimesMatrix.setZero();
    retrieveLegendreGaussLobattoConstaints("oddTimesMatrix", oddTimesMatrix);

    // determine the polynomialCoefficient Matrix
    Eigen::MatrixXd dynamicsMatrix(6,8);                    dynamicsMatrix.setZero();
    Eigen::MatrixXd polynomialCoefficientMatrix(6,8);       polynomialCoefficientMatrix.setZero();
    dynamicsMatrix.block(0,0,6,4) = segmentOddStates;
    dynamicsMatrix.block(0,4,6,4) = deltaTime * segmentOddStateDerivatives;

    polynomialCoefficientMatrix = dynamicsMatrix * oddTimesMatrix.inverse();

    // construct the interpolationVector
    Eigen::VectorXd interpolationTimeVector(8); interpolationTimeVector.setZero();
    interpolationTimeVector(0) = pow(interpolationTime, 0.0);
    interpolationTimeVector(1) = pow(interpolationTime, 1.0);
    interpolationTimeVector(2) = pow(interpolationTime, 2.0);
    interpolationTimeVector(3) = pow(interpolationTime, 3.0);
    interpolationTimeVector(4) = pow(interpolationTime, 4.0);
    interpolationTimeVector(5) = pow(interpolationTime, 5.0);
    interpolationTimeVector(6) = pow(interpolationTime, 6.0);
    interpolationTimeVector(7) = pow(interpolationTime, 7.0);

    outputVector = polynomialCoefficientMatrix * interpolationTimeVector;

    return outputVector;

}

Eigen::VectorXcd computeStateViaPolynomialInterpolationComplex(const Eigen::MatrixXcd segmentOddStates, const Eigen::MatrixXcd segmentOddStateDerivatives, std::complex<double> deltaTime, std::complex<double> interpolationTime)
{
    // Define relevant variables
    Eigen::VectorXcd outputVector(6);                                             outputVector.setZero();
    Eigen::MatrixXd oddTimesMatrix(8,8);                                         oddTimesMatrix.setZero();
    retrieveLegendreGaussLobattoConstaints("oddTimesMatrix", oddTimesMatrix);
    Eigen::MatrixXcd oddTimesMatrixComplex = oddTimesMatrix;

    // determine the polynomialCoefficient Matrix
    Eigen::MatrixXcd dynamicsMatrix(6,8);                    dynamicsMatrix.setZero();
    Eigen::MatrixXcd polynomialCoefficientMatrix(6,8);       polynomialCoefficientMatrix.setZero();
    dynamicsMatrix.block(0,0,6,4) = segmentOddStates;
    dynamicsMatrix.block(0,4,6,4) = deltaTime * segmentOddStateDerivatives;

    polynomialCoefficientMatrix = dynamicsMatrix * oddTimesMatrixComplex.inverse();

    // construct the interpolationVector
    Eigen::VectorXcd interpolationTimeVector(8); interpolationTimeVector.setZero();
    interpolationTimeVector(0) = std::complex<double>(1.0,0.0);
    interpolationTimeVector(1) = pow(interpolationTime, 1.0);
    interpolationTimeVector(2) = pow(interpolationTime, 2.0);
    interpolationTimeVector(3) = pow(interpolationTime, 3.0);
    interpolationTimeVector(4) = pow(interpolationTime, 4.0);
    interpolationTimeVector(5) = pow(interpolationTime, 5.0);
    interpolationTimeVector(6) = pow(interpolationTime, 6.0);
    interpolationTimeVector(7) = pow(interpolationTime, 7.0);

    outputVector = polynomialCoefficientMatrix * interpolationTimeVector;

    return outputVector;
}

void computeInterpolationSegmentsAndTimes(const Eigen::VectorXd newNodeTimes, const Eigen::VectorXd currentNodeTimes, const int numberOfCollocationPoints, Eigen::VectorXd& newOddPointTimesDimensional, Eigen::VectorXd& newOddPointTimesNormalized, Eigen::VectorXd& oddPointsSegments, Eigen::VectorXd& newTimeIntervals )
{

    // delcare relevant variables
    int numberOfSegments = numberOfCollocationPoints - 1;
    int numberOfOddPoints = 3*numberOfSegments+1;
    Eigen::MatrixXd legendreGaussLobattoTimes(7,1); legendreGaussLobattoTimes.setZero();
    retrieveLegendreGaussLobattoConstaints("nodeTimes", legendreGaussLobattoTimes);
    Eigen::MatrixXd timeIntervalMatrix(numberOfCollocationPoints-1,2); timeIntervalMatrix.setZero();

    // Compute the dimensional times of all odd points  and fill the dimensionalTimeVectorMatrix
    for(int i = 0; i < numberOfSegments; i++)
        {
            // determine segment times of the new mesh
            Eigen::VectorXd segmentTimes = Eigen::VectorXd::Zero(4);
            double initialSegmentTime = newNodeTimes(i);
            double finalSegmentTime = newNodeTimes(i+1);
            double segmentTimeInterval = finalSegmentTime - initialSegmentTime;

            // determine segment times of the old mesh
            double currentInitialSegmentTime = currentNodeTimes(i);
            double currentFinalSegmentTime = currentNodeTimes(i+1);

            timeIntervalMatrix(i,0) = currentInitialSegmentTime;
            timeIntervalMatrix(i,1) = currentFinalSegmentTime;

            for(int j = 0; j < 4; j++)
            {
                if (j == 0)
                {
                    segmentTimes(j) = initialSegmentTime;
                } else if (j == 1)
                {
                    segmentTimes(j) = initialSegmentTime+legendreGaussLobattoTimes(2,0)*segmentTimeInterval;

                } else if (j == 2)
                {
                    segmentTimes(j) = initialSegmentTime+legendreGaussLobattoTimes(4,0)*segmentTimeInterval;
                }
                else
                {
                    segmentTimes(j) = finalSegmentTime;
                }
            }

            newOddPointTimesDimensional.segment(3*i,4) = segmentTimes;
            newTimeIntervals(i) = segmentTimeInterval;

        }

        // determine on which segment of old mesh the new odd points lie and compute their nondimensional time on that mesh
        for (int i = 0; i < numberOfOddPoints; i++)
        {
            double currentOddPointTime = newOddPointTimesDimensional(i);

            for(int j = 0; j < numberOfSegments; j++)
            {
                if (currentOddPointTime >= timeIntervalMatrix(j,0) and currentOddPointTime <= timeIntervalMatrix(j,1) )
                {
                    oddPointsSegments(i) = j;

                    double timeInterval = timeIntervalMatrix(j,1) - timeIntervalMatrix(j,0);
                    double initialIntervalTime = timeIntervalMatrix(j,0);
                    newOddPointTimesNormalized(i) = (currentOddPointTime - initialIntervalTime)/timeInterval;

                }
            }
        }

}

void computeNewMesh(const Eigen::VectorXd collocationDesignVector,  const Eigen::VectorXd thrustAndMassParameters, const Eigen::VectorXd nodeTimes, const Eigen::VectorXd newNodeTimes, const int numberOfCollocationPoints, Eigen::VectorXd& newDesignVector)
{
    // declare initial variables
    int numberOfSegments = numberOfCollocationPoints - 1;
    int numberOfOddPoints = 3*numberOfSegments+1;
    Eigen::MatrixXd oddStates(6*numberOfSegments,4);                       oddStates.setZero();
    Eigen::MatrixXd oddStateDerivatives(6*numberOfSegments,4);             oddStateDerivatives.setZero();
    Eigen::VectorXd timeIntervals(numberOfSegments);                       timeIntervals.setZero();
    Eigen::VectorXd nodeTimesRedundant(numberOfCollocationPoints);         nodeTimesRedundant.setZero();
    Eigen::VectorXd newOddPointTimes(numberOfOddPoints);                   newOddPointTimes.setZero();

    Eigen::VectorXd segmentVector(numberOfOddPoints);                      segmentVector.setZero();
    Eigen::VectorXd newOddPointTimesNormalized(numberOfOddPoints);         newOddPointTimesNormalized.setZero();
    Eigen::VectorXd newOddPointTimesDimensional(numberOfOddPoints);        newOddPointTimesDimensional.setZero();
    Eigen::VectorXd newTimeIntervals(numberOfSegments);                    newTimeIntervals.setZero();



    // compute the information needed for interpolation (oddStates, OddStateDerivatives, TimeIntervals)
    computeTimeIntervals(collocationDesignVector, numberOfCollocationPoints, timeIntervals, nodeTimesRedundant);

    computeSegmentProperties(collocationDesignVector, thrustAndMassParameters, numberOfCollocationPoints, oddStates, oddStateDerivatives, timeIntervals );

    // compute the times on the new mesh for interior and node points and determine which segment is needed to interpolate the new states
    computeInterpolationSegmentsAndTimes(newNodeTimes, nodeTimes, numberOfCollocationPoints, newOddPointTimesDimensional, newOddPointTimesNormalized, segmentVector, newTimeIntervals );

    // perform interpolation and the new times to construct the new mesh
    bool designVecIncludesThrustAndMass = false;
    if (collocationDesignVector.rows() == 11*numberOfOddPoints )
    {
        designVecIncludesThrustAndMass = true;
    }

    int startingSegment = 0;

    for (int i = 0; i < numberOfOddPoints; i++)
    {
        auto segmentNumber = static_cast<int>(segmentVector(i));
        double interpolationTime = newOddPointTimesNormalized(i);
        double oddPointTime = newOddPointTimesDimensional(i);
        double segmentTimeInterval = timeIntervals(segmentNumber);

        Eigen::MatrixXd segmentOddStates = oddStates.block(6*segmentNumber,0,6,4);
        Eigen::MatrixXd segmentOddStateDerivatives = oddStateDerivatives.block(6*segmentNumber,0,6,4);

        Eigen::VectorXd interpolatedOddPoint = computeStateViaPolynomialInterpolation(segmentOddStates, segmentOddStateDerivatives, segmentTimeInterval, interpolationTime);

        // store in design vector, check on format to decide how to store it! if 11 it is isimplem if not nodes include time while interior points do not.

        if (designVecIncludesThrustAndMass == true)
        {
            newDesignVector.segment(i*11,6) = interpolatedOddPoint;
            newDesignVector.segment(i*11+6,4) = thrustAndMassParameters;
            newDesignVector(i*11+10) = oddPointTime;

        } else
        {
            int numberOfElements = 0;
            if(i % 3 == 0)
            {
                numberOfElements =  7;
                newDesignVector.segment(startingSegment, 6) = interpolatedOddPoint;
                newDesignVector(startingSegment+6) = oddPointTime;

            } else
            {
                numberOfElements = 6;
                newDesignVector.segment(startingSegment, 6) = interpolatedOddPoint;

            }

            startingSegment = startingSegment+numberOfElements;

        }

    }

}

void computeTimeIntervals(const Eigen::VectorXd collocationDesignVector, const int numberOfCollocationPoints, Eigen::VectorXd& timeIntervals, Eigen::VectorXd& nodeTimes)
{

    bool designVecIncludesThrustAndMass = false;
    if (collocationDesignVector.rows() == 11*(3*(numberOfCollocationPoints-1)+1) )
    {
        designVecIncludesThrustAndMass = true;
    }

    for(int i = 0; i < (numberOfCollocationPoints -1); i++)
    {
        if (designVecIncludesThrustAndMass == true)
        {
            Eigen::VectorXd localCollocationDesignVector = collocationDesignVector.segment(33*i,44);
            double initialNodeTime = localCollocationDesignVector(10);
            double finalNodeTime = localCollocationDesignVector(43);
            timeIntervals(i) = finalNodeTime - initialNodeTime;

            nodeTimes(i) = initialNodeTime;
            nodeTimes(i+1) = finalNodeTime;

        } else
        {
            Eigen::VectorXd localCollocationDesignVector = collocationDesignVector.segment(19*i,26);
            double initialNodeTime = localCollocationDesignVector(6);
            double finalNodeTime = localCollocationDesignVector(25);
            timeIntervals(i) = finalNodeTime - initialNodeTime;

            nodeTimes(i) = initialNodeTime;
            nodeTimes(i+1) = finalNodeTime;

        }


    }


}

void computeSegmentDerivatives(Eigen::MatrixXd& segmentDerivatives, Eigen::MatrixXd& oddStates,Eigen::MatrixXd& oddStateDerivatives, const Eigen::VectorXd timeIntervals, const int numberOfCollocationPoints)
{
    // Compute the oddTimes Matrices inverse
    Eigen::MatrixXd oddTimesMatrix(8,8); oddTimesMatrix.setZero();
    Eigen::MatrixXd oddTimesMatrixINV(8,8); oddTimesMatrixINV.setZero();

    retrieveLegendreGaussLobattoConstaints("oddTimesMatrix", oddTimesMatrix);

    oddTimesMatrixINV = oddTimesMatrix.inverse();


    // per segment, create the dynamics matrix using MCOLL and TOM NOTATION
    for(int i = 0; i < ( numberOfCollocationPoints-1 ); i++)
    {
        // select the odd states and state derivatives of the particular segment
        Eigen::MatrixXd segmentOddPoints = oddStates.block(6*i,0,6,4);
        Eigen::MatrixXd segmentOddPointDerivatives = oddStateDerivatives.block(6*i,0,6,4);

        // Assemble the dynamics MAtrices
        Eigen::MatrixXd dynamicsMatrix = Eigen::MatrixXd::Zero(6,8);

        dynamicsMatrix.block(0,0,6,4) = segmentOddPoints;
        dynamicsMatrix.block(0,4,6,4) = timeIntervals(i)*segmentOddPointDerivatives;

        // compute the factorial 7 divided by time interval
        double factorialSeven = 7.0*6.0*5.0*4.0*3.0*2.0*1.0;
        double timeIntervalToThePowerSeventh = pow( timeIntervals(i), 7.0);

        segmentDerivatives.block(0,i,6,1) = (factorialSeven / timeIntervalToThePowerSeventh ) * dynamicsMatrix * oddTimesMatrixINV.block(0,7,8,1) ;

    }

}

void computeSegmentProperties(const Eigen::VectorXd collocationDesignVector, const Eigen::VectorXd thrustAndMassParameters, const int numberOfCollocationPoints, Eigen::MatrixXd& oddStates, Eigen::MatrixXd& oddStateDervatives, Eigen::VectorXd& timeIntervals)
{
    bool designVecIncludesThrustAndMass = false;
    if (collocationDesignVector.rows() == 11*(3*(numberOfCollocationPoints-1)+1) )
    {
        designVecIncludesThrustAndMass = true;
    }

    if(designVecIncludesThrustAndMass == false)
    {
        for(int i = 0; i < numberOfCollocationPoints -1; i++)
        {
            // select local State Vector
            Eigen::VectorXd segmentDesignVector = collocationDesignVector.segment(19*i,26);
            timeIntervals(i) = segmentDesignVector(25) - segmentDesignVector(6);

            Eigen::VectorXd localSegmentState = Eigen::VectorXd::Zero(6);
            for(int j = 0; j < 4; j++)
            {
                // extract the local state
                if (j == 0)
                {
                    localSegmentState = segmentDesignVector.segment(0,6);

                } else
                {
                    localSegmentState = segmentDesignVector.segment(6*j+1,6);

                }

                // compute the localStateDerivative;
                Eigen::VectorXd fullLocalState = Eigen::VectorXd::Zero(10);
                fullLocalState.segment(0,6) = localSegmentState;
                fullLocalState.segment(6,4) = thrustAndMassParameters;

                Eigen::MatrixXd stateDerivativeInclSTM = computeStateDerivativeAugmented(0.0, getFullInitialStateAugmented(fullLocalState) );

                // store the state and state derivative in the oddStates and oddStateDerivative matrices
                oddStates.block(6*i,j,6,1)           = localSegmentState;
                oddStateDervatives.block(6*i,j,6,1)  = stateDerivativeInclSTM.block(0,0,6,1);
            }
        }
    }
    else {

        for(int i = 0; i < numberOfCollocationPoints -1; i++)
        {
            // select local State Vector
            Eigen::VectorXd segmentDesignVector = collocationDesignVector.segment(33*i,44);
            timeIntervals(i) = segmentDesignVector(43) - segmentDesignVector(10);

            Eigen::VectorXd localSegmentState = Eigen::VectorXd::Zero(6);
            for(int j = 0; j < 4; j++)
            {

                localSegmentState = segmentDesignVector.segment(j*11,11);

                // compute the localStateDerivative;
                Eigen::VectorXd fullLocalState = Eigen::VectorXd::Zero(10);
                fullLocalState.segment(0,6) = localSegmentState;
                fullLocalState.segment(6,4) = thrustAndMassParameters;

                Eigen::MatrixXd stateDerivativeInclSTM = computeStateDerivativeAugmented(0.0, getFullInitialStateAugmented(fullLocalState) );

                // store the state and state derivative in the oddStates and oddStateDerivative matrices
                oddStates.block(6*i,j,6,1)           = localSegmentState;
                oddStateDervatives.block(6*i,j,6,1)  = stateDerivativeInclSTM.block(0,0,6,1);
            }
        }
    }
}

void computeSegmentErrors(Eigen::VectorXd collocationDesignVector, const Eigen::VectorXd thrustAndMassParameters, int numberOfCollocationPoints, Eigen::VectorXd& segmentErrors, Eigen::VectorXd& eightOrderDerivatives, const double computableConstant )
{
    // declare relevant variables
    int numberOfSegments = numberOfCollocationPoints - 1;
    Eigen::VectorXd outputVector = Eigen::VectorXd::Zero(numberOfCollocationPoints-1);
    Eigen::MatrixXd oddStates = Eigen::MatrixXd::Zero(6*numberOfSegments,4);
    Eigen::MatrixXd oddStateDerivatives = Eigen::MatrixXd::Zero(6*numberOfSegments,4);
    Eigen::VectorXd timeIntervals = Eigen::VectorXd::Zero(numberOfSegments);
    Eigen::MatrixXd segmentDerivatives = Eigen::MatrixXd::Zero(6,numberOfSegments);

    computeSegmentProperties(collocationDesignVector, thrustAndMassParameters, numberOfCollocationPoints, oddStates, oddStateDerivatives, timeIntervals );


    computeSegmentDerivatives(segmentDerivatives, oddStates, oddStateDerivatives, timeIntervals, numberOfCollocationPoints);

    Eigen::VectorXd eightOrderDerivativeMagnitudes = Eigen::VectorXd::Zero(numberOfSegments);
    for(int i = 0; i < numberOfSegments; i++)
    {
        double eightOrderDerivative = 0.0;

        if (i == 0)
        {
            double deltaTime = timeIntervals(0)+timeIntervals(1);
            Eigen::VectorXd derivativeDifference = ( segmentDerivatives.block(0,0,6,1) - segmentDerivatives.block(0,1,6,1) ).cwiseAbs();
            eightOrderDerivative =( ( 2.0 / deltaTime )*derivativeDifference ).maxCoeff();


        } else if (i == (numberOfSegments-1))
        {

            double deltaTime = timeIntervals(i-1)+timeIntervals(i);
            Eigen::VectorXd derivativeDifference = ( segmentDerivatives.block(0,i,6,1) - segmentDerivatives.block(0,i-1,6,1) ).cwiseAbs();
            eightOrderDerivative =( ( 2.0 / deltaTime )*derivativeDifference ).maxCoeff();


        } else {

            double deltaTime1 = timeIntervals(i-1) + timeIntervals(i);
            double deltaTime2 = timeIntervals(i) + timeIntervals(i+1);
            Eigen::VectorXd derivativeDifference1 = ( segmentDerivatives.block(0,i-1,6,1) - segmentDerivatives.block(0,i,6,1) ).cwiseAbs();
            Eigen::VectorXd derivativeDifference2 = ( segmentDerivatives.block(0,i+1,6,1) - segmentDerivatives.block(0,i,6,1) ).cwiseAbs();
            Eigen::VectorXd derivativeDifferenceSum = (derivativeDifference1/deltaTime1 + derivativeDifference2/deltaTime2);
            eightOrderDerivative = derivativeDifferenceSum.maxCoeff();

        }

        eightOrderDerivativeMagnitudes(i) = eightOrderDerivative;
    }


    // compute the errors per segment
    Eigen::VectorXd timeIntervalsSquared = timeIntervals.cwiseProduct(timeIntervals);
    Eigen::VectorXd timeIntervalsToThePowerFourth = timeIntervalsSquared.cwiseProduct(timeIntervalsSquared);
    Eigen::VectorXd timeIntervalsToThePowerEigth = timeIntervalsToThePowerFourth.cwiseProduct(timeIntervalsToThePowerFourth);

    outputVector = computableConstant * timeIntervalsToThePowerEigth.cwiseProduct(eightOrderDerivativeMagnitudes);

    segmentErrors = outputVector;
    eightOrderDerivatives = eightOrderDerivativeMagnitudes;

}

void applyMeshRefinement(Eigen::MatrixXd& collocationDesignVector, Eigen::VectorXd& segmentErrorDistribution, const Eigen::VectorXd thrustAndMassParameters, int numberOfCollocationPoints )
{

    int numberOfSegments = numberOfCollocationPoints - 1;
    Eigen::VectorXd currentCollocationDesignVector(collocationDesignVector.rows());  currentCollocationDesignVector.setZero();
    Eigen::VectorXd segmentErrors(numberOfSegments);         segmentErrors.setZero();
    Eigen::VectorXd eightOrderDerivatives(numberOfSegments); eightOrderDerivatives.setZero();
    Eigen::VectorXd timeIntervals(numberOfSegments);         timeIntervals.setZero();
    Eigen::VectorXd nodeTimes(numberOfCollocationPoints);    nodeTimes.setZero();
    Eigen::VectorXd meshIntegral(numberOfSegments);          meshIntegral.setZero();
    Eigen::VectorXd newNodeTimes(numberOfCollocationPoints); newNodeTimes.setZero();

    currentCollocationDesignVector = collocationDesignVector.block(0,0,collocationDesignVector.rows(),1);

    computeSegmentErrors(currentCollocationDesignVector, thrustAndMassParameters, numberOfCollocationPoints, segmentErrors, eightOrderDerivatives );

    computeTimeIntervals(currentCollocationDesignVector, numberOfCollocationPoints, timeIntervals, nodeTimes);


    // compute the mesh integral
    double integralValue = 0.0;
    for(int i =0; i < numberOfCollocationPoints-1;i++)
    {
        double segmentIntegralValue = timeIntervals(i)*pow(eightOrderDerivatives(i), 1.0/8.0);
        integralValue = integralValue + segmentIntegralValue;
        meshIntegral(i) = integralValue;


    }

    // compute the new node times!
    double deltaIntegralNew = meshIntegral((meshIntegral.rows()-1)) / numberOfSegments;

    for(int i = 0; i < numberOfCollocationPoints; i++)
    {
        if (i == 0 or i == (numberOfCollocationPoints-1) )
        {
            newNodeTimes(i) = nodeTimes(i);
        } else
        {
            double newIntegral = i * deltaIntegralNew;
            double deltaTime = (newIntegral - meshIntegral(i-1) ) / (pow(eightOrderDerivatives(i-1), 1.0/8.0));
            newNodeTimes(i) = nodeTimes(i) + deltaTime;

        }
    }


    // interpolate the polynomials
    Eigen::VectorXd newDesignVector(currentCollocationDesignVector.rows()); newDesignVector.setZero();
    computeNewMesh( currentCollocationDesignVector, thrustAndMassParameters, nodeTimes, newNodeTimes, numberOfCollocationPoints, newDesignVector);

    // output the desired quantities
    collocationDesignVector.block(0,0,collocationDesignVector.rows(),1) = newDesignVector;
    segmentErrorDistribution = segmentErrors;


}


