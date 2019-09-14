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

    // compute the information needed for interpolation (oddStates, OddStateDerivatives, TimeIntervals)
    computeTimeIntervals(collocationDesignVector, numberOfCollocationPoints, timeIntervals, nodeTimesRedundant);

    computeSegmentProperties(collocationDesignVector, thrustAndMassParameters, numberOfCollocationPoints, oddStates, oddStateDerivatives, timeIntervals );

    // Compute the non-dimensional times of all odd points
    for(int i = 0; i < numberOfSegments; i++)
        {
            Eigen::VectorXd segmentTimes = Eigen::VectorXd::Zero(4);
            for(int j = 0; j < 4; j++)
            {
                if (j == 0)
                {
                    segmentTimes(j) = newNodeTimes(i);
                } else if (j == 3)
                {
                    segmentTimes(j) = newNodeTimes(i+1);
                }
            }

            newOddPointTimes.segment(3*i,4) = segmentTimes;

        }

    std::cout << "newNodeTimes: \n" << newNodeTimes << std::endl;
    std::cout << "newOddPointTimes: \n" << newOddPointTimes << std::endl;





    // loop per node/interior point
        // determine the new dimensional time of the node point
            // if node: take it directly from newNodeTimes
            // if interior point: determine delta_t of the segment, use oddTimes to compute the exact time
            // save in vector which holds all node times!
        // determine in which segment the node lies (0 = segment 1 etc.) via if loops
        // determine the non-dimensional time within the segment
        // interpolate via a function which has inputs: oddStates and StateDerivatives of the segment, nondim Time and DeltaTime (do not make same mistake as last time!)
        // store new node in the newDesignVector
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

void applyMeshRefinement(Eigen::MatrixXd& collocationDesignVector, const Eigen::VectorXd thrustAndMassParameters, int numberOfCollocationPoints )
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



}


