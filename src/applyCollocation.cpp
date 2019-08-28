#include <cmath>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Dense>
#include <map>

#include <chrono>

#include <boost/function.hpp>

#include "Tudat/Astrodynamics/BasicAstrodynamics/celestialBodyConstants.h"
#include "Tudat/Astrodynamics/Gravitation/librationPoint.h"
#include "Tudat/Astrodynamics/Gravitation/jacobiEnergy.h"
#include "Tudat/InputOutput/basicInputOutput.h"

#include "createLowThrustInitialConditions.h"
#include "applyCollocation.h"
#include "stateDerivativeModel.h"
#include "stateDerivativeModelAugmented.h"
#include "propagateOrbitAugmented.h"

void computeOddPoints(const Eigen::VectorXd initialStateVector, Eigen::MatrixXd& internalPointsMatrix, int numberOfCollocationPoints, const double massParameter)
{
        Eigen::MatrixXd nodeTimesNormalized;
        retrieveLegendreGaussLobattoConstaints("nodeTimes",nodeTimesNormalized);


        for(int i = 0; i < (numberOfCollocationPoints-1); i++ )
        {
            // add initial and final node to the internal points matrix
           internalPointsMatrix.block(i*11,0,11,1) = initialStateVector.segment(i*11,11);
           internalPointsMatrix.block(i*11,3,11,1) = initialStateVector.segment((i+1)*11,11);

           // compute the times of the interior points
           double initialSegmentTime = initialStateVector(i*11+10);
           double finalSegmentTime = initialStateVector((i+1)*11+10);
           Eigen::VectorXd segmentNodeTimes = convertNodeTimes( nodeTimesNormalized, initialSegmentTime, finalSegmentTime);

           double timeInteriorPoint1 = segmentNodeTimes(2);
           double timeInteriorPoint2 = segmentNodeTimes(4);

           // Compute the interior point states via propagatedAugmentedToFinalCondition
           std::map< double, Eigen::VectorXd > stateHistory;
           std::pair< Eigen::MatrixXd, double > stateVectorInclSTMAndTimePoint1 = propagateOrbitAugmentedToFinalCondition(
                       getFullInitialStateAugmented( initialStateVector.segment(i*11,10) ), massParameter, timeInteriorPoint1, 1, stateHistory, -1, initialSegmentTime );

           Eigen::MatrixXd stateVectorInclSTMPoint1 = stateVectorInclSTMAndTimePoint1.first;
           double actualTimeInteriorPoint1 = stateVectorInclSTMAndTimePoint1.second;
           Eigen::VectorXd stateInteriorPoint1 = stateVectorInclSTMPoint1.block(0,0,10,1);

           std::pair< Eigen::MatrixXd, double > stateVectorInclSTMAndTimePoint2 = propagateOrbitAugmentedToFinalCondition(
                       getFullInitialStateAugmented( initialStateVector.segment(i*11,10) ), massParameter, timeInteriorPoint2, 1, stateHistory, -1, initialSegmentTime );

           Eigen::MatrixXd stateVectorInclSTMPoint2 = stateVectorInclSTMAndTimePoint2.first;
           double actualTimeInteriorPoint2 = stateVectorInclSTMAndTimePoint2.second;
           Eigen::VectorXd stateInteriorPoint2 = stateVectorInclSTMPoint2.block(0,0,10,1);

           // Store the points in the Matrix
           internalPointsMatrix.block(i*11,1,10,1) = stateInteriorPoint1;
           internalPointsMatrix(i*11+10,1) = actualTimeInteriorPoint1;
           internalPointsMatrix.block(i*11,2,10,1) = stateInteriorPoint2;
           internalPointsMatrix(i*11+10,2) = actualTimeInteriorPoint2;


        }


}

Eigen::VectorXd convertNodeTimes(Eigen::MatrixXd nodeTimesNormalized, double lowerBound, double upperBound)
{
    Eigen::VectorXd convertedNodeTimes = Eigen::VectorXd::Zero(7);

    double segmentDuration = upperBound - lowerBound;

    convertedNodeTimes(0) = lowerBound;
    convertedNodeTimes(6) = upperBound;

    for(int i = 1; i < 6; i++)
    {
        convertedNodeTimes(i) = lowerBound + nodeTimesNormalized(i)*segmentDuration;
    }

    return convertedNodeTimes;
}

void retrieveLegendreGaussLobattoConstaints(const std::string desiredQuantity, Eigen::MatrixXd& outputMatrix)
{
    if(desiredQuantity == "nodeTimes")
    {
        Eigen::VectorXd nodeTimeVector = Eigen::VectorXd::Zero(7);
        nodeTimeVector(0) = 0.0;
        nodeTimeVector(1) = 0.084888051860717;
        nodeTimeVector(2) = 0.265575603264643;
        nodeTimeVector(3) = 0.5;
        nodeTimeVector(4) = 0.734424396735357;
        nodeTimeVector(5) = 0.915111948139284;
        nodeTimeVector(6) = 1.0;

        outputMatrix = nodeTimeVector;
    }
}


Eigen::VectorXd applyCollocation(const Eigen::MatrixXd initialCollocationGuess, const double massParameter, const int numberOfCollocationPoints, Eigen::VectorXd& collocatedGuess,
                                                         const double maxPositionDeviationFromPeriodicOrbit, const double maxVelocityDeviationFromPeriodicOrbit, const double maxPeriodDeviationFromPeriodicOrbit, const int maxNumberOfIterations)
{
    // initialize Variables
    Eigen::VectorXd outputVector = Eigen::VectorXd(25);
    int numberOfCorrections = 0;







    // Compute variables for outputVector and collocatedGuess

    collocatedGuess.segment(0,11*(numberOfCollocationPoints-1))  = initialCollocationGuess.block(0,0,11*(numberOfCollocationPoints-1),1);
    collocatedGuess.segment(11*(numberOfCollocationPoints-1),11) = initialCollocationGuess.block(11*(numberOfCollocationPoints-2),3,11,1);

    Eigen::VectorXd  initialCondition = collocatedGuess.segment(0,10);
    Eigen::VectorXd  finalCondition = collocatedGuess.segment(11*(numberOfCollocationPoints-1),10);

    double orbitalPeriod = collocatedGuess(11*(numberOfCollocationPoints-1)+10) - collocatedGuess(10);

    double hamiltonianInitialCondition  = computeHamiltonian( massParameter, initialCondition);
    double hamiltonianEndState          = computeHamiltonian( massParameter, finalCondition  );

    outputVector.segment(0,10) = initialCondition;
    outputVector(10) = orbitalPeriod;
    outputVector(11) = hamiltonianInitialCondition;
    outputVector.segment(12,10) = finalCondition;
    outputVector(22) = collocatedGuess(11*(numberOfCollocationPoints-1) + 10);
    outputVector(23) = hamiltonianEndState;
    outputVector(24) = numberOfCorrections;

    return outputVector;

}

