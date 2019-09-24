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
#include "applyPredictionCorrection.h"
#include "applyLineSearchAttenuation.h"
#include "stateDerivativeModel.h"
#include "stateDerivativeModelAugmented.h"
#include "propagateOrbitAugmented.h"
#include "computeCollocationCorrection.h"
#include "applyMeshRefinement.h"
#include "interpolatePolynomials.h"


void  writeTrajectoryErrorDataToFile(const int numberOfCollocationPoints, const Eigen::VectorXd fullPeriodDeviations, const Eigen::VectorXd defectVectorMS, const Eigen::VectorXd collocatedDefects, const Eigen::VectorXd integrationErrors, const int magnitudeNoiseOffset, const double amplitude )
{

    // define the matrices to be written to text files
    Eigen::VectorXd fullPeriodDeviationNorms(2);
    Eigen::VectorXd shootingDeviationNorms(2*(numberOfCollocationPoints-1));
    Eigen::VectorXd collocationDeviationNorms(2*(numberOfCollocationPoints-1));

    for(int i = 0; i < (numberOfCollocationPoints-1); i++)
    {
        Eigen::VectorXd localShootingDeviations    = defectVectorMS.segment(i*11,11);
        shootingDeviationNorms(2*i) = (localShootingDeviations.segment(0,3)).norm();
        shootingDeviationNorms(2*i+1) = (localShootingDeviations.segment(3,3)).norm();

        Eigen::VectorXd localcollocationDeviations = collocatedDefects.segment(i*18,18);
        collocationDeviationNorms(2*i) = sqrt( ( localcollocationDeviations.segment(0,3) ).squaredNorm() +
                                         ( localcollocationDeviations.segment(6,3) ).squaredNorm() +
                                         ( localcollocationDeviations.segment(12,3) ).squaredNorm() );
        collocationDeviationNorms(2*i+1) = sqrt( ( localcollocationDeviations.segment(3,3) ).squaredNorm() +
                                         ( localcollocationDeviations.segment(9,3) ).squaredNorm() +
                                         ( localcollocationDeviations.segment(15,3) ).squaredNorm() );

    }

    fullPeriodDeviationNorms(0) = (fullPeriodDeviations.segment(0,3)).norm();
    fullPeriodDeviationNorms(1) = (fullPeriodDeviations.segment(3,3)).norm();

    // save output to file, use typeOfInput as variable!
     std::string directoryString = "../data/raw/collocation/mesh_effect";

     std::string fileNameStringFullPeriod;
     std::string fileNameStringDeviationsMS;
     std::string fileNameStringDeviationsColloc;
     std::string fileNameStringDeviationsErrors;


     fileNameStringFullPeriod = (std::to_string(amplitude) + "_" + std::to_string(numberOfCollocationPoints) +  "_" + std::to_string(magnitudeNoiseOffset) +  "_fullPeriodDeviations.txt");
     fileNameStringDeviationsMS = (std::to_string(amplitude) + "_" + std::to_string(numberOfCollocationPoints) +  "_" + std::to_string(magnitudeNoiseOffset) + "_shootingDeviations.txt");
     fileNameStringDeviationsColloc = ( std::to_string(amplitude) + "_" + std::to_string(numberOfCollocationPoints) +  "_" + std::to_string(magnitudeNoiseOffset) +  "_collocationDeviations.txt");
     fileNameStringDeviationsErrors = ( std::to_string(amplitude) + "_" + std::to_string(numberOfCollocationPoints) +  "_" + std::to_string(magnitudeNoiseOffset) +  "_collocationErrors.txt");




     tudat::input_output::writeMatrixToFile( fullPeriodDeviationNorms, fileNameStringFullPeriod, 16, directoryString);
     tudat::input_output::writeMatrixToFile( shootingDeviationNorms, fileNameStringDeviationsMS, 16, directoryString);
     tudat::input_output::writeMatrixToFile( collocationDeviationNorms, fileNameStringDeviationsColloc, 16, directoryString);
     tudat::input_output::writeMatrixToFile( integrationErrors, fileNameStringDeviationsErrors, 16, directoryString);





}

Eigen::VectorXd rewriteOddPointsToVector(const Eigen::MatrixXd& oddNodesMatrix, const int numberOfCollocationPoints)
{
    int numberOfStates = 3*(numberOfCollocationPoints-1)+1;
    Eigen::VectorXd outputVector = Eigen::VectorXd::Zero(11*numberOfStates);

    for(int i = 0; i < numberOfCollocationPoints-1; i ++)
    {
        Eigen::MatrixXd segmentMatrix = oddNodesMatrix.block(i*11,0,11,4);
        Eigen::VectorXd segmentConvertedToVector = Eigen::VectorXd(44);

        for(int j = 0; j < 4; j++)
        {
            segmentConvertedToVector.segment(j*11,11) = segmentMatrix.block(0,j,11,1);
        }

        outputVector.segment(33*i,44) = segmentConvertedToVector;
    }

    return outputVector;
}

void shiftTimeOfConvergedCollocatedGuess(const Eigen::MatrixXd collocationDesignVector, Eigen::VectorXd& collocatedGuess, Eigen::VectorXd& collocatedNodes, const int numberOfCollocationPoints, Eigen::VectorXd thrustAndMassParameters)
{
    // extract the node times for gauss lobato strategy
    Eigen::MatrixXd nodeTimes = Eigen::MatrixXd::Zero(7,1);
    retrieveLegendreGaussLobattoConstaints("nodeTimes", nodeTimes );
    double deltaTime = collocationDesignVector(6);

    // per segment
    int nodeNumber = 1;

    for(int i = 0; i < (numberOfCollocationPoints - 1); i++)
    {

        Eigen::VectorXd segmentDesignVector = collocationDesignVector.block(19*i,0,26,1);
        double initialSegmentTime = segmentDesignVector(6)-deltaTime;
        double timeInterval = segmentDesignVector(25)-segmentDesignVector(6);

        Eigen::VectorXd newNodeVector = Eigen::VectorXd::Zero(11);
        Eigen::VectorXd localDesignVector = Eigen::VectorXd::Zero(6);

        nodeNumber--;

        for(int j = 0; j < 4; j++)
        {
            if (j == 0)
            {
                localDesignVector = segmentDesignVector.segment(0,6);

            } else
            {
                localDesignVector = segmentDesignVector.segment(j*6+1,6);
            }

            Eigen::VectorXd newNodeVector = Eigen::VectorXd::Zero(11);
            newNodeVector.segment(0,6) = localDesignVector;
            newNodeVector.segment(6,4) = thrustAndMassParameters;
            newNodeVector(10) = initialSegmentTime + timeInterval*nodeTimes(2*j,0);

            collocatedGuess.segment(nodeNumber*11,11) = newNodeVector;

            if (j == 0 )
            {
                collocatedNodes.segment(i*11,11) = newNodeVector;
            }

            if (j == 3 and i == numberOfCollocationPoints-2)
            {
                collocatedNodes.segment((i+1)*11,11) = newNodeVector;
            }

            nodeNumber++;


        }
    }

}

void propagateAndSaveCollocationProcedure(const Eigen::MatrixXd oddPointsInput, Eigen::VectorXd timeIntervals, Eigen::VectorXd thrustAndMassParameters, const int numberOfCollocationPoints, const int typeOfInput, const double massParameter)
{
    // declare relevant variables for the function and set all to zero or empty them
    Eigen::VectorXd inputStateVector(11*numberOfCollocationPoints);
    Eigen::VectorXd defectVector(11*(numberOfCollocationPoints-1));
    Eigen::MatrixXd propagatedStatesInclSTM(10*(numberOfCollocationPoints-1),11);
    std::map< double, Eigen::VectorXd > stateHistory;
    Eigen::VectorXd deviationNorms(7);


    inputStateVector.setZero(); defectVector.setZero();
    propagatedStatesInclSTM.setZero(); stateHistory.clear();
    deviationNorms.setZero();

    // rewrite the initialGuess to the desired format
    if (typeOfInput == 0 or typeOfInput == 1)\
    {
        for(int i = 0; i < (numberOfCollocationPoints-1); i++)
        {
            inputStateVector.segment(i*11,11) = oddPointsInput.block(i*11,0,11,1);

            if (i == numberOfCollocationPoints-2)
            {
                inputStateVector.segment((i+1)*11,11) = oddPointsInput.block(i*11,3,11,1);
            }
        }

    } else{

        for(int i = 0; i < (numberOfCollocationPoints-1); i++)
        {

            Eigen::VectorXd segmentNodes = oddPointsInput.block(i*19,0,26,1);

            inputStateVector.segment(i*11,6) = segmentNodes.segment(0,6);
            inputStateVector.segment(i*11+6,4) = thrustAndMassParameters;
            inputStateVector(i*11+10) = segmentNodes(6);


            if (i == numberOfCollocationPoints-2)
            {
                inputStateVector.segment((i+1)*11,6) = segmentNodes.segment(19,6);
                inputStateVector.segment((i+1)*11+6,4) = thrustAndMassParameters;
                inputStateVector((i+1)*11+10) = segmentNodes(25);

            }
        }
    }

    // compute deviations via multiple shooting and the respective deviation norms
    computeOrbitDeviations(inputStateVector, numberOfCollocationPoints, propagatedStatesInclSTM, defectVector, stateHistory, massParameter);


    // compute Deviation Norms
    Eigen::VectorXd positionDeviationTotal (3*(numberOfCollocationPoints-1));
    Eigen::VectorXd velocityDeviationTotal (3*(numberOfCollocationPoints-1));
    Eigen::VectorXd timeDeviationTotal ((numberOfCollocationPoints-1));
    Eigen::VectorXd positionDeviationInternal (3*(numberOfCollocationPoints-2));
    Eigen::VectorXd velocityDeviationInternal (3*(numberOfCollocationPoints-2));
    Eigen::VectorXd positionDeviationExternal(3);
    Eigen::VectorXd velocityDeviationExternal(3);

    for(int i = 0; i < (numberOfCollocationPoints-1); i++)
    {
        Eigen::VectorXd localDefectVector = defectVector.segment(i*11,11);
        positionDeviationTotal.segment(3*i,3) = localDefectVector.segment(0,3);
        velocityDeviationTotal.segment(3*i,3) = localDefectVector.segment(3,3);
        timeDeviationTotal(i) = localDefectVector(10);

        if(i < (numberOfCollocationPoints -2 ))
        {
            positionDeviationInternal.segment(3*i,3) = localDefectVector.segment(0,3);
            velocityDeviationInternal.segment(3*i,3) = localDefectVector.segment(3,3);

        }
        if (i == (numberOfCollocationPoints -2 ))
        {
            positionDeviationExternal = localDefectVector.segment(0,3);
            velocityDeviationExternal = localDefectVector.segment(3,3);
        }

    }

    deviationNorms(0) = positionDeviationTotal.norm();
    deviationNorms(1) = velocityDeviationTotal.norm();
    deviationNorms(2) = positionDeviationInternal.norm();
    deviationNorms(3) = velocityDeviationInternal.norm();
    deviationNorms(4) = positionDeviationExternal.norm();
    deviationNorms(5) = velocityDeviationExternal.norm();
    deviationNorms(6) = timeDeviationTotal.norm();



    // save output to file, use typeOfInput as variable!
     std::string directoryString = "../data/raw/collocation/";

     std::string fileNameStringStateVector;
     std::string fileNameStringStateHistory;
     std::string fileNameStringDeviations;
     std::string fileNameStringPropagatedStates;


     fileNameStringStateVector = (std::to_string(typeOfInput) +"_stateVectors.txt");
     fileNameStringStateHistory = (std::to_string(typeOfInput) +"_stateHistory.txt");
     fileNameStringDeviations = (std::to_string(typeOfInput) +"_deviations.txt");
     fileNameStringPropagatedStates = (std::to_string(typeOfInput) +"_propagatedStates.txt");

     Eigen::VectorXd propagatedStates = propagatedStatesInclSTM.block(0,0,10*(numberOfCollocationPoints-1),1);


     tudat::input_output::writeMatrixToFile( inputStateVector, fileNameStringStateVector, 16, directoryString);
     tudat::input_output::writeDataMapToTextFile( stateHistory, fileNameStringStateHistory, directoryString );
     tudat::input_output::writeMatrixToFile( deviationNorms, fileNameStringDeviations, 16, directoryString);
     tudat::input_output::writeMatrixToFile( propagatedStates, fileNameStringPropagatedStates, 16, directoryString);


}

Eigen::VectorXd computeCollocationDeviationNorms(const Eigen::MatrixXd collocationDefectVector, const Eigen::MatrixXd collocationDesignVector, const int numberOfCollocationPoints)
{
    Eigen::VectorXd deviationNorms(5);
    deviationNorms.setZero();

    int numberOfDefects = 3*(numberOfCollocationPoints -1);
    Eigen::VectorXd positionDeviations(3*numberOfDefects);
    Eigen::VectorXd velocityDeviations(3*numberOfDefects);
    Eigen::VectorXd periodicityDeviation(6);
    Eigen::VectorXd periodicityPositionDeviation(3);
    Eigen::VectorXd periodicityVelocityDeviation(3);
    double phaseConstraint = 0.0;


    int nodeNumber = 0;
    positionDeviations.setZero(); velocityDeviations.setZero(); periodicityPositionDeviation.setZero(); periodicityVelocityDeviation.setZero();
    for(int i = 0; i < (numberOfCollocationPoints-1); i++)
    {

        Eigen::VectorXd segmentDeviationVector = collocationDefectVector.block(18*i,0,18,1);

        for(int j = 0; j < 3; j++)
        {
            Eigen::VectorXd nodeDefects(6);
            nodeDefects = segmentDeviationVector.block(j*6,0,6,1);

            positionDeviations.segment(3*nodeNumber,3) = nodeDefects.segment(0,3);
            velocityDeviations.segment(3*nodeNumber,3) = nodeDefects.segment(3,3);

            nodeNumber++;

        }



    }

    periodicityDeviation = collocationDesignVector.block(0,0,6,1)-collocationDesignVector.block(19*(numberOfCollocationPoints-1),0,6,1);
    periodicityPositionDeviation = periodicityDeviation.segment(0,3);
    periodicityVelocityDeviation = periodicityDeviation.segment(3,3);
    phaseConstraint = collocationDefectVector(collocationDefectVector.rows()-1,0);

    deviationNorms(0) = positionDeviations.norm();
    deviationNorms(1) = velocityDeviations.norm();
    deviationNorms(2) = periodicityPositionDeviation.norm();
    deviationNorms(3) = periodicityVelocityDeviation.norm();
    deviationNorms(4) = phaseConstraint;


    return deviationNorms;

}

Eigen::MatrixXd evaluateVectorFields(const Eigen::MatrixXd initialCollocationGuess, const int numberOfCollocationPoints)
{
    Eigen::MatrixXd outputMatrix = Eigen::MatrixXd::Zero(10*(numberOfCollocationPoints-1),4);
    for(int i = 0; i < (numberOfCollocationPoints - 1); i++)
    {
        for(int j = 0; j < 4; j++)
        {
            Eigen::MatrixXd initialStateVectorInclSTM(10,11);
            initialStateVectorInclSTM.block(0,0,10,1) = initialCollocationGuess.block(11*i,j,10,1);
            initialStateVectorInclSTM.block(0,1,10,10).setIdentity();
            Eigen::MatrixXd localStateDerivativeInclSTM = computeStateDerivativeAugmented(0.0, initialStateVectorInclSTM);

            outputMatrix.block(10*i,j,10,1) = localStateDerivativeInclSTM.block(0,0,10,1);
        }
    }

    return outputMatrix;
}

void extractDurationAndDynamicsFromInput(const Eigen::MatrixXd initialCollocationGuess, const Eigen::MatrixXd initialCollocationDerivatives, const int numberOfCollocationPoints, Eigen::MatrixXd& oddPointsDynamics, Eigen::MatrixXd& oddPointsDynamicsDerivatives,  Eigen::VectorXd& timeIntervals)
{

    for(int i = 0; i < (numberOfCollocationPoints -1); i++ )
    {
        timeIntervals(i) = initialCollocationGuess(i*11+10,3)-initialCollocationGuess(i*11+10,0);
        oddPointsDynamics.block(i*6,0,6,4) = initialCollocationGuess.block(i*11,0,6,4);
        oddPointsDynamicsDerivatives.block(i*6,0,6,4) = initialCollocationDerivatives.block(i*10,0,6,4);
    }





}

void computeOddPoints(const Eigen::VectorXd initialStateVector, Eigen::MatrixXd& internalPointsMatrix, int numberOfCollocationPoints, const double massParameter, const bool firstCollocationGuess)
{
        Eigen::MatrixXd nodeTimesNormalized;
        retrieveLegendreGaussLobattoConstaints("nodeTimes",nodeTimesNormalized);

        if(firstCollocationGuess == true)
        {
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

         if(firstCollocationGuess == false ){
            for(int i = 0; i < (numberOfCollocationPoints-1); i++ )
               {
                  Eigen::VectorXd segmentNodes = initialStateVector.segment(i*33,44);
                  for(int j = 0; j < 4; j++)
                  {
                      internalPointsMatrix.block(i*11,j,11,1) = segmentNodes.segment(j*11,11);
                  }
               }

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

    if(desiredQuantity == "oddTimesMatrix")
    {
        // Retrieve nodeVector
        Eigen::VectorXd tau = Eigen::VectorXd::Zero(7);
        tau(0) = 0.0;
        tau(1) = 0.084888051860717;
        tau(2) = 0.265575603264643;
        tau(3) = 0.5;
        tau(4) = 0.734424396735357;
        tau(5) = 0.915111948139284;
        tau(6) = 1.0;

        Eigen::MatrixXd oddTimesMatrix = Eigen::MatrixXd(8,8);
        Eigen::ArrayXd exponentVector = Eigen::ArrayXd::LinSpaced(8,0.0,7.0);
        for(int j = 0; j<4; j++)
        {
            for(int i = 0; i < 8; i++)
            {
                oddTimesMatrix(i,j) = std::pow(tau(2*j), exponentVector(i) );
            }

        }

        for(int j = 4; j<8; j++)
        {
            oddTimesMatrix(0,j) = 0.0;
            for(int i = 1; i < 8; i++)
            {
                oddTimesMatrix(i,j) = ( exponentVector(i) ) * std::pow(tau(2*(j-4)), exponentVector(i)-1 );
            }

        }

        outputMatrix = oddTimesMatrix;

        // optional redistribute oddTimexMatrix like Tom's notation into outputMatrix
//        outputMatrix.block(0,0,8,1) = oddTimesMatrix.block(0,0,8,1);
//        outputMatrix.block(0,1,8,1) = oddTimesMatrix.block(0,4,8,1);
//        outputMatrix.block(0,2,8,1) = oddTimesMatrix.block(0,1,8,1);
//        outputMatrix.block(0,3,8,1) = oddTimesMatrix.block(0,5,8,1);
//        outputMatrix.block(0,4,8,1) = oddTimesMatrix.block(0,2,8,1);
//        outputMatrix.block(0,5,8,1) = oddTimesMatrix.block(0,6,8,1);
//        outputMatrix.block(0,6,8,1) = oddTimesMatrix.block(0,3,8,1);
//        outputMatrix.block(0,7,8,1) = oddTimesMatrix.block(0,7,8,1);


    }

    if (desiredQuantity == "evenTimesMatrix")
    {
        // Retrieve nodeVector
        Eigen::VectorXd tau = Eigen::VectorXd::Zero(7);
        tau(0) = 0.0;
        tau(1) = 0.084888051860717;
        tau(2) = 0.265575603264643;
        tau(3) = 0.5;
        tau(4) = 0.734424396735357;
        tau(5) = 0.915111948139284;
        tau(6) = 1.0;

        Eigen::MatrixXd evenTimesMatrix = Eigen::MatrixXd(8,3);
        Eigen::ArrayXd exponentVector = Eigen::ArrayXd::LinSpaced(8,0.0,7.0);

        for(int j = 0; j<3; j++)
        {
            for(int i = 0; i < 8; i++)
            {
                evenTimesMatrix(i,j) = std::pow(tau((2*j)+1), exponentVector(i) );
            }

        }

        outputMatrix = evenTimesMatrix;

    }

    if (desiredQuantity == "evenTimesMatrixDerivative")
    {
        // Retrieve nodeVector
        Eigen::VectorXd tau = Eigen::VectorXd::Zero(7);
        tau(0) = 0.0;
        tau(1) = 0.084888051860717;
        tau(2) = 0.265575603264643;
        tau(3) = 0.5;
        tau(4) = 0.734424396735357;
        tau(5) = 0.915111948139284;
        tau(6) = 1.0;

        Eigen::MatrixXd evenTimesMatrixDerivative = Eigen::MatrixXd(8,3);
        Eigen::ArrayXd exponentVector = Eigen::ArrayXd::LinSpaced(8,0.0,7.0);

        for(int j = 0; j<3; j++)
        {
            evenTimesMatrixDerivative(0,j) = 0.0;
            for(int i = 1; i < 8; i++)
            {
                evenTimesMatrixDerivative(i,j) = ( exponentVector(i) ) * std::pow(tau((2*j)+1), exponentVector(i)-1 );
            }

        }

        outputMatrix = evenTimesMatrixDerivative;

    }

     if(desiredQuantity == "AConstants")
     {
         Eigen::MatrixXd AConstants = Eigen::MatrixXd::Zero(3,4);
         // temporary storage of coefficients for testing defect state approximation
         double ai1 =  0.618612232711785;  double ai21 = 0.334253095933642;   double ai31 = 0.0152679626438851;double aip1 = 0.0318667087106879;
         double aic =  0.141445282326366;  double ai2c = 0.358554717673634;   double ai3c = 0.358554717673634; double aipc = 0.141445282326366;
         double ai4 =  0.0318667087106879; double ai24 = 0.0152679626438851;  double ai34 = 0.334253095933642; double aip4 = 0.618612232711785;

         AConstants << ai1, ai21, ai31, aip1,
                         aic, ai2c, ai3c, aipc,
                         ai4, ai24, ai34, aip4;

         outputMatrix = AConstants;


     }

     if (desiredQuantity == "VConstants")
     {
         Eigen::MatrixXd VConstants = Eigen::MatrixXd::Zero(3,4);
         double vi1 =  0.0257387738427162; double vi21 = -0.0550098654524528; double vi31 = -0.0153026046503702; double vip1 = -0.00238759243962924;
         double vic =  0.00992317607754556;double vi2c = 0.0962835932121973;  double vi3c = -0.0962835932121973; double vipc = -0.00992317607754556;
         double vi4 = 0.00238759243962924; double vi24 = 0.0153026046503702;  double vi34 =  0.0550098654524528; double vip4 = -0.0257387738427162;

         VConstants << vi1, vi21, vi31, vip1,
                         vic, vi2c, vi3c, vipc,
                         vi4, vi24, vi34, vip4;

         outputMatrix = VConstants;

     }

     if (desiredQuantity == "BConstants")
     {
         Eigen::MatrixXd BConstants = Eigen::MatrixXd::Zero(3,4);
         double bi1 =   0.884260109348311; double bi21 = -0.823622559094327;  double bi31 = -0.0235465327970606; double bip1 = -0.0370910174569208;
         double bic =  0.0786488731947674; double bi2c =  0.800076026297266;  double bi3c = -0.800076026297266;  double bipc = -0.0786488731947674;
         double bi4 =  0.0370910174569208; double bi24 =  0.0235465327970606; double bi34 = 0.823622559094327;   double bip4 = -0.884260109348311;


         BConstants << bi1, bi21, bi31, bip1,
                         bic, bi2c, bi3c, bipc,
                         bi4, bi24, bi34, bip4;

         outputMatrix = BConstants;

     }

     if (desiredQuantity == "WConstants")
     {

         Eigen::MatrixXd WConstants = Eigen::MatrixXd::Zero(3,5);
         double wi1 =   0.0162213410652341; double wi11 =  0.138413023680783; double wi21 =  0.0971662045547156; double wi31 =  0.0185682012187242; double wip1 =  0.00274945307600086;
         double wic =  0.00483872966828888; double wi2c =  0.100138284831491; double wicc =  0.243809523809524;  double wi3c = 0.100138284831491;   double wipc =  0.00483872966828888;
         double wi4 =  0.00274945307600086; double wi24 = 0.0185682012187242; double wi34 =  0.0971662045547156; double wi44 = 0.138413023680783;   double wip4 =  0.0162213410652341;

         WConstants << wi1, wi11, wi21, wi31, wip1,
                         wic, wi2c, wicc, wi3c, wipc,
                         wi4, wi24, wi34, wi44, wip4;

         outputMatrix = WConstants;

     }


     if(desiredQuantity == "weightingMatrixEvenStates")
     {
         Eigen::MatrixXd weightingMatrix(3,3);
         weightingMatrix.setIdentity();
         weightingMatrix(0,0) = 1.0*0.276826047361566;
         weightingMatrix(1,1) = 1.0*0.487619047619048;
         weightingMatrix(2,2) = 1.0*0.276826047361566;

        outputMatrix = weightingMatrix;


     }

}


void computeCollocationDefects(Eigen::MatrixXd& collocationDefectVector, Eigen::MatrixXd& collocationDesignVector, const Eigen::MatrixXd oddStates, const Eigen::MatrixXd oddStatesDerivatives, Eigen::VectorXd timeIntervals, Eigen::VectorXd thrustAndMassParameters, const int numberOfCollocationPoints, const double initialTime, const int continuationIndex, const Eigen::VectorXd previousDesignVector)
{
    // Load relevant constants
    Eigen::MatrixXd oddTimesMatrix(8,8);            Eigen::MatrixXd evenTimesMatrix(8,3);
    Eigen::MatrixXd evenTimesMatrixDerivative(8,3); Eigen::MatrixXd weightingMatrixEvenStates(3,3);
    Eigen::MatrixXd AConstants(3,4);                Eigen::MatrixXd VConstants(3,4);
    Eigen::MatrixXd BConstants(3,4);                Eigen::MatrixXd WConstants(3,5);

    retrieveLegendreGaussLobattoConstaints("oddTimesMatrix", oddTimesMatrix);
    retrieveLegendreGaussLobattoConstaints("evenTimesMatrix", evenTimesMatrix);
    retrieveLegendreGaussLobattoConstaints("evenTimesMatrixDerivative", evenTimesMatrixDerivative);
    retrieveLegendreGaussLobattoConstaints("weightingMatrixEvenStates", weightingMatrixEvenStates);
    retrieveLegendreGaussLobattoConstaints("AConstants", AConstants);
    retrieveLegendreGaussLobattoConstaints("VConstants", VConstants);
    retrieveLegendreGaussLobattoConstaints("BConstants", BConstants);
    retrieveLegendreGaussLobattoConstaints("WConstants", WConstants);

    for(int i = 0; i < (numberOfCollocationPoints -1); i++)
    {
        Eigen::VectorXd localCollocationDefectVector(18,1);
        Eigen::VectorXd localCollocationDesignVector(19,1);
        Eigen::VectorXd localCollocationDesignVectorFinal(26,1);

        localCollocationDefectVector.setZero();
        localCollocationDesignVector.setZero();
        localCollocationDesignVectorFinal.setZero();

        // Extract segment odd points, derivatives and time duration
        Eigen::MatrixXd localOddStates = oddStates.block(6*i,0,6,4);
        Eigen::MatrixXd localOddStatesDerivatives = oddStatesDerivatives.block(6*i,0,6,4);
        double deltaTime = timeIntervals(i);


        // Define local vectors to be calculated
        Eigen::MatrixXd polynomialCoefficientMatrix(6,8);
        Eigen::MatrixXd localEvenStates(6,3);
        Eigen::MatrixXd localEvenStatesTOM(6,3);
        Eigen::MatrixXd localEvenStatesDerivatives(6,3);
        Eigen::MatrixXd localEvenStatesDerivativesTOM(6,3);
        Eigen::MatrixXd localConstraints(6,3);

        // Compute C_{i}\A_{i}
        Eigen::MatrixXd dynamicsMatrix(6,8);

        dynamicsMatrix.block(0,0,6,1) = localOddStates.block(0,0,6,1);
        dynamicsMatrix.block(0,1,6,1) = localOddStates.block(0,1,6,1);
        dynamicsMatrix.block(0,2,6,1) = localOddStates.block(0,2,6,1);
        dynamicsMatrix.block(0,3,6,1) = localOddStates.block(0,3,6,1);

        dynamicsMatrix.block(0,4,6,1) = deltaTime * localOddStatesDerivatives.block(0,0,6,1);
        dynamicsMatrix.block(0,5,6,1) = deltaTime * localOddStatesDerivatives.block(0,1,6,1);
        dynamicsMatrix.block(0,6,6,1) = deltaTime * localOddStatesDerivatives.block(0,2,6,1);
        dynamicsMatrix.block(0,7,6,1) = deltaTime * localOddStatesDerivatives.block(0,3,6,1);

        polynomialCoefficientMatrix = dynamicsMatrix * oddTimesMatrix.inverse();


        // Compute defect States
        localEvenStates = polynomialCoefficientMatrix * evenTimesMatrix;

        localEvenStatesTOM.block(0,0,6,1) = AConstants(0,0) * localOddStates.block(0,0,6,1) + AConstants(0,1) * localOddStates.block(0,1,6,1) +
                                            AConstants(0,2) * localOddStates.block(0,2,6,1) + AConstants(0,3) * localOddStates.block(0,3,6,1) +
                              deltaTime * ( VConstants(0,0) * localOddStatesDerivatives.block(0,0,6,1) + VConstants(0,1) * localOddStatesDerivatives.block(0,1,6,1) +
                                            VConstants(0,2) * localOddStatesDerivatives.block(0,2,6,1) + VConstants(0,3) * localOddStatesDerivatives.block(0,3,6,1) );
        localEvenStatesTOM.block(0,1,6,1) = AConstants(1,0) * localOddStates.block(0,0,6,1) + AConstants(1,1) * localOddStates.block(0,1,6,1) +
                                            AConstants(1,2) * localOddStates.block(0,2,6,1) + AConstants(1,3) * localOddStates.block(0,3,6,1) +
                              deltaTime * ( VConstants(1,0) * localOddStatesDerivatives.block(0,0,6,1) + VConstants(1,1) * localOddStatesDerivatives.block(0,1,6,1) +
                                            VConstants(1,2) * localOddStatesDerivatives.block(0,2,6,1) + VConstants(1,3) * localOddStatesDerivatives.block(0,3,6,1) );
        localEvenStatesTOM.block(0,2,6,1) = AConstants(2,0) * localOddStates.block(0,0,6,1) + AConstants(2,1) * localOddStates.block(0,1,6,1) +
                                            AConstants(2,2) * localOddStates.block(0,2,6,1) + AConstants(2,3) * localOddStates.block(0,3,6,1) +
                              deltaTime * ( VConstants(2,0) * localOddStatesDerivatives.block(0,0,6,1) + VConstants(2,1) * localOddStatesDerivatives.block(0,1,6,1) +
                                            VConstants(2,2) * localOddStatesDerivatives.block(0,2,6,1) + VConstants(2,3) * localOddStatesDerivatives.block(0,3,6,1) );



        // compute local defect state derivatives
        for (int j = 0; j < 3; j++)
        {

            Eigen::VectorXd localEvenStateWithParameters(10);
            localEvenStateWithParameters.segment(0,6) = localEvenStates.block(0,j,6,1);
            localEvenStateWithParameters.segment(6,4) = thrustAndMassParameters;


            Eigen::MatrixXd localStateDerivative(10,11);
            localStateDerivative = computeStateDerivativeAugmented(0.0, getFullInitialStateAugmented(localEvenStateWithParameters));

            Eigen::VectorXd localFieldVector = localStateDerivative.block(0,0,6,1);
            localEvenStatesDerivatives.block(0,j,6,1) = localFieldVector;
        }

        localEvenStatesDerivativesTOM =   polynomialCoefficientMatrix * evenTimesMatrixDerivative;


        // compute the defect constraints
        Eigen::MatrixXd defectConstraints(6,3);
        Eigen::MatrixXd defectConstraintsALT(6,3);
        Eigen::MatrixXd intermediateMatrix(6,3);
        intermediateMatrix = ( polynomialCoefficientMatrix * evenTimesMatrixDerivative ) - localEvenStates;

        defectConstraintsALT =( ( polynomialCoefficientMatrix * evenTimesMatrixDerivative ) - localEvenStates)*weightingMatrixEvenStates;

        Eigen::VectorXd xsi1(6);
        Eigen::VectorXd xsi2(6);
        Eigen::VectorXd xsi3(6);

        xsi1 = BConstants(0,0)*localOddStates.block(0,0,6,1) + BConstants(0,1)*localOddStates.block(0,1,6,1) +
               BConstants(0,2)*localOddStates.block(0,2,6,1) + BConstants(0,3)*localOddStates.block(0,3,6,1) + deltaTime * (
               WConstants(0,0) * localOddStatesDerivatives.block(0,0,6,1) + WConstants(0,1) * localEvenStatesDerivatives.block(0,0,6,1)  +
               WConstants(0,2) * localOddStatesDerivatives.block(0,1,6,1) + WConstants(0,3) * localOddStatesDerivatives.block(0,2,6,1)  +
               WConstants(0,4) * localOddStatesDerivatives.block(0,3,6,1));

        xsi2 = BConstants(1,0)*localOddStates.block(0,0,6,1) + BConstants(1,1)*localOddStates.block(0,1,6,1) +
               BConstants(1,2)*localOddStates.block(0,2,6,1) + BConstants(1,3)*localOddStates.block(0,3,6,1) + deltaTime * (
               WConstants(1,0) * localOddStatesDerivatives.block(0,0,6,1) + WConstants(1,1) * localOddStatesDerivatives.block(0,1,6,1)  +
               WConstants(1,2) * localEvenStatesDerivatives.block(0,1,6,1) + WConstants(1,3) * localOddStatesDerivatives.block(0,2,6,1)  +
               WConstants(1,4) * localOddStatesDerivatives.block(0,3,6,1));

        xsi3 = BConstants(2,0)*localOddStates.block(0,0,6,1) + BConstants(2,1)*localOddStates.block(0,1,6,1) +
               BConstants(2,2)*localOddStates.block(0,2,6,1) + BConstants(2,3)*localOddStates.block(0,3,6,1) + deltaTime * (
               WConstants(2,0) * localOddStatesDerivatives.block(0,0,6,1) + WConstants(2,1) * localOddStatesDerivatives.block(0,1,6,1)  +
               WConstants(2,2) * localOddStatesDerivatives.block(0,2,6,1) + WConstants(2,3) * localEvenStatesDerivatives.block(0,2,6,1)  +
               WConstants(2,4) * localOddStatesDerivatives.block(0,3,6,1));

        defectConstraints.block(0,0,6,1) = xsi1;
        defectConstraints.block(0,1,6,1) = xsi2;
        defectConstraints.block(0,2,6,1) = xsi3;


        // construct localCollocationDefectVector and add to complete defectVector
        localCollocationDefectVector.segment(0,6) = xsi1;
        localCollocationDefectVector.segment(6,6) = xsi2;
        localCollocationDefectVector.segment(12,6) = xsi3;

        collocationDefectVector.block(18*i,0,18,1) = localCollocationDefectVector;

        // construct localCollocationDesignVector and add to complete DesignVector

        if (i < ( numberOfCollocationPoints - 2) )
        {
            double localNodeTime;
            if (i == 0)
            {
                localNodeTime = initialTime;
            } else
            {
                localNodeTime = initialTime + ( timeIntervals.segment(0,i) ).sum();
            }
            localCollocationDesignVector.segment(0,6) = localOddStates.block(0,0,6,1);
            localCollocationDesignVector(6) = localNodeTime;
            localCollocationDesignVector.segment(7,6) = localOddStates.block(0,1,6,1);
            localCollocationDesignVector.segment(13,6) = localOddStates.block(0,2,6,1);

            collocationDesignVector.block(19*i,0,19,1) = localCollocationDesignVector;

        } else
        {
            double localNodeTime = initialTime + ( timeIntervals.segment(0,i) ).sum();
            double localNodeTimeFinal= initialTime + ( timeIntervals.segment(0,i+1) ).sum();

            localCollocationDesignVectorFinal.segment(0,6) = localOddStates.block(0,0,6,1);
            localCollocationDesignVectorFinal(6) = localNodeTime;
            localCollocationDesignVectorFinal.segment(7,6) = localOddStates.block(0,1,6,1);
            localCollocationDesignVectorFinal.segment(13,6) = localOddStates.block(0,2,6,1);
            localCollocationDesignVectorFinal.segment(19,6) = localOddStates.block(0,3,6,1);
            localCollocationDesignVectorFinal(25) = localNodeTimeFinal;


            collocationDesignVector.block(19*i,0,26,1) = localCollocationDesignVectorFinal;
        }


    }

// Add the periodicity defects:
    int numberOfDefectPoints = (numberOfCollocationPoints-1)*3;
    Eigen::VectorXd initialState = Eigen::VectorXd::Zero(6);
    Eigen::VectorXd finalState = Eigen::VectorXd::Zero(6);

    initialState = oddStates.block(0,0,6,1);
    finalState = oddStates.block(6*(numberOfCollocationPoints-2),3,6,1 );
    collocationDefectVector.block(numberOfDefectPoints*6,0,6,1) = initialState - finalState;

// Compute phase constraint
    int lengthDefectVectorMinusOne = static_cast<int>(collocationDefectVector.rows()) - 1;
    double integralPhaseConstraint = 0.0;
    if ( continuationIndex == 1 )
    {



        integralPhaseConstraint = computeIntegralPhaseConstraint(collocationDesignVector, numberOfCollocationPoints, previousDesignVector );

        collocationDefectVector(collocationDefectVector.rows()-1,0) = integralPhaseConstraint;

    }



}


Eigen::VectorXd applyCollocation(const Eigen::MatrixXd initialCollocationGuess, const double massParameter, int& numberOfCollocationPoints, Eigen::VectorXd& collocatedGuess, Eigen::VectorXd& collocatedNodes, Eigen::VectorXd& deviationNorms, Eigen::VectorXd& collocatedDefects, const int continuationIndex, const Eigen::VectorXd previousDesignVector,
                                                          double maxPositionDeviationFromPeriodicOrbit,  double maxVelocityDeviationFromPeriodicOrbit,  double maxPeriodDeviationFromPeriodicOrbit, const int maxNumberOfCollocationIterations, const double maximumErrorTolerance)
{
    std::cout << "previousDesignVector: \n" << previousDesignVector << std::endl;

    // ======= initialize variables and rewrite input for the collocation procedure ====== //
    Eigen::VectorXd outputVector = Eigen::VectorXd::Zero(27);
    Eigen::VectorXd outputDesignVector;
    Eigen::VectorXd outputDefectVector;
    double outputDeltaDistribution = 0.0;
    int maxNumberOfCorrections  = 0;
    int maxNumberOfCorrectionsEquidistribution  = 0;

    double convergedGuessMaxSegmentError = 0.0;

    double maximumErrorPerSegment = 10.0*maximumErrorTolerance;
    Eigen::MatrixXd collocationGuessStart = initialCollocationGuess;

    // introduce a variable which will replace the initial collocation guess in the loop
    while(maximumErrorPerSegment > maximumErrorTolerance  )
    {
        // Evaluate the vector field at all odd points
        Eigen::MatrixXd initialCollocationGuessDerivatives = Eigen::MatrixXd::Zero(11*(numberOfCollocationPoints-1),4);
        initialCollocationGuessDerivatives = evaluateVectorFields(collocationGuessStart, numberOfCollocationPoints);

        // Extract state information and time intervals from the input
        Eigen::VectorXd thrustAndMassParameters = Eigen::VectorXd::Zero(4);
        thrustAndMassParameters.segment(0,4) = collocationGuessStart.block(6,0,4,1);
        double initialTime = collocationGuessStart(10,0);

        Eigen::VectorXd timeIntervals = Eigen::VectorXd::Zero(numberOfCollocationPoints -1);
        Eigen::MatrixXd oddStates = Eigen::MatrixXd::Zero(6*(numberOfCollocationPoints-1),4);
        Eigen::MatrixXd oddStatesDerivatives = Eigen::MatrixXd::Zero(6*(numberOfCollocationPoints-1),4);

        extractDurationAndDynamicsFromInput(collocationGuessStart, initialCollocationGuessDerivatives, numberOfCollocationPoints,oddStates, oddStatesDerivatives,timeIntervals);


        // compute the input to the correction algorithm
        int lengthOfDefectVector;
        if (continuationIndex == 1)
        {
            lengthOfDefectVector = (numberOfCollocationPoints-1)*18+6+1;
        } else
        {
            lengthOfDefectVector = (numberOfCollocationPoints-1)*18+6;
        }

        Eigen::VectorXd collocationDeviationNorms(5);
        Eigen::MatrixXd collocationDefectVector(lengthOfDefectVector,1);
        Eigen::MatrixXd collocationDesignVector((numberOfCollocationPoints-1)*19+7,1);
        Eigen::MatrixXd collocationDesignVectorEquidistribution((numberOfCollocationPoints-1)*19+7,1);
        Eigen::MatrixXd collocationDefectVectorEquidistribution(lengthOfDefectVector,1);
        Eigen::VectorXd segmentErrorDistribution(numberOfCollocationPoints-1);
        Eigen::MatrixXd eightOrderDerivatives(6,numberOfCollocationPoints-1);
        int meshRefinementCounter = 0;
        int numberOfCorrections = 0;
        // ======= Start the loop for mesh refinement ====== //
        double distributionDeltaPreviousIteration = 1.0E3;
        double distributionDeltaCurrentIteration = 1.0E2;

        while (distributionDeltaPreviousIteration > distributionDeltaCurrentIteration and distributionDeltaCurrentIteration > 1.0E-12)
        {
            computeCollocationDefects(collocationDefectVector, collocationDesignVector, oddStates, oddStatesDerivatives, timeIntervals, thrustAndMassParameters, numberOfCollocationPoints, initialTime, continuationIndex, previousDesignVector);

            collocationDeviationNorms = computeCollocationDeviationNorms(collocationDefectVector, collocationDesignVector, numberOfCollocationPoints);

            double positionDefectDeviations = collocationDeviationNorms(0);
            double velocityDefectDeviations= collocationDeviationNorms(1);
            double periodicityPositionDeviations= collocationDeviationNorms(2);
            double periodicityVelocityDeviations = collocationDeviationNorms(3);
            double phaseDeviations = collocationDeviationNorms(4);


            numberOfCorrections = 0;
            bool continueColloc = true;
            distributionDeltaPreviousIteration = distributionDeltaCurrentIteration;

            std::cout << "\nDeviations at the start of collocation procedure: " << std::endl;
            std::cout << "numberOfCorrections: " << numberOfCorrections << std::endl;
            std::cout << "positionDefectDeviations: " << positionDefectDeviations << std::endl;
            std::cout << "velocityDefectDeviations: " << velocityDefectDeviations << std::endl;
            std::cout << "periodicityPositionDeviations: " << periodicityPositionDeviations << std::endl;
            std::cout << "periodicityVelocityDeviations: " << periodicityVelocityDeviations << std::endl;
            std::cout << "phaseDeviations: " << phaseDeviations << std::endl;
            std::cout << "collocationDefectVector.Norm(): " << collocationDefectVector.norm() << std::endl;
            std::cout << "distributionDeltaCurrentIteration: " << distributionDeltaCurrentIteration << std::endl;

            Eigen::VectorXd initialDesignVector = collocationDesignVector.block(0,0,collocationDesignVector.rows(),1);

            // ======= Start the loop for collocation procedure ====== //
            while( (collocationDefectVector.norm() > 1.0E-12) && continueColloc  )
            {
                // compute the correction
                Eigen::VectorXd collocationCorrectionVector(collocationDesignVector.rows());
                collocationCorrectionVector.setZero();


                collocationCorrectionVector = computeCollocationCorrection(collocationDefectVector, collocationDesignVector, timeIntervals, thrustAndMassParameters, numberOfCollocationPoints, continuationIndex, previousDesignVector);

                // apply line search, select design vector which produces the smallest norm
                applyLineSearchAttenuation(collocationCorrectionVector, collocationDefectVector, collocationDesignVector, timeIntervals, thrustAndMassParameters, numberOfCollocationPoints, continuationIndex, previousDesignVector);

                // Relax the tolerances if a certain number of corrections is reached
                numberOfCorrections++;
                if ( numberOfCorrections > maxNumberOfCollocationIterations)
                {
                                maxPositionDeviationFromPeriodicOrbit = 1.0E-11;
                                maxVelocityDeviationFromPeriodicOrbit = 1.0E-11;

                }

                if ( numberOfCorrections > 2*maxNumberOfCollocationIterations)
                {
                                maxPositionDeviationFromPeriodicOrbit = 1.0E-10;
                                maxVelocityDeviationFromPeriodicOrbit = 1.0E-10;

                }

                if ( numberOfCorrections > 3*maxNumberOfCollocationIterations)
                {
                                maxPositionDeviationFromPeriodicOrbit = 1.0E-9;
                                maxVelocityDeviationFromPeriodicOrbit = 1.0E-9;

                }
                if ( numberOfCorrections > 3*maxNumberOfCollocationIterations+1)
                {
                                maxPositionDeviationFromPeriodicOrbit = 1.0E-7;
                                maxVelocityDeviationFromPeriodicOrbit = 1.0E-7;

                }
                if ( numberOfCorrections > 3*maxNumberOfCollocationIterations+2)
                {
                                maxPositionDeviationFromPeriodicOrbit = 1.0E-4;
                                maxVelocityDeviationFromPeriodicOrbit = 1.0E-4;

                }

                // compute defects after line search attenuation to determine if convergence has been reached
                collocationDeviationNorms = computeCollocationDeviationNorms(collocationDefectVector, collocationDesignVector, numberOfCollocationPoints);

                positionDefectDeviations = collocationDeviationNorms(0);
                velocityDefectDeviations= collocationDeviationNorms(1);
                periodicityPositionDeviations = collocationDeviationNorms(2);
                periodicityVelocityDeviations = collocationDeviationNorms(3);
                phaseDeviations = collocationDeviationNorms(4);

            }


            // store the solution in a seperate variables
            Eigen::VectorXd convergedDesignVector = collocationDesignVector.block(0,0,collocationDesignVector.rows(),1);
            Eigen::VectorXd convergedDefectVector = collocationDefectVector.block(0,0,collocationDefectVector.rows(),1);

            // Apply mesh refinement and compute the errors per segment
            applyMeshRefinement( collocationDesignVector, segmentErrorDistribution, thrustAndMassParameters, numberOfCollocationPoints);
            Eigen::VectorXd meshRefinedDesignVector = collocationDesignVector.block(0,0,collocationDesignVector.rows(),1);
            Eigen::VectorXd timeShiftCollocation = computeProcedureTimeShifts( initialDesignVector, convergedDesignVector, numberOfCollocationPoints);
            Eigen::VectorXd timeShiftMeshRefinement = computeProcedureTimeShifts( convergedDesignVector, meshRefinedDesignVector, numberOfCollocationPoints);
            computeSegmentProperties( collocationDesignVector, thrustAndMassParameters, numberOfCollocationPoints, oddStates, oddStatesDerivatives, timeIntervals);
            initialTime = collocationDesignVector(6);
            distributionDeltaCurrentIteration = segmentErrorDistribution.maxCoeff() - segmentErrorDistribution.minCoeff();


            std::cout << "\nTRAJECTORY CONVERGED AFTER " << numberOfCorrections << " COLLOCATION CORRECTIONS, REMAINING DEVIATIONS: " << std::endl;
            std::cout << "positionDefectDeviations: " << positionDefectDeviations << std::endl;
            std::cout << "velocityDefectDeviations: " << velocityDefectDeviations << std::endl;
            std::cout << "periodicityPositionDeviations: " << periodicityPositionDeviations << std::endl;
            std::cout << "periodicityVelocityDeviations: " << periodicityVelocityDeviations << std::endl;
            std::cout << "phaseDeviations: " << phaseDeviations << std::endl;
            std::cout << "collocationDefectVector.Norm(): " << collocationDefectVector.norm() << std::endl;
            std::cout << "distributionDeltaCurrentIteration: " << distributionDeltaCurrentIteration << std::endl;

            if ( numberOfCorrections > maxNumberOfCorrections)
            {
                maxNumberOfCorrections = numberOfCorrections;
            }

            if (distributionDeltaCurrentIteration >= distributionDeltaPreviousIteration)
            {
                // rol back design vector to the converged solution of the previous mesh
                collocationDesignVector = collocationDesignVectorEquidistribution;
                collocationDefectVector = collocationDefectVectorEquidistribution;
                initialTime = collocationDesignVector(6);
                distributionDeltaCurrentIteration = distributionDeltaPreviousIteration;
                maxNumberOfCorrections = maxNumberOfCorrectionsEquidistribution;
            }
            else
            {
                // Set solution with best equidistribution as the current solution
                collocationDesignVectorEquidistribution = convergedDesignVector;
                collocationDefectVectorEquidistribution = convergedDefectVector;
                maximumErrorPerSegment = segmentErrorDistribution.maxCoeff();
                maxNumberOfCorrectionsEquidistribution = maxNumberOfCorrections;
                meshRefinementCounter++;
            }


        }



        if (maximumErrorPerSegment > maximumErrorTolerance)
        {

            std::cout << "\n === MESH SOLVED AND EQUIDISTRIBUTED BUT DOES NOT MEET ERROR CRITERIA, ADDING PATCH POINTS === " << std::endl
                      << "collocationDefectVector Eculidian Norm: " << collocationDefectVector.norm() << std::endl
                      << "maximumError: " << maximumErrorPerSegment << std::endl
                      << "error Tolerance: : " << maximumErrorTolerance << std::endl
                      << "distributionDeltaCurrentIteration: " << distributionDeltaCurrentIteration << std::endl
                      << "current Number of Collocation Points: : " << numberOfCollocationPoints << std::endl;


            // save the old number of collocation points
            int oldNumberOfCollocationPoints = numberOfCollocationPoints;

            // compute the new number of collocation points
            double currentNumberOfCollocationPoints = static_cast<double>(numberOfCollocationPoints);
            double currentNumberOfSegments =  static_cast<double>(currentNumberOfCollocationPoints) -1.0;
            double orderOfCollocationScheme = 12.0;

            double newNumberSegments = std::round( currentNumberOfSegments * pow((10.0*maximumErrorPerSegment)/maximumErrorTolerance,1.0/(orderOfCollocationScheme+1.0)) + 5);
            numberOfCollocationPoints = static_cast<int>(newNumberSegments) +1;

            collocationGuessStart.resize(11*(numberOfCollocationPoints-1),4); collocationGuessStart.setZero();

            interpolatePolynomials(collocationDesignVector, oldNumberOfCollocationPoints, collocationGuessStart, numberOfCollocationPoints, thrustAndMassParameters, massParameter  );

            std::cout << "new Number Of Patch Points: : " << numberOfCollocationPoints << std::endl;

        } else
        {
            std::cout << "=== error tolerance achieved! ===" << std::endl
            << "maximumErrorPerSegment: " << maximumErrorPerSegment << std::endl
            << "maximumErrorTolerance: " << maximumErrorTolerance << std::endl;

            outputDeltaDistribution = distributionDeltaCurrentIteration;

            outputDesignVector = collocationDesignVector.block(0,0,collocationDesignVector.rows(),1);
            outputDefectVector = collocationDefectVector.block(0,0,collocationDefectVector.rows(),1);
            int finalNumberOfSegments = numberOfCollocationPoints-1;
            int finalNumberOfOddPoints = 3*finalNumberOfSegments+1;

            deviationNorms = computeCollocationDeviationNorms( outputDefectVector, outputDesignVector, numberOfCollocationPoints);
            collocatedGuess.resize(11*finalNumberOfOddPoints);    collocatedGuess.setZero();
            collocatedNodes.resize(11*numberOfCollocationPoints); collocatedNodes.setZero();

            shiftTimeOfConvergedCollocatedGuess(collocationDesignVector, collocatedGuess, collocatedNodes, numberOfCollocationPoints, thrustAndMassParameters);


        }

    }

    std::cout << "\n === COLLOCATION PROCEDURE COMPLETED, MESH IS SOLVED, EQUIDISTRIBUTED AND SATISFIES ERROR TOLERANCE CRITERIA === " << std::endl
              << "collocationDefectVector Eculidian Norm: " << outputDefectVector.norm() << std::endl
              << "maximumError: "                           << maximumErrorPerSegment << std::endl
              << "error Tolerance: : "                      << maximumErrorTolerance << std::endl
              << "distributionDeltaCurrentIteration: "      << outputDeltaDistribution << std::endl
              << "current Number of Collocation Points: : " << numberOfCollocationPoints << std::endl;


    // Compute variables for outputVector and collocatedGuess, rewrite collocationDesignVector to vector format (incldue interior points or not?)
    // and store in collocated guess
    Eigen::VectorXd  initialCondition = collocatedNodes.segment(0,10);
    Eigen::VectorXd  finalCondition = collocatedNodes.segment(11*(numberOfCollocationPoints-1),10);

    double orbitalPeriod = collocatedNodes(11*(numberOfCollocationPoints-1)+10) - collocatedNodes(10);

    double hamiltonianInitialCondition  = computeHamiltonian( massParameter, initialCondition);
    double hamiltonianEndState          = computeHamiltonian( massParameter, finalCondition  );

    outputVector.segment(0,10) = initialCondition;
    outputVector(10) = orbitalPeriod;
    outputVector(11) = hamiltonianInitialCondition;
    outputVector.segment(12,10) = finalCondition;
    outputVector(22) = collocatedNodes(11*(numberOfCollocationPoints-1) + 10);
    outputVector(23) = hamiltonianEndState;
    outputVector(24) = maxNumberOfCorrections;
    outputVector(25) = maximumErrorPerSegment;
    outputVector(26) = outputDeltaDistribution;


    return outputVector;

}

