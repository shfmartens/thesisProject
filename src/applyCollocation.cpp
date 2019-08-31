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
#include "applyLineSearchAttenuation.h"
#include "stateDerivativeModel.h"
#include "stateDerivativeModelAugmented.h"
#include "propagateOrbitAugmented.h"
#include "computeCollocationCorrection.h"

Eigen::VectorXd computeCollocationDeviationNorms(const Eigen::VectorXd collocationDefectVector, const int numberOfCollocationPoints)
{
    Eigen::VectorXd deviationNorms(2);
    deviationNorms.setZero();

    int numberOfDefects = 3*(numberOfCollocationPoints -1);
    Eigen::VectorXd positionDeviations(3*numberOfDefects);
    Eigen::VectorXd velocityDeviations(3*numberOfDefects);
    Eigen::VectorXd periodicityDeviation(6);


    positionDeviations.setZero(); velocityDeviations.setZero();
    for(int i = 0; i < numberOfDefects; i++)
    {
        Eigen::VectorXd nodeDefects(6);
        nodeDefects = collocationDefectVector.segment(i*6,6);
        positionDeviations.segment(i*3,3) = nodeDefects.segment(0,3);
        velocityDeviations.segment(i*3,3) = nodeDefects.segment(3,3);

    }


    deviationNorms(0) = positionDeviations.norm();
    deviationNorms(1) = velocityDeviations.norm();



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


void computeCollocationDefects(Eigen::MatrixXd& collocationDefectVector, Eigen::MatrixXd& collocationDesignVector, const Eigen::MatrixXd oddStates, const Eigen::MatrixXd oddStatesDerivatives, Eigen::VectorXd timeIntervals, Eigen::VectorXd thrustAndMassParameters, const int numberOfCollocationPoints)
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
        Eigen::VectorXd localCollocationDesignVector(18,1);
        Eigen::VectorXd localCollocationDesignVectorFinal(24,1);

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
            localCollocationDesignVector.segment(0,6) = localOddStates.block(0,0,6,1);
            localCollocationDesignVector.segment(6,6) = localOddStates.block(0,1,6,1);
            localCollocationDesignVector.segment(12,6) = localOddStates.block(0,2,6,1);

            collocationDesignVector.block(18*i,0,18,1) = localCollocationDesignVector;
        } else
        {
            localCollocationDesignVectorFinal.segment(0,6) = localOddStates.block(0,0,6,1);
            localCollocationDesignVectorFinal.segment(6,6) = localOddStates.block(0,1,6,1);
            localCollocationDesignVectorFinal.segment(12,6) = localOddStates.block(0,2,6,1);
            localCollocationDesignVectorFinal.segment(18,6) = localOddStates.block(0,3,6,1);


            collocationDesignVector.block(18*i,0,24,1) = localCollocationDesignVectorFinal;
        }



//        std::cout << "segment: " << i << std::endl;
//        std::cout << "localCollocationDefectVector: \n" << localCollocationDefectVector << std::endl;
//        std::cout << "localCollocationDesignVector: \n" << localCollocationDesignVector << std::endl;


    }

//    std::cout << "collocationDefectVector: \n" << collocationDefectVector << std::endl;
//    std::cout << "collocationDesignVector: \n" << collocationDesignVector << std::endl;
//    std::cout << "collocationDefectVector Size: "  << collocationDefectVector.size() << std::endl;
//    std::cout << "collocationDesignVector Size: "  << collocationDesignVector.size() << std::endl;



}


Eigen::VectorXd applyCollocation(const Eigen::MatrixXd initialCollocationGuess, const double massParameter, const int numberOfCollocationPoints, Eigen::VectorXd& collocatedGuess,
                                                         const double maxPositionDeviationFromPeriodicOrbit, const double maxVelocityDeviationFromPeriodicOrbit, const double maxPeriodDeviationFromPeriodicOrbit, const int maxNumberOfIterations)
{
    // initialize Variables
    Eigen::VectorXd outputVector = Eigen::VectorXd(25);
    int numberOfCorrections = 0;

    // Evaluate the vector field at all odd points
    Eigen::MatrixXd initialCollocationGuessDerivatives = Eigen::MatrixXd::Zero(11*(numberOfCollocationPoints-1),4);
    initialCollocationGuessDerivatives = evaluateVectorFields(initialCollocationGuess, numberOfCollocationPoints);

    // Extract state information and time intervals from the input
    Eigen::VectorXd thrustAndMassParameters = Eigen::VectorXd::Zero(4);
    thrustAndMassParameters.segment(0,4) = initialCollocationGuess.block(6,0,4,1);


    Eigen::VectorXd timeIntervals = Eigen::VectorXd::Zero(numberOfCollocationPoints -1);
    Eigen::MatrixXd oddStates = Eigen::MatrixXd::Zero(6*(numberOfCollocationPoints-1),4);
    Eigen::MatrixXd oddStatesDerivatives = Eigen::MatrixXd::Zero(6*(numberOfCollocationPoints-1),4);

    extractDurationAndDynamicsFromInput(initialCollocationGuess, initialCollocationGuessDerivatives, numberOfCollocationPoints,oddStates, oddStatesDerivatives,timeIntervals);

    // compute the input to the correction algorithm
    Eigen::MatrixXd collocationDefectVector((numberOfCollocationPoints-1)*18,1);
    Eigen::MatrixXd collocationDesignVector((numberOfCollocationPoints-1)*18+6,1);

    computeCollocationDefects(collocationDefectVector, collocationDesignVector, oddStates, oddStatesDerivatives, timeIntervals, thrustAndMassParameters, numberOfCollocationPoints);
    Eigen::VectorXd collocationDeviationNorms = computeCollocationDeviationNorms(collocationDefectVector, numberOfCollocationPoints);

    double positionDefectDeviations = collocationDeviationNorms(0);
    double velocityDefectDeviations= collocationDeviationNorms(1);

    std::cout << "\nDeviations at the start of collocation procedure: " << std::endl;
    std::cout << "numberOfCorrections: " << numberOfCorrections << std::endl;
    std::cout << "positionDefectDeviations: " << positionDefectDeviations << std::endl;
    std::cout << "velocityDefectDeviations: " << velocityDefectDeviations << std::endl;

    while( positionDefectDeviations > maxPositionDeviationFromPeriodicOrbit
           or velocityDefectDeviations > maxVelocityDeviationFromPeriodicOrbit  )
    {

        // compute the correction
        Eigen::VectorXd collocationCorrectionVector(collocationDesignVector.size());
        collocationCorrectionVector.setZero();
        collocationCorrectionVector = computeCollocationCorrection(collocationDefectVector, collocationDesignVector, timeIntervals, thrustAndMassParameters, numberOfCollocationPoints);
        //std::cout << " collocationCorrectionVector: " << collocationCorrectionVector << std::endl;

        // apply line search, select design vector which produces the smallest norm
        applyLineSearchAttenuation(collocationCorrectionVector, collocationDefectVector, collocationDesignVector, timeIntervals, thrustAndMassParameters, numberOfCollocationPoints);

        numberOfCorrections++;
        // compute defects after line search attenuation to determine if convergence has been reached
        collocationDeviationNorms = computeCollocationDeviationNorms(collocationDefectVector, numberOfCollocationPoints);

        positionDefectDeviations = collocationDeviationNorms(0);
        velocityDefectDeviations= collocationDeviationNorms(1);

        std::cout << "\nCollocation applied, remaining deviations are: " << std::endl;
        std::cout << "numberOfCorrections: " << numberOfCorrections << std::endl;
        std::cout << "positionDefectDeviations: " << positionDefectDeviations << std::endl;
        std::cout << "velocityDefectDeviations: " << velocityDefectDeviations << std::endl;

    }

    std::cout << "\nTRAJECTORY CONVERGED AFTER " << numberOfCorrections << " COLLOCATION CORRECTIONS, REMAINING DEVIATIONS: " << std::endl;
    std::cout << "numberOfCorrections: " << numberOfCorrections << std::endl;
    std::cout << "positionDefectDeviations: " << positionDefectDeviations << std::endl;
    std::cout << "velocityDefectDeviations: " << velocityDefectDeviations << std::endl;

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

