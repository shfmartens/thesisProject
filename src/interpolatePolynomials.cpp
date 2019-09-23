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
#include "interpolatePolynomials.h"

void rewriteDesignVectorToFullFormatComplex(const Eigen::VectorXcd collocationDesignVector, const int numberOfCollocationPoints, const Eigen::VectorXcd thrustAndMassParameters, Eigen::VectorXcd& currentDesignVector )
{
    Eigen::MatrixXd nodeTimes(7,1); nodeTimes.setZero();
    nodeTimes.setZero();
    retrieveLegendreGaussLobattoConstaints("nodeTimes", nodeTimes);
    Eigen::MatrixXcd nodeTimesComplex = nodeTimes;

    for (int i = 0; i < numberOfCollocationPoints; i++)
    {
        Eigen::VectorXcd localRewrittenVector(44); localRewrittenVector.setZero();

        Eigen::VectorXcd segmentDesignVector = collocationDesignVector.segment(19*i,26);
        std::complex<double> segmentInitialTime = segmentDesignVector(6);
        std::complex<double> segmentFinalTime = segmentDesignVector(25);
        std::complex<double> deltaTime = segmentFinalTime - segmentInitialTime;

        // SHIFT IN COMPUTATION OF NODE POINT SHOULD NOT ALTER TIME OF INTERIOR POINTS! HENCE
        // THE SEGMENT TIME MAY CHANGE BUT IS IGNORED TO KEEP THE PROPAGATION PONTS AT THE EXACT
        // SAME PLACE

        deltaTime.imag(0.0);

        Eigen::VectorXcd localDesignVector(11);

        for (int j = 0; j < 4; j++)
        {
            if(j == 0)
            {
                localDesignVector.segment(0,6) = segmentDesignVector.segment(0,6);

            } else
            {
                localDesignVector.segment(0,6) = segmentDesignVector.segment((6*j)+1,6);
            }

            localDesignVector.segment(6,4) = thrustAndMassParameters;
            localDesignVector(10) = segmentInitialTime + deltaTime*nodeTimesComplex(2*j,0);
            localRewrittenVector.segment(j*11,11) = localDesignVector;
        }

       currentDesignVector.segment(33*i,44) = localRewrittenVector;
    }

}

void rewriteDesignVectorToFullFormat(const Eigen::MatrixXd collocationDesignVector, const int numberOfCollocationPoints, const Eigen::VectorXd thrustAndMassParameters, Eigen::VectorXd& currentDesignVector )
{
    Eigen::MatrixXd nodeTimes(7,1); nodeTimes.setZero();
    nodeTimes.setZero();
    retrieveLegendreGaussLobattoConstaints("nodeTimes", nodeTimes);

    for (int i = 0; i < numberOfCollocationPoints-1; i++)
    {
        Eigen::VectorXd localRewrittenVector(44); localRewrittenVector.setZero();

        Eigen::VectorXd segmentDesignVector = collocationDesignVector.block(19*i,0,26,1);
        double segmentInitialTime = segmentDesignVector(6);
        double segmentFinalTime = segmentDesignVector(25);
        double deltaTime = segmentFinalTime - segmentInitialTime;
        Eigen::VectorXd localDesignVector(11);

        for (int j = 0; j < 4; j++)
        {
            if(j == 0)
            {
                localDesignVector.segment(0,6) = segmentDesignVector.segment(0,6);

            } else
            {
                localDesignVector.segment(0,6) = segmentDesignVector.segment((6*j)+1,6);
            }

            localDesignVector.segment(6,4) = thrustAndMassParameters;
            localDesignVector(10) = segmentInitialTime + deltaTime*nodeTimes(2*j,0);
            localRewrittenVector.segment(j*11,11) = localDesignVector;
        }

       currentDesignVector.segment(33*i,44) = localRewrittenVector;
    }
}

double computeIntegralPhaseConstraint(const Eigen::MatrixXd collocationDesignVector, const int numberOfCollocationPoints,const Eigen::VectorXd previousDesignVector )
{
    std::cout << "START computeIntegralPhaseConstraint " << std::endl;
    // initialize variables
    double phaseIntegralConstraint = 0.0;
    int currentNumberOfSegments = numberOfCollocationPoints - 1;
    int currentNumberOfOddPoints = 3*currentNumberOfSegments+1;

    int previousNumberOfSegments = (((static_cast<int>(previousDesignVector.rows()))/11)-1)/3;
    int previousNumberOfCollocationPoints = previousNumberOfSegments+1;


    // Compute the properties of the previous guess, necessary for interpolation
    Eigen::VectorXd thrustAndMassParameters = previousDesignVector.segment(6,4);
    Eigen::MatrixXd oddStates(6*previousNumberOfSegments,4);
    Eigen::MatrixXd oddStateDerivatives(6*previousNumberOfSegments,4);
    Eigen::VectorXd timeIntervals(previousNumberOfSegments);

    computeSegmentProperties(previousDesignVector, thrustAndMassParameters, previousNumberOfCollocationPoints, oddStates, oddStateDerivatives, timeIntervals);

    // compute the time information of the new nodes
    Eigen::VectorXd currentDesignVector(11*currentNumberOfOddPoints);   currentDesignVector.setZero();
    Eigen::VectorXd oddPointTimesDimensional(currentNumberOfOddPoints); oddPointTimesDimensional.setZero();
    Eigen::VectorXd oddPointTimesNormalized(currentNumberOfOddPoints);  oddPointTimesNormalized.setZero();
    Eigen::VectorXd segmentVector(currentNumberOfOddPoints);            segmentVector.setZero();

    rewriteDesignVectorToFullFormat(collocationDesignVector, numberOfCollocationPoints, thrustAndMassParameters, currentDesignVector);

    computeTimeAndSegmentInformationFromPhase(currentDesignVector, previousDesignVector, numberOfCollocationPoints, oddStates, previousNumberOfCollocationPoints,
                                              oddPointTimesDimensional, oddPointTimesNormalized, segmentVector);

    // perform Interpolation and compute the derivatives
    Eigen::VectorXd incrementOddPoints(6*currentNumberOfOddPoints);
    Eigen::VectorXd currentGuessOddPoints(6*currentNumberOfOddPoints);
    Eigen::VectorXd previousGuessOddPointsSynced(6*currentNumberOfOddPoints);
    Eigen::VectorXd previousGuessOddDerivatesSynced(6*currentNumberOfOddPoints);
    Eigen::VectorXd oddPointPhaseConstraints(currentNumberOfOddPoints); oddPointPhaseConstraints.setZero();

    for(int i = 0; i < currentNumberOfOddPoints; i++)
    {
        Eigen::VectorXd oddPointStateVectorCurrentGuess(6); oddPointStateVectorCurrentGuess.setZero();
        Eigen::VectorXd oddPointStateVectorPreviousGuess(6); oddPointStateVectorPreviousGuess.setZero();
        Eigen::VectorXd oddPointDerivativePreviousGuess(6); oddPointDerivativePreviousGuess.setZero();

        // select relevant parameters for interpolation
        auto segmentNumber = static_cast<int>(segmentVector(i));
        double interpolationTime = oddPointTimesNormalized(i);
        double segmentTimeInterval = timeIntervals(segmentNumber);

        Eigen::MatrixXd segmentOddStates = oddStates.block(6*segmentNumber,0,6,4);
        Eigen::MatrixXd segmentOddStateDerivatives = oddStateDerivatives.block(6*segmentNumber,0,6,4);

        // perform interpolation
        Eigen::VectorXd interpolatedOddPoint = computeStateViaPolynomialInterpolation(segmentOddStates, segmentOddStateDerivatives, segmentTimeInterval, interpolationTime);
        Eigen::VectorXd oddPointStateVectorInclParameters(10);
        oddPointStateVectorInclParameters.segment(0,6) = interpolatedOddPoint;
        oddPointStateVectorInclParameters.segment(6,4) = thrustAndMassParameters;

        // Fill relevant variables
        oddPointStateVectorPreviousGuess= interpolatedOddPoint;
        oddPointDerivativePreviousGuess = computeStateDerivativeAugmented(0.0, getFullInitialStateAugmented(oddPointStateVectorInclParameters)).block(0,0,6,1);
        oddPointStateVectorCurrentGuess = currentDesignVector.segment(i*11,6);

        // Store in constraint components in the vectors
        previousGuessOddPointsSynced.segment(6*i,6)     = oddPointStateVectorPreviousGuess;
        previousGuessOddDerivatesSynced.segment(6*i,6)  = oddPointDerivativePreviousGuess;
        currentGuessOddPoints.segment(6*i,6)            = oddPointStateVectorCurrentGuess;
        incrementOddPoints.segment(6*i,6) = oddPointStateVectorCurrentGuess - oddPointStateVectorPreviousGuess;
    }

    // Compute versions of the integral constraint
    Eigen::VectorXd phaseConstraintPoincare (currentNumberOfOddPoints);    phaseConstraintPoincare.setZero();
    Eigen::VectorXd phaseConstraintLiterature (currentNumberOfOddPoints);  phaseConstraintLiterature.setZero();

    double alternativeOuput = 0.0;
    for(int i = 0; i < currentNumberOfOddPoints; i++)
    {
        Eigen::VectorXd currentIncrement = incrementOddPoints.segment(6*i,6);
        Eigen::VectorXd currentOddPoint = currentGuessOddPoints.segment(6*i,6);
        Eigen::VectorXd previousOddDerivative = previousGuessOddDerivatesSynced.segment(6*i,6);

        phaseConstraintPoincare(i) = currentIncrement.transpose() * previousOddDerivative;
        phaseConstraintLiterature(i) = currentOddPoint.transpose() * previousOddDerivative;
        alternativeOuput = alternativeOuput + currentIncrement.transpose() * previousOddDerivative;
    }

//    // could it be that absolute version should be summed?
//    phaseIntegralConstraint = phaseConstraintPoincare.sum();

//    std::cout << "phaseConstraintPoincare:  \n" << phaseConstraintPoincare << std::endl;
//    std::cout << "phaseIntegralConstraint:  " << phaseIntegralConstraint << std::endl;
//    std::cout << "alternativeOuput:  \n" << alternativeOuput << std::endl;
//    std::cout << "completed computeIntegralPhaseConstraint " << std::endl;


    return phaseIntegralConstraint;

}

void computeTimeAndSegmentInformationFromPhaseComplex(const Eigen::VectorXcd currentDesignVector, const Eigen::VectorXcd previousDesignVector, const int currentNumberOfCollocationPoints, const Eigen::MatrixXcd oddStates, const int previousNumberOfCollocationPoints,
                                                      Eigen::VectorXcd& oddPointTimesDimensional, Eigen::VectorXcd& oddPointTimesNormalized, Eigen::VectorXd& segmentVector)
{
    //compute number of segements and odd points for both guesses
    int currentNumberOfSegments = currentNumberOfCollocationPoints-1;
    int previousNumberOfSegments = previousNumberOfCollocationPoints-1;

    int currentNumberOfOddPoints = 3*currentNumberOfSegments+1;
    int previousNumberOfOddPoints = 3*previousNumberOfSegments+1;


    //compute from the phases of oddPoitns off the current guess  and dimensional times of each odd point
    //on the previous guess corresponding to these phases
    Eigen::VectorXcd currentGuessPhaseVector(currentNumberOfOddPoints);     currentGuessPhaseVector.setZero();
    std::complex<double> initialTimeCurrentGuess = currentDesignVector(10);
    std::complex<double> finalTimeCurrentGuess = currentDesignVector((currentNumberOfOddPoints-1)*11+10);

    std::complex<double> initialTimePreviousGuess = previousDesignVector(10);
    std::complex<double> finalTimePreviousGuess = previousDesignVector((previousNumberOfOddPoints-1)*11+10);
    std::complex<double> orbitalPeriodPreviousGuess = finalTimePreviousGuess - initialTimePreviousGuess;


    for(int i = 0; i < currentNumberOfOddPoints; i++)
    {
        // compute currentGuessPhaseVector
        Eigen::VectorXcd oddStateVector = currentDesignVector.segment(i*11,11);
        std::complex<double> oddPointTime = oddStateVector(10);
        std::complex<double> oddPointPhase = (oddPointTime - initialTimeCurrentGuess)/(finalTimeCurrentGuess-initialTimeCurrentGuess);
        currentGuessPhaseVector(i) = (oddPointTime - initialTimeCurrentGuess)/(finalTimeCurrentGuess-initialTimeCurrentGuess);

        std::complex<double> dimensionalTimeOnPreviousGuess = initialTimePreviousGuess + oddPointPhase*orbitalPeriodPreviousGuess;
        oddPointTimesDimensional(i) = dimensionalTimeOnPreviousGuess;
    }


    // construct the timeIntervalMatrix which hold the time bounds per segment!
    Eigen::MatrixXcd timeIntervalMatrix(previousNumberOfSegments,2); timeIntervalMatrix.setZero();
    for(int i = 0; i < previousNumberOfSegments; i++)
    {
        Eigen::VectorXcd localStateVector = previousDesignVector.segment(33*i,44);
        std::complex<double> initialTime = localStateVector(10);
        std::complex<double> finalTime = localStateVector(43);

        timeIntervalMatrix(i,0) = initialTime;
        timeIntervalMatrix(i,1) = finalTime;

    }


    for(int i = 0; i < currentNumberOfOddPoints; i++ )
    {
        for(int j = 0; j<previousNumberOfSegments; j++)
        {
            double segmentNumber = j;
            std::complex<double> segmentInitialTime = timeIntervalMatrix(j,0);
            std::complex<double> segmentFinalTime = timeIntervalMatrix(j,1);
            std::complex<double> segmentTimeInterval = segmentFinalTime - segmentInitialTime;
            std::complex<double> oddPointTime = oddPointTimesDimensional(i);

            double oddPointTimeMagnitude = std::abs(oddPointTime);
            double segmentInitialTimeMagnitude = std::abs(segmentInitialTime);
            double segmentFinalTimeMagnitude = std::abs(segmentFinalTime);

            if ( oddPointTimeMagnitude >= segmentInitialTimeMagnitude && oddPointTimeMagnitude <= segmentFinalTimeMagnitude  )
            {
                segmentVector(i) = segmentNumber;
                oddPointTimesNormalized(i) = (oddPointTime - segmentInitialTime) / (segmentTimeInterval);
            }

        }
    }




}
void computeTimeAndSegmentInformationFromPhase(const Eigen::VectorXd currentDesignVector, const Eigen::VectorXd previousDesignVector, const int currentNumberOfCollocationPoints, const Eigen::MatrixXd oddStates, const int previousNumberOfCollocationPoints,
                                               Eigen::VectorXd& oddPointTimesDimensional, Eigen::VectorXd& oddPointTimesNormalized, Eigen::VectorXd& segmentVector)
{

    //compute number of segements and odd points for both guesses
    int currentNumberOfSegments = currentNumberOfCollocationPoints-1;
    int previousNumberOfSegments = previousNumberOfCollocationPoints-1;

    int currentNumberOfOddPoints = 3*currentNumberOfSegments+1;
    int previousNumberOfOddPoints = 3*previousNumberOfSegments+1;


    //compute from the phases of oddPoitns off the current guess  and dimensional times of each odd point
    //on the previous guess corresponding to these phases
    Eigen::VectorXd currentGuessPhaseVector = Eigen::VectorXd::Zero(currentNumberOfOddPoints);
    double initialTimeCurrentGuess = currentDesignVector(10);
    double finalTimeCurrentGuess = currentDesignVector((currentNumberOfOddPoints-1)*11+10);

    double initialTimePreviousGuess = previousDesignVector(10);
    double finalTimePreviousGuess = previousDesignVector((previousNumberOfOddPoints-1)*11+10);
    double orbitalPeriodPreviousGuess = finalTimePreviousGuess - initialTimePreviousGuess;

    for(int i = 0; i < currentNumberOfOddPoints; i++)
    {
        // compute currentGuessPhaseVector
        Eigen::VectorXd oddStateVector = currentDesignVector.segment(i*11,11);
        double oddPointTime = oddStateVector(10);
        double oddPointPhase = (oddPointTime - initialTimeCurrentGuess)/(finalTimeCurrentGuess-initialTimeCurrentGuess);
        currentGuessPhaseVector(i) = (oddPointTime - initialTimeCurrentGuess)/(finalTimeCurrentGuess-initialTimeCurrentGuess);

        double dimensionalTimeOnPreviousGuess = initialTimePreviousGuess + oddPointPhase*orbitalPeriodPreviousGuess;
        oddPointTimesDimensional(i) = dimensionalTimeOnPreviousGuess;
    }


    // construct the timeIntervalMatrix which hold the time bounds per segment!
    Eigen::MatrixXd timeIntervalMatrix(previousNumberOfSegments,2); timeIntervalMatrix.setZero();
    for(int i = 0; i < previousNumberOfSegments; i++)
    {
        Eigen::VectorXd localStateVector = previousDesignVector.segment(33*i,44);
        double initialTime = localStateVector(10);
        double finalTime = localStateVector(43);

        timeIntervalMatrix(i,0) = initialTime;
        timeIntervalMatrix(i,1) = finalTime;

    }

    for(int i = 0; i < currentNumberOfOddPoints; i++ )
    {
        for(int j = 0; j<previousNumberOfSegments; j++)
        {
            double segmentNumber = j;
            double segmentInitialTime = timeIntervalMatrix(j,0);
            double segmentFinalTime = timeIntervalMatrix(j,1);
            double segmentTimeInterval = segmentFinalTime - segmentInitialTime;
            double oddPointTime = oddPointTimesDimensional(i);
            if ( oddPointTime >= segmentInitialTime && oddPointTime <= segmentFinalTime  )
            {
                segmentVector(i) = segmentNumber;
                oddPointTimesNormalized(i) = (oddPointTime - segmentInitialTime) / (segmentTimeInterval);
            }

        }
    }

}

void computeStateIncrementFromInterpolation (const Eigen::VectorXd previousGuess, Eigen::VectorXd currentGuess, Eigen::VectorXd& stateIncrement)
{


        // compute numberOfCollocationPoints per guess
        int previousNumberOfCollocationPoints = ((previousGuess.size()-3)/11-1)/3  + 1;
        int currentNumberOfCollocationPoints = ((currentGuess.size()-3)/11-1)/3  + 1;

        int previousNumberOfSegments = previousNumberOfCollocationPoints - 1;
        int currentNumberOfSegments = currentNumberOfCollocationPoints  - 1;

        int previousNumberOfOddPoints = 3*previousNumberOfSegments + 1;
        int currentNumberOfOddPoints = 3*currentNumberOfSegments + 1;

        //Extract the collocation Design Vectors from the input
        Eigen::VectorXd previousDesignVector = previousGuess.segment(3,11*previousNumberOfOddPoints);
        Eigen::VectorXd currentDesignVector = currentGuess.segment(3,11*currentNumberOfOddPoints);

        // compute segment properties of the previous guess
        Eigen::VectorXd thrustAndMassParameters = previousDesignVector.segment(6,4);
        Eigen::MatrixXd oddStates(6*previousNumberOfSegments,4);            oddStates.setZero();
        Eigen::MatrixXd oddStateDerivatives(6*previousNumberOfSegments,4);  oddStateDerivatives.setZero();
        Eigen::VectorXd timeIntervals(previousNumberOfSegments);            timeIntervals.setZero();

        computeSegmentProperties(previousDesignVector, thrustAndMassParameters, previousNumberOfCollocationPoints,
                                 oddStates, oddStateDerivatives, timeIntervals);

        // compute Time and Segment Information From phase information of the current guess
        Eigen::VectorXd oddPointTimesDimensional(currentNumberOfOddPoints); oddPointTimesDimensional.setZero();
        Eigen::VectorXd oddPointTimesNormalized(currentNumberOfOddPoints);  oddPointTimesDimensional.setZero();
        Eigen::VectorXd segmentVector(currentNumberOfOddPoints);            segmentVector.setZero();

        computeTimeAndSegmentInformationFromPhase(currentDesignVector, previousDesignVector, currentNumberOfCollocationPoints, oddStates, previousNumberOfCollocationPoints,
                                                  oddPointTimesDimensional, oddPointTimesNormalized, segmentVector);

        Eigen::VectorXd previousDesignVectorInterpolated(11*currentNumberOfOddPoints); previousDesignVectorInterpolated.setZero();
        for(int i = 0; i < currentNumberOfOddPoints; i++)
        {
            // select relevant parameters for interpolation
            auto segmentNumber = static_cast<int>(segmentVector(i));
            double interpolationTime = oddPointTimesNormalized(i);
            double oddPointTime = oddPointTimesDimensional(i);
            double segmentTimeInterval = timeIntervals(segmentNumber);

            Eigen::MatrixXd segmentOddStates = oddStates.block(6*segmentNumber,0,6,4);
            Eigen::MatrixXd segmentOddStateDerivatives = oddStateDerivatives.block(6*segmentNumber,0,6,4);

            // perform interpolation
            Eigen::VectorXd interpolatedOddPoint = computeStateViaPolynomialInterpolation(segmentOddStates, segmentOddStateDerivatives, segmentTimeInterval, interpolationTime);

            // create the state vector with thrust and mass variables and the dimensional time of the odd point
            Eigen::VectorXd localStateVector(11); localStateVector.setZero();
            localStateVector.segment(0,6) = interpolatedOddPoint;
            localStateVector.segment(6,4) = thrustAndMassParameters;
            localStateVector(10) = oddPointTimesDimensional(i);

            previousDesignVectorInterpolated.segment(i*11,11) = localStateVector;
       }

        // compute the stateIncrement via substraction ofcurrentDesignVector DesignVector and  previousDesignVectorInterpolated
       stateIncrement = currentDesignVector - previousDesignVectorInterpolated;






}

void computeTimeAndSegmentInformation(const Eigen::MatrixXd collocationDesignVector, const int oldNumberOfCollocationPoints, const int newNumberOfCollocationPoints, Eigen::VectorXd& oddPointTimesNormalized, Eigen::VectorXd& oddPointTimesDimensional, Eigen::VectorXd& segmentVector)
{

    // extract old mesh time information from the current guess (T and time bounds of segments
    int oldNumberOfSegments = oldNumberOfCollocationPoints-1;
    Eigen::MatrixXd timeIntervalMatrix(oldNumberOfSegments,2 );
    for(int i = 0; i < oldNumberOfSegments;i++)
    {
        Eigen::VectorXd localDesignVector = collocationDesignVector.block(19*i,0,26,1);
        timeIntervalMatrix(i,0) = localDesignVector(6);
        timeIntervalMatrix(i,1) = localDesignVector(25);

    }

    double initialTime = timeIntervalMatrix(0,0);
    double finalTime =   timeIntervalMatrix(oldNumberOfSegments-1,1);
    double orbitalPeriod =finalTime - initialTime;

    //  compute the dimensional Node Points
    Eigen::VectorXd nodeTimesDimensional(newNumberOfCollocationPoints); nodeTimesDimensional.setZero();
    int newNumberOfSegments = newNumberOfCollocationPoints-1;
    double timeInterval =  orbitalPeriod /  static_cast<double>(newNumberOfSegments);
    for(int i = 0; i < newNumberOfCollocationPoints; i++)
    {
        nodeTimesDimensional(i) = initialTime + static_cast<double>(i)*timeInterval;


    }

    // compute the dimensional oddPoints
    Eigen::MatrixXd oddTimes(7,1); oddTimes.setZero();
    retrieveLegendreGaussLobattoConstaints("nodeTimes",oddTimes);

    for(int i = 0; i < newNumberOfSegments; i++)
    {
        Eigen::VectorXd localSegmentTimes(4); localSegmentTimes.setZero();

        double initialSegmentTime = nodeTimesDimensional(i);
        double finalSegmentTime = nodeTimesDimensional(i+1);
        double segmentTimeInterval = finalSegmentTime - initialSegmentTime;
        for(int j = 0; j < 4; j++)
        {

            localSegmentTimes(j) = initialSegmentTime + oddTimes(2*j)*segmentTimeInterval;

        }

        oddPointTimesDimensional.segment(3*i,4) = localSegmentTimes;
    }

    // compute the segmentVector and nodeTimesNormalizedVector
    int newNumberOfOddPoints = 3*newNumberOfSegments+1;

    for(int i = 0; i < newNumberOfOddPoints; i++ )
    {
        for(int j = 0; j<oldNumberOfSegments; j++)
        {
            double segmentNumber = j;
            double segmentInitialTime = timeIntervalMatrix(j,0);
            double segmentFinalTime = timeIntervalMatrix(j,1);
            double segmentTimeInterval = segmentFinalTime - segmentInitialTime;
            double oddPointTime = oddPointTimesDimensional(i);
            if ( oddPointTime >= segmentInitialTime && oddPointTime <= segmentFinalTime  )
            {
                segmentVector(i) = j;
                oddPointTimesNormalized(i) = (oddPointTime - segmentInitialTime) / (segmentTimeInterval);
            }

        }
    }

}

void interpolatePolynomials(const Eigen::MatrixXd collocationDesignVector, const int oldNumberOfCollocationPoints, Eigen::MatrixXd& collocationGuessStart, const int newNumberOfCollocationPoints, const Eigen::VectorXd thrustAndMassParameters, const double massParameter )
{
    // declare variables
    int oldNumberOfSegments = oldNumberOfCollocationPoints-1;
    int newNumberOfSegments = newNumberOfCollocationPoints-1;
    int oldNumberOfOddPoints = 3*oldNumberOfSegments+1;
    int newNumberOfOddPoints = 3*newNumberOfSegments+1;

    Eigen::MatrixXd oddStates(6*oldNumberOfSegments, 4);
    Eigen::MatrixXd oddStateDerivatives(6*oldNumberOfSegments, 4);
    Eigen::VectorXd oldTimeIntervals(oldNumberOfSegments);


    // compute the oddStates, oddStateDerivatives and timeIntervals, necessary for the interpolation
    computeSegmentProperties( collocationDesignVector, thrustAndMassParameters, oldNumberOfCollocationPoints, oddStates, oddStateDerivatives, oldTimeIntervals );


    // compute the time information of the new nodes
    Eigen::VectorXd oddPointTimesDimensional(newNumberOfOddPoints); oddPointTimesDimensional.setZero();
    Eigen::VectorXd oddPointTimesNormalized(newNumberOfOddPoints);  oddPointTimesDimensional.setZero();
    Eigen::VectorXd segmentVector(newNumberOfOddPoints);            segmentVector.setZero();

    computeTimeAndSegmentInformation(collocationDesignVector, oldNumberOfCollocationPoints, newNumberOfCollocationPoints, oddPointTimesNormalized, oddPointTimesDimensional, segmentVector);

    int segmentOutputMatrix= 0;
    for(int i = 0; i < newNumberOfOddPoints; i++)
    {
        // select relevant parameters for interpolation
        auto segmentNumber = static_cast<int>(segmentVector(i));
        double interpolationTime = oddPointTimesNormalized(i);
        double oddPointTime = oddPointTimesDimensional(i);
        double segmentTimeInterval = oldTimeIntervals(segmentNumber);

        Eigen::MatrixXd segmentOddStates = oddStates.block(6*segmentNumber,0,6,4);
        Eigen::MatrixXd segmentOddStateDerivatives = oddStateDerivatives.block(6*segmentNumber,0,6,4);

        // perform interpolation
        Eigen::VectorXd interpolatedOddPoint = computeStateViaPolynomialInterpolation(segmentOddStates, segmentOddStateDerivatives, segmentTimeInterval, interpolationTime);

        // create the state vector with thrust and mass variables and the dimensional time of the odd point
        Eigen::VectorXd localStateVector(11); localStateVector.setZero();
        localStateVector.segment(0,6) = interpolatedOddPoint;
        localStateVector.segment(6,4) = thrustAndMassParameters;
        localStateVector(10) = oddPointTimesDimensional(i);

        // if oddPoint is a node, store also the final column of the current segment
        if (i > 0 && i % 3 == 0)
        {
            collocationGuessStart.block(11*segmentOutputMatrix,3,11,1) = localStateVector;
            segmentOutputMatrix++;

        }

        int columnNumber = i % 3;

        if( i < (newNumberOfOddPoints -1) )
        {
            collocationGuessStart.block(11*segmentOutputMatrix, columnNumber, 11,1) = localStateVector;
        }


    }

}



