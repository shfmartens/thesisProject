#include <Eigen/Core>
#include <Eigen/LU>
#include <iostream>

#include "computeLevel1MassRefinementCorrection.h"
#include "stateDerivativeModelAugmentedVaryingMass.h"
#include "propagateMassVaryingOrbitAugmented.h"
#include "propagateOrbitAugmented.h"




Eigen::VectorXd computeLevel1MassRefinementCorrection( const Eigen::VectorXd deviationVector, const Eigen::MatrixXd propagatedStatesInclSTM, const Eigen::VectorXd initialGuess, const int numberOfPatchPoints, double massParameter, int& numberOfLevelICorrections )
{

    // Initialize variables and set vectors and matrices to zero
    Eigen::VectorXd inputGuess(numberOfPatchPoints*11);
    Eigen::VectorXd correctedGuess(numberOfPatchPoints*11);
    Eigen::VectorXd localDefectVector(3);
    double currentPatchPointTime;
    double nextPatchPointTime;
    double positionDeviationNorm;
    Eigen::MatrixXd localPropagatedStateInclSTM(10,11);
    Eigen::MatrixXd localUpdateMatrix(3,3);
    Eigen::VectorXd localUpdateVector(3);
    Eigen::VectorXd localStateVector(10);
    Eigen::VectorXd targetStateVector(10);
    Eigen::VectorXd stateVectorOnly(10);

    std::map< double, Eigen::VectorXd > stateHistory;
    stateHistory.clear();

    correctedGuess.setZero();

    numberOfLevelICorrections = 0;

    inputGuess = initialGuess;
    for(int i = 0; i < (numberOfPatchPoints-1); i++ )
    {
        // For patch point i+1, compute the first correction using the already computated errors
        localStateVector = inputGuess.segment(11*i,10);
        targetStateVector = inputGuess.segment(11*(i+1),10);

        if ( i == numberOfPatchPoints -2)
        {
           targetStateVector = inputGuess.segment(0,10);
        }

        currentPatchPointTime = inputGuess(11*i+10);
        nextPatchPointTime = inputGuess(11*(i+1)+10);

        localDefectVector = deviationVector.segment(11*i+3,3);
        localPropagatedStateInclSTM = propagatedStatesInclSTM.block(10*i,0,10,11);
        localUpdateMatrix = localPropagatedStateInclSTM.block(0,4,3,3);

        positionDeviationNorm = localDefectVector.norm();

        int numberOfCorrectionsPerStage = 0;
        while (positionDeviationNorm > 1.0E-12)
        {

            if (numberOfCorrectionsPerStage > 50 )
            {
                std::cout << std::endl;
                correctedGuess = Eigen::VectorXd::Zero(11*numberOfPatchPoints);
                return correctedGuess;
            }

            localUpdateVector = localUpdateMatrix.inverse()*localDefectVector;
            localStateVector.segment(3,3) = localStateVector.segment(3,3) + localUpdateVector;

            std::pair< Eigen::MatrixXd, double > localEndSTMAndTime = propagateMassVaryingOrbitAugmentedToFinalCondition(
                        getFullInitialStateAugmented( localStateVector), massParameter, nextPatchPointTime, 1, stateHistory, -1, currentPatchPointTime);

            localPropagatedStateInclSTM = localEndSTMAndTime.first;
            double integrationEndTime = localEndSTMAndTime.second;
            stateVectorOnly = localPropagatedStateInclSTM.block(0,0,10,1);

            localDefectVector = targetStateVector.segment(0,3)-stateVectorOnly.segment(0,3);
            localUpdateMatrix = localPropagatedStateInclSTM.block(0,4,3,3);
            positionDeviationNorm = localDefectVector.norm();

            //std::cout << "Amount of corrections applied: " << numberOfCorrections << std::endl
            //          << "Position Deviation Norm: " << positionDeviationNorm << std::endl;

            numberOfLevelICorrections++;
            numberOfCorrectionsPerStage++;
        }

        // Store converged guess into the outputVector
        correctedGuess.segment(i*11,10) = localStateVector;
        correctedGuess(i*11+10) = currentPatchPointTime;


        //Store the mass at end of integration
        inputGuess((i+1)*11+9) = stateVectorOnly(9);

        // Add last patch point to corrected guess
        if (i == ( numberOfPatchPoints-2 ) )
        {
        correctedGuess.segment(11*(i+1),11) = inputGuess.segment(11*(i+1),11);

        }
    }

    return correctedGuess;
}
