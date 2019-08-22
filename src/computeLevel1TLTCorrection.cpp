#include <Eigen/Core>
#include <Eigen/LU>
#include <iostream>

#include "Tudat/Basics/utilityMacros.h"
#include "Tudat/Astrodynamics/Gravitation/librationPoint.h"
#include "Tudat/Astrodynamics/BasicAstrodynamics/celestialBodyConstants.h"
#include "Tudat/Astrodynamics/BasicAstrodynamics/physicalConstants.h"
#include "Tudat/Astrodynamics/BasicAstrodynamics/astrodynamicsFunctions.h"

#include "computeLevel1MassRefinementCorrection.h"
#include "stateDerivativeModelAugmentedVaryingMass.h"
#include "stateDerivativeModelAugmentedTLT.h"

#include "propagateTLTOrbitAugmented.h"
#include "propagateOrbitAugmented.h"

void   computeLevelIDeviations(Eigen::VectorXd localStateVector,Eigen::MatrixXd& localPropagatedStateInclSTMThrust, Eigen::MatrixXd& localPropagatedStateInclSTMCoast, const double initialTimeThrust, const double finalTimeThrust, const double finalTimeCoast, const double massParameter )
{
    std::map< double, Eigen::VectorXd > stateHistory;
    // Propagate the thrust arc
    std::pair< Eigen::MatrixXd, double > endStateAndSTMAndTimeThrust = propagateTLTOrbitAugmentedToFinalCondition(getFullInitialStateAugmented(localStateVector.segment(0,10)), massParameter, finalTimeThrust, 1, stateHistory, 1000, initialTimeThrust);

    Eigen::MatrixXd endStateAndSTMThrust      = endStateAndSTMAndTimeThrust.first;
    double endTimeThrust                  = endStateAndSTMAndTimeThrust.second;
    Eigen::VectorXd stateVectorOnlyThrust = endStateAndSTMThrust.block( 0, 0, 10, 1 );

    // propagate the coast Arc
    Eigen::VectorXd initialStateVectorCoast = stateVectorOnlyThrust;
    initialStateVectorCoast(6) = 0.0;
    double initialTimeCoast = endTimeThrust;

    std::pair< Eigen::MatrixXd, double > endStateAndSTMAndTimeCoast = propagateTLTOrbitAugmentedToFinalCondition(getFullInitialStateAugmented(initialStateVectorCoast) , massParameter, finalTimeCoast, 1, stateHistory, 1000, initialTimeCoast);

    Eigen::MatrixXd endStateAndSTMCoast  = endStateAndSTMAndTimeCoast.first;
    double endTimeCoast                  = endStateAndSTMAndTimeCoast.second;
    Eigen::VectorXd stateVectorOnlyCoast = endStateAndSTMCoast.block( 0, 0, 10, 1 );

// Store information in the localpropagatedStatesInclSTMThrust localpropagatedStatesInclSTMThrust
    localPropagatedStateInclSTMThrust = endStateAndSTMThrust;
    localPropagatedStateInclSTMCoast = endStateAndSTMCoast;

}

Eigen::VectorXd LocalLevelITLTUpdate(const Eigen::VectorXd localStateVector, const Eigen::VectorXd localDefectVector, const Eigen::MatrixXd localPropagatedStateInclSTMThrust, const Eigen::MatrixXd localPropagatedStateInclSTMCoast, double& finalTimeThrust, double s )
{
    // initialize Variables
    Eigen::VectorXd correctedStateVector(10);
    Eigen::MatrixXd updateMatrix(3,7);
    Eigen::VectorXd corrections(7);
    Eigen::VectorXd constraintVector = -1.0*localDefectVector;

    // Compute the required derivative elements
    Eigen::MatrixXd StateTransitionMatrixThrust = localPropagatedStateInclSTMThrust.block(0,1,10,10);
    Eigen::MatrixXd StateTransitionMatrixCoast = localPropagatedStateInclSTMCoast.block(0,1,10,10);
    Eigen::VectorXd finalStateThrust = localPropagatedStateInclSTMThrust.block(0,0,10,1);
    Eigen::VectorXd initialStateCoast = localPropagatedStateInclSTMThrust.block(0,0,10,1);
    initialStateCoast(6) = 0.0;

    Eigen::VectorXd stateDerivativePTMinus = computeStateDerivativeAugmentedTLT(0.0, getFullInitialStateAugmented(finalStateThrust));
    Eigen::VectorXd stateDerivativePTPlus = computeStateDerivativeAugmentedTLT(0.0, getFullInitialStateAugmented(initialStateCoast));


        // Compute sub elements of the thrust phase
        Eigen::VectorXd accelerationsPTMinus = stateDerivativePTMinus.segment(3,3);
        Eigen::MatrixXd B_PF = StateTransitionMatrixThrust.block(0,3,3,3);
        Eigen::MatrixXd D_PF = StateTransitionMatrixThrust.block(3,3,3,3);
        Eigen::MatrixXd F_PF = StateTransitionMatrixThrust.block(0,6,3,1);
        Eigen::MatrixXd J_PF = StateTransitionMatrixThrust.block(3,6,3,1);
        Eigen::MatrixXd G_PF = StateTransitionMatrixThrust.block(0,7,3,1);
        Eigen::MatrixXd K_PF = StateTransitionMatrixThrust.block(3,7,3,1);
        Eigen::MatrixXd H_PF = StateTransitionMatrixThrust.block(0,8,3,1);
        Eigen::MatrixXd L_PF = StateTransitionMatrixThrust.block(3,8,3,1);


        // Compute sub elements of the coast phase
        Eigen::MatrixXd A_PF_BAR =StateTransitionMatrixCoast.block(0,0,3,3);
        Eigen::MatrixXd B_PF_BAR =StateTransitionMatrixCoast.block(0,3,3,3);
        Eigen::MatrixXd C_PF_BAR =StateTransitionMatrixCoast.block(3,0,3,3);
        Eigen::MatrixXd D_PF_BAR =StateTransitionMatrixCoast.block(3,3,3,3);
        Eigen::VectorXd accelerationsPTPlus = stateDerivativePTPlus.segment(3,3);



    // Compute the Jacobian entries and insert into Jacobian
       Eigen::MatrixXd jacobianEntry1 = A_PF_BAR * F_PF + B_PF_BAR * J_PF;
       Eigen::MatrixXd jacobianEntry2 = A_PF_BAR * G_PF + B_PF_BAR * K_PF;
       Eigen::MatrixXd jacobianEntry3 = A_PF_BAR * H_PF + B_PF_BAR * L_PF;
       Eigen::MatrixXd jacobianEntry4 = A_PF_BAR * B_PF + B_PF_BAR * D_PF;
       Eigen::MatrixXd jacobianEntry5 =  B_PF_BAR * (accelerationsPTMinus - accelerationsPTPlus);



       updateMatrix.block(0,0,3,1) = jacobianEntry1;
       updateMatrix.block(0,1,3,1) = jacobianEntry2;
       updateMatrix.block(0,2,3,1) = jacobianEntry3;
       updateMatrix.block(0,3,3,3) = jacobianEntry4;
       updateMatrix.block(0,6,3,1) = jacobianEntry5;

    // Apply corrections
    correctedStateVector = localStateVector;
    corrections = -1.0 * ( updateMatrix.transpose() ) * ( updateMatrix * ( updateMatrix.transpose() ) ).inverse() * constraintVector;


    corrections = corrections * std::pow(0.8,s);

    corrections(1) = corrections(1)*180.0/tudat::mathematical_constants::PI;
    corrections(2) = corrections(2)*180.0/tudat::mathematical_constants::PI;


    correctedStateVector(6) = correctedStateVector(6) + corrections(0);
    correctedStateVector(7) = correctedStateVector(7) + corrections(1);
    correctedStateVector(8) = correctedStateVector(8) + corrections(2);
    correctedStateVector.segment(3,3) = correctedStateVector.segment(3,3) + corrections.segment(3,3);
    finalTimeThrust = finalTimeThrust + corrections(6);

    return correctedStateVector;


}

Eigen::VectorXd computeLevel1TLTCorrection( const Eigen::VectorXd deviationVector, const Eigen::MatrixXd propagatedStatesInclSTMThrust, const Eigen::MatrixXd propagatedStatesInclSTMCoast, const Eigen::VectorXd initialGuess, const int numberOfPatchPoints, double massParameter, int& numberOfLevelICorrections )
{

    // Initialize variables and set vectors and matrices to zero
    Eigen::VectorXd inputGuess(numberOfPatchPoints*12);
    Eigen::VectorXd correctedGuess(numberOfPatchPoints*12);
    Eigen::VectorXd localDefectVector(3);
    double initialTimeThrust;
    double finalTimeThrust;
    double finalTimeCoast;
    double positionDeviationNorm;
    Eigen::MatrixXd localPropagatedStateInclSTMThrust(10,11);
    Eigen::MatrixXd localPropagatedStateInclSTMCoast(10,11);
    Eigen::MatrixXd localUpdateMatrix(3,7);
    Eigen::VectorXd localUpdateVector(7);
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
        localStateVector = inputGuess.segment(12*i,10);
        targetStateVector = inputGuess.segment(12*(i+1),10);

        if ( i == numberOfPatchPoints -2)
        {
           targetStateVector = inputGuess.segment(0,10);
        }

        // propagated Times
        initialTimeThrust = inputGuess(12*i+10);
        finalTimeThrust = inputGuess(i*12+11);
        finalTimeCoast = inputGuess(12*(i+1)+10);

        localDefectVector = deviationVector.segment(12*i,3);
        localPropagatedStateInclSTMThrust = propagatedStatesInclSTMThrust.block(10*i,0,10,11);
        localPropagatedStateInclSTMCoast = propagatedStatesInclSTMCoast.block(10*i,0,10,11);

        positionDeviationNorm = localDefectVector.norm();
        double referenceDeviationNorm = positionDeviationNorm;
        double massForNextPatchPoint;

        while (positionDeviationNorm > 1.0E-12)
        {
            double s = 0.0;
            if ( referenceDeviationNorm <= positionDeviationNorm)
            {
                // apply correction scheme and change the finalTimeThrust
                localStateVector = LocalLevelITLTUpdate(localStateVector, localDefectVector, localPropagatedStateInclSTMThrust, localPropagatedStateInclSTMCoast, finalTimeThrust, s);

                // propagate the orbit
                computeLevelIDeviations(localStateVector, localPropagatedStateInclSTMThrust, localPropagatedStateInclSTMCoast, initialTimeThrust, finalTimeThrust, finalTimeCoast, massParameter );

                // compute deviations and the deviation norm
                   Eigen::VectorXd stateVectorMinus = localPropagatedStateInclSTMCoast.block(0,0,10,1);
                   localDefectVector = targetStateVector.segment(0,3)-stateVectorMinus.segment(0,3);
                   positionDeviationNorm = localDefectVector.norm();

                   massForNextPatchPoint = stateVectorMinus(9);
                   //std::cout << "\ntest s: " << s << std::endl;
                   s = s + 1.0;
            }

            referenceDeviationNorm = positionDeviationNorm;

            //std::cout << "patch points: " << i << std::endl;
            //std::cout << "Number Of Corrections: " << numberOfLevelICorrections+1 << std::endl
                      //<< "current deviation: " << positionDeviationNorm << std::endl;

            numberOfLevelICorrections++;


        }

        // Store information in the corrected guess
        correctedGuess.segment(12*i,10) = localStateVector;
        correctedGuess(12*i+10) = initialTimeThrust;
        correctedGuess(12*i+11) = finalTimeThrust;

        inputGuess(12*(i+1) + 9 ) = massForNextPatchPoint;


        // Add last patch point to corrected guess
        if (i == ( numberOfPatchPoints-2 ) )
        {
        correctedGuess.segment(12*(i+1),12) = inputGuess.segment(12*(i+1),12);

        }

    }



    return correctedGuess;
}
