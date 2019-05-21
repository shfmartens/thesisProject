#include <Eigen/Core>
#include <Eigen/LU>
#include <iostream>

#include "computeLevel2Correction.h"
#include "stateDerivativeModelAugmented.h"
#include "propagateOrbitAugmented.h"



Eigen::VectorXd computeLevel2Correction( const Eigen::VectorXd deviationVector, const Eigen::MatrixXd propagatedStatesInclSTM, const Eigen::VectorXd initialGuess, const int numberOfPatchPoints )
{

// Initialize variables and set vectors and matrices to zero
Eigen::VectorXd differentialCorrection(11*numberOfPatchPoints);
Eigen::VectorXd correctionsAtPatchPoint(4*numberOfPatchPoints);
Eigen::VectorXd constraintVector(3*(numberOfPatchPoints-2));
Eigen::VectorXd constraintVectorPeriodic(3*(numberOfPatchPoints));
Eigen::MatrixXd updateMatrix(3*(numberOfPatchPoints-2),4*numberOfPatchPoints);
Eigen::MatrixXd updateMatrixPeriodic(3*(numberOfPatchPoints),4*numberOfPatchPoints);
Eigen::VectorXd corrections(4*numberOfPatchPoints);

Eigen::MatrixXd periodicityTermR_0(3,3);
Eigen::VectorXd periodicityTermTime_0(3,1);
Eigen::MatrixXd periodicityTermR_1(3,3);
Eigen::VectorXd periodicityTermTime_1(3,1);
Eigen::MatrixXd periodicityTermR_FinalMinus1(3,3);
Eigen::VectorXd periodicityTermTime_FinalMinus1(3,1);
Eigen::MatrixXd periodicityTermR_Final(3,3);
Eigen::VectorXd periodicityTermTime_Final(3,1);



constraintVector.setZero();
updateMatrix.setZero();
differentialCorrection.setZero();
//std::cout.precision(8);
//std::cout << "====DEBUGGING LII CORRECTOR===" << std::endl
//          << "DeviationVector: \n" << deviationVector << std::endl
//          << "PropagatedStatesInclSTM: \n" << propagatedStatesInclSTM << std::endl
//          << "initialGuess: \n" << initialGuess << std::endl
//          << "numberOfPatchPoints: " << numberOfPatchPoints << std::endl;


// compute velocity continuity for all interior patch points
for(int k = 1; k < numberOfPatchPoints-1; k++){

    //std::cout << "CURRENT PATCH POINT: " << k << std::endl;

    // define the STM's from k-1 to k (past) and k to k+1 (future), accelerations at k, and propagatedStates
    Eigen::MatrixXd STM_PO(10,10);
    Eigen::MatrixXd STM_FP(10,10);
    Eigen::MatrixXd A_PO(3,3);
    Eigen::MatrixXd A_FP(3,3);
    Eigen::MatrixXd B_PO(3,3);
    Eigen::MatrixXd B_FP(3,3);
    Eigen::MatrixXd C_PO(3,3);
    Eigen::MatrixXd C_FP(3,3);
    Eigen::MatrixXd D_PO(3,3);
    Eigen::MatrixXd D_FP(3,3);

    Eigen::MatrixXd STM_OP(10,10);
    Eigen::MatrixXd STM_PF(10,10);
    Eigen::MatrixXd A_OP(3,3);
    Eigen::MatrixXd A_PF(3,3);
    Eigen::MatrixXd B_OP(3,3);
    Eigen::MatrixXd B_PF(3,3);
    Eigen::MatrixXd C_OP(3,3);
    Eigen::MatrixXd C_PF(3,3);
    Eigen::MatrixXd D_OP(3,3);
    Eigen::MatrixXd D_PF(3,3);

    Eigen::VectorXd accelerationsPPlus(3);
    Eigen::VectorXd accelerationsPMinus(3);
    Eigen::VectorXd velocityOPlus(3);
    Eigen::VectorXd velocityPMinus(3);
    Eigen::VectorXd velocityPPlus(3);
    Eigen::VectorXd velocityFMinus(3);



    // Assign values to variables
    STM_PO = propagatedStatesInclSTM.block(10*(k-1),1,6,6);
    STM_FP = propagatedStatesInclSTM.block(10*k,1,6,6);
    STM_OP = STM_PO.inverse();
    STM_PF = STM_FP.inverse();

//    std::cout << "STM_PO: \n" << STM_PO << std::endl
//              << "STM_FP: \n" << STM_FP << std::endl
//              << "STM_OP: \n" << STM_OP << std::endl
//              << "STM_PF: \n" << STM_PF << std::endl;

    accelerationsPPlus = computeStateDerivativeAugmented( 0.0, getFullInitialStateAugmented(initialGuess.segment(k*11,10) ) ).block(3,0,3,1);
    accelerationsPMinus = computeStateDerivativeAugmented( 0.0, getFullInitialStateAugmented(propagatedStatesInclSTM.block(((k-1)*10),0,10,1))).block(3,0,3,1);

    velocityOPlus = initialGuess.segment((11*(k-1))+3,3);
    velocityPMinus = propagatedStatesInclSTM.block((10*(k-1)+3),0,3,1);
    velocityPPlus = initialGuess.segment((11*k)+3,3);
    velocityFMinus = propagatedStatesInclSTM.block((10*(k)+3),0,3,1);

//    std::cout << "Final Acc Test: \n" << computeStateDerivativeAugmented( 0.0, getFullInitialStateAugmented(initialGuess.segment(k*11,10) ) ).block(0,0,10,1) << std::endl
//              << "accelerationsPPlus STATES: \n" << getFullInitialStateAugmented(initialGuess.segment(k*11,10) ).block(0,0,10,1) << std::endl
//              << "accelerationsPMinus ST: \n" << getFullInitialStateAugmented(propagatedStatesInclSTM.block((k-1)*10,0,10,11)).block(0,0,10,1) << std::endl
//              << "accelerationsPPlus: \n" << accelerationsPPlus << std::endl
//              << "accelerationsPMinus: \n" << accelerationsPMinus << std::endl
//              << "VelocityOPlus: \n" << velocityOPlus << std::endl
//              << "velocityPMinus: \n" << velocityPMinus << std::endl
//              << "velocityPPlus: \n" << velocityPPlus << std::endl
//              << "velocityFMinus: \n" << velocityFMinus << std::endl;

    A_PO = STM_PO.block(0,0,3,3);
    A_FP = STM_FP.block(0,0,3,3);
    B_PO = STM_PO.block(0,3,3,3);
    B_FP = STM_FP.block(0,3,3,3);
    C_PO = STM_PO.block(3,0,3,3);
    C_FP = STM_FP.block(3,0,3,3);
    D_PO = STM_PO.block(3,3,3,3);
    D_FP = STM_FP.block(3,3,3,3);

    A_OP = STM_OP.block(0,0,3,3);
    A_PF = STM_PF.block(0,0,3,3);
    B_OP = STM_OP.block(0,3,3,3);
    B_PF = STM_PF.block(0,3,3,3);
    C_OP = STM_OP.block(3,0,3,3);
    C_PF = STM_PF.block(3,0,3,3);
    D_OP = STM_OP.block(3,3,3,3);
    D_PF = STM_PF.block(3,3,3,3);

//    std::cout << "A_PO: \n" << A_PO << std::endl
//              << "B_PO: \n" << B_PO << std::endl
//              << "C_PO: \n" << C_PO << std::endl
//              << "D_PO: \n" << D_PO << std::endl
//              << "A_OP: \n" << A_OP << std::endl
//              << "B_OP: \n" << B_OP << std::endl
//              << "C_OP: \n" << C_OP << std::endl
//              << "D_OP: \n" << D_OP << std::endl
//              << "A_FP: \n" << A_FP << std::endl
//              << "B_FP: \n" << B_FP << std::endl
//              << "C_FP: \n" << C_FP << std::endl
//              << "D_FP: \n" << D_FP << std::endl
//              << "A_PF: \n" << A_PF << std::endl
//              << "B_PF: \n" << B_PF << std::endl
//              << "C_PF: \n" << C_PF << std::endl
//              << "D_PF: \n" << D_PF << std::endl;

//    std::cout << "========TESTING INVERSIBILITY ======="<< std::endl
//              << "STM_PO: \n" << STM_PO << std::endl
//              << "STM_OP \n" << STM_OP << std::endl
//              << "B_PO: \n" << B_PO << std::endl
//              << "B_OP: \n" << B_OP << std::endl
//              << "B_PO: TEST \n" << (C_OP-D_OP*B_OP.inverse()*A_OP).inverse() << std::endl
//              << "B_PO.inverse: \n" << B_PO.inverse() << std::endl
//              << "B_OP.inverse: \n" << B_OP.inverse() << std::endl
//              << "=====================================" << std::endl;

    // Compute partial derivatives
    Eigen::MatrixXd M_O(3,3);
    Eigen::MatrixXd M_timeO(3,1);
    Eigen::MatrixXd M_P(3,3);
    Eigen::MatrixXd M_timeP(3,1);
    Eigen::MatrixXd M_F(3,3);
    Eigen::MatrixXd M_timeF(3,1);

    // Compute partial derivatives
    Eigen::MatrixXd DeltaV_Rpast(3,3);
    Eigen::MatrixXd DeltaV_Timepast(3,1);
    Eigen::MatrixXd DeltaV_Rpresent(3,3);
    Eigen::MatrixXd DeltaV_Timepresent(3,1);
    Eigen::MatrixXd DeltaV_Rfuture(3,3);
    Eigen::MatrixXd DeltaV_Timefuture(3,1);

    M_O = D_PO*(B_PO.inverse())*A_PO - C_PO;
    M_timeO = accelerationsPMinus - D_PO*(B_PO.inverse())*velocityPMinus;
    M_P = D_PF*(B_PF.inverse())-D_PO*(B_PO.inverse());
    M_timeP = D_PO*(B_PO.inverse())*velocityPMinus - D_PF*(B_PF.inverse())*velocityPPlus + accelerationsPPlus - accelerationsPMinus;
    M_F = C_PF - D_PF*(B_PF.inverse())*A_PF;
    M_timeF = D_PF*(B_PF.inverse())*velocityPPlus - accelerationsPPlus;

    DeltaV_Rpast = -1.0*(B_OP.inverse());
    DeltaV_Timepast = (B_OP.inverse())*velocityOPlus;
    DeltaV_Rpresent = (B_OP.inverse())*A_OP - (B_FP.inverse())*A_FP;
    DeltaV_Timepresent = accelerationsPPlus - accelerationsPMinus + D_PO*(B_PO.inverse())*velocityPMinus
            - D_PF * (B_PF.inverse()) * velocityPPlus;
    DeltaV_Rfuture = (B_FP.inverse());
    DeltaV_Timefuture = -1.0*(B_FP.inverse())*velocityFMinus;

//    std::cout << "M_O : \n" << M_O << std::endl
//              << "M_timeO: \n" << M_timeO << std::endl
//              << "M_P: \n" << M_P << std::endl
//              << "M_timeP: \n" << M_timeP << std::endl
//              << "M_F: \n" << M_F << std::endl
//              << "M_timeF: \n" << M_timeF << std::endl;

//    std::cout << "DeltaV_Rpast : \n" << DeltaV_Rpast << std::endl
//              << "DeltaV_Timepast : \n" << DeltaV_Timepast << std::endl
//              << "DeltaV_Rpresent : \n" << DeltaV_Rpresent << std::endl
//              << "DeltaV_Timepresent : \n" << DeltaV_Timepresent << std::endl
//              << "DeltaV_Rfuture : \n" << DeltaV_Rfuture << std::endl
//              << "DeltaV_Timefuture : \n" << DeltaV_Timefuture << std::endl;

//    std::cout << "Difference derivative 1: \n" << M_O - DeltaV_Rpast << std::endl
//              << "Difference derivative 2: \n" << M_timeO - DeltaV_Timepast << std::endl
//              << "Difference derivative 3: \n" << M_P - DeltaV_Rpresent << std::endl
//              << "Difference derivative 4: \n" << M_timeP - DeltaV_Timepresent << std::endl
//              << "Difference derivative 5: \n" << M_F - DeltaV_Rfuture << std::endl
//              << "Difference derivative 6: \n" << M_timeF - DeltaV_Timefuture << std::endl;



//    std::cout << "PARTIAL DERIVATIVES COMPUTED" << std::endl;

    // Construct the updateMatrix
    Eigen::MatrixXd updateMatrixAtPatchPoint(3,12);

    updateMatrixAtPatchPoint.block(0,0,3,3) = M_O;
    updateMatrixAtPatchPoint.block(0,3,3,1) = M_timeO;
    updateMatrixAtPatchPoint.block(0,4,3,3) = M_P;
    updateMatrixAtPatchPoint.block(0,7,3,1) = M_timeP;
    updateMatrixAtPatchPoint.block(0,8,3,3) = M_F;
    updateMatrixAtPatchPoint.block(0,11,3,1) = M_timeF;

    //updateMatrixAtPatchPoint.block(0,0,3,3) = DeltaV_Rpast;
    //updateMatrixAtPatchPoint.block(0,3,3,1) = DeltaV_Timepast;
    //updateMatrixAtPatchPoint.block(0,4,3,3) = DeltaV_Rpresent;
    //updateMatrixAtPatchPoint.block(0,7,3,1) = DeltaV_Timepresent;
   //updateMatrixAtPatchPoint.block(0,8,3,3) = DeltaV_Rfuture;
    //updateMatrixAtPatchPoint.block(0,11,3,1) = DeltaV_Timepresent;

//    std::cout << "updateMatrix: \n" << updateMatrix << std::endl;
//    std::cout << "UPDATE MATRIX CONSTRUCTED" << std::endl;

    // Construct the constraint Vector (deviations in position and velocity)
    Eigen::VectorXd constraintVectorAtPatchPoint(3);
    constraintVectorAtPatchPoint = deviationVector.segment(((k-1)*11+3),3);

// std::cout << "constraintVectorAtPatchPoint: \n" << constraintVectorAtPatchPoint << std::endl;

//    // Place the constraint and update matrices at K in the large matrices
    constraintVector.segment((k-1)*3,3) = constraintVectorAtPatchPoint;
    updateMatrix.block((k-1)*3,(k-1)*4,3,12)= updateMatrixAtPatchPoint;

    if (k == 1){
        // compute accelerations
        Eigen::VectorXd startStateDerivative = computeStateDerivativeAugmented(0.0, initialGuess.block(0,0,10,1) );
        Eigen::VectorXd startStateAccelerations = startStateDerivative.block(3,0,3,1);
        Eigen::VectorXd startStateVelocities = startStateDerivative.block(0,0,3,1);

        periodicityTermR_0 = -1.0*(B_PO.inverse())*A_PO;
        periodicityTermTime_0 = startStateAccelerations - D_OP*(B_OP.inverse())*startStateVelocities;
        periodicityTermR_1 = B_PO.inverse();
        periodicityTermTime_1 = -1.0*(B_PO.inverse())*velocityPMinus;

        //std::cout << "periodicityTermR_0: \n" << periodicityTermR_0 << std::endl
        //          << "periodicityTermTime_0: \n" << periodicityTermTime_0 << std::endl
        //          << "periodicityTermR_1: \n" << periodicityTermR_1 << std::endl
        //          << "periodicityTermTime_1: \n" << periodicityTermTime_1 << std::endl;

    } else if (k == numberOfPatchPoints-2) {
        Eigen::VectorXd startStateDerivative = computeStateDerivativeAugmented(0.0, initialGuess.block(0,0,10,1) );
        Eigen::VectorXd startStateAccelerations = startStateDerivative.block(3,0,3,1);
        Eigen::VectorXd startStateVelocities = startStateDerivative.block(0,0,3,1);

        Eigen::VectorXd endStateDerivative = computeStateDerivativeAugmented(0.0, getFullInitialStateAugmented(propagatedStatesInclSTM.block(10*(k-1),0,10,1)) );
        Eigen::VectorXd endStateAccelerations = endStateDerivative.block(3,0,3,1);
        Eigen::VectorXd endStateVelocities = endStateDerivative.block(0,0,3,1);

        periodicityTermR_FinalMinus1 = -1.0*(B_PF.inverse());
        periodicityTermTime_FinalMinus1 = (B_PF.inverse())*startStateVelocities; //LIKELY ERROR PROBABLY V_N_MINUS = velocityFMinus;
        periodicityTermR_Final = (B_PF.inverse())*A_PF;
        periodicityTermTime_Final = -1.0*(endStateVelocities - D_FP*(B_FP.inverse())*endStateVelocities);

        //std::cout << "periodicityTermR_FinalMinus1: \n" << periodicityTermR_FinalMinus1 << std::endl
        //          << "periodicityTermTime_FinalMinus1: \n" << periodicityTermTime_FinalMinus1 << std::endl
        //          << "periodicityTermR_Final: \n" << periodicityTermR_Final << std::endl
        //          << "periodicityTermTime_Final: \n" << periodicityTermTime_Final << std::endl;
    }

}

//    std::cout << "constraintVector: \n" << constraintVector << std::endl
//              << "updateMatrix: \n" << updateMatrix << std::endl;

    // Add the periodicity constraints to the constraint vectors
    Eigen::Vector6d startingState = initialGuess.segment(0,6);
    Eigen::Vector6d finalState = propagatedStatesInclSTM.block((10*(numberOfPatchPoints-2)),0,6,1);
    //std::cout << "initial state of init guess: "<<  startingState << std::endl;
    //std::cout << "propagatedStatesInclSTM cjeck: \n" << propagatedStatesInclSTM << std::endl;
    //std::cout << "End state of propagated guess: "<<  finalState << std::endl;
    //std::cout << "INIT - END: "<<  startingState - finalState << std::endl;

    constraintVectorPeriodic.segment(0,3*(numberOfPatchPoints-2)) = constraintVector;
    constraintVectorPeriodic.segment(3*(numberOfPatchPoints-2),6) = (startingState - finalState);


    //std::cout << "constraintVector: \n" << constraintVector << std::endl
    //          << "constraintVectorPeriodic: \n" << constraintVectorPeriodic << std::endl;

    // Construct the two additional rows for the periodicity constraints and add them to update Matrix
    Eigen::MatrixXd periodicityMatrix1(3,4*numberOfPatchPoints);
    Eigen::MatrixXd periodicityMatrix2(3,4*numberOfPatchPoints);
    Eigen::MatrixXd identityMatrix(3,3);

    //std::cout.precision(4);
    //std::cout << "periodicityMatrix2INIT: \n" << periodicityMatrix2 << std::endl;

    identityMatrix.setIdentity();
    periodicityMatrix1.setZero();
    periodicityMatrix2.setZero();

    // fill first matrix
    periodicityMatrix1.block(0,0,3,3) = identityMatrix;
    periodicityMatrix1.block(0,((4*numberOfPatchPoints)-4),3,3) = -1.0*identityMatrix;

    //std::cout << "periodicityTermTime_1: \n" << periodicityTermTime_1 << std::endl;

    periodicityMatrix2.block(0,0,3,3) = periodicityTermR_0;
    periodicityMatrix2.block(0,3,3,1) = periodicityTermTime_0;
    periodicityMatrix2.block(0,4,3,3) = periodicityTermR_1;
    periodicityMatrix2.block(0,7,3,1) = periodicityTermTime_1;
    periodicityMatrix2.block(0,(numberOfPatchPoints*4-8),3,3) = periodicityTermR_FinalMinus1;
    periodicityMatrix2.block(0,(numberOfPatchPoints*4-5),3,1) = periodicityTermTime_FinalMinus1;
    periodicityMatrix2.block(0,(numberOfPatchPoints*4-4),3,3) = periodicityTermR_Final;
    periodicityMatrix2.block(0,(numberOfPatchPoints*4-1),3,1) = periodicityTermTime_Final;


    updateMatrixPeriodic.block(0,0,3*(numberOfPatchPoints-2),4*numberOfPatchPoints) = updateMatrix;
    updateMatrixPeriodic.block(3*(numberOfPatchPoints-2),0,3,4*numberOfPatchPoints) = periodicityMatrix1;
    updateMatrixPeriodic.block(3*(numberOfPatchPoints-1),0,3,4*numberOfPatchPoints) = periodicityMatrix2;

    //std::cout.precision(4);
    //std::cout << "periodicityMatrix1: \n" << periodicityMatrix1 << std::endl
    //          << "periodicityMatrix2: \n" << periodicityMatrix2 << std::endl;

//    // compute the corrections
    //corrections = -1.0*(updateMatrix.transpose())*(updateMatrix*(updateMatrix.transpose())).inverse()*constraintVector;
    corrections =-1.0*(updateMatrixPeriodic.transpose())*((updateMatrixPeriodic*(updateMatrixPeriodic.transpose())).inverse())*constraintVectorPeriodic;

    //std::cout.precision(4);
    //std::cout << "updateMatrix: \n" << updateMatrix << std::endl
    //          << "updateMatrixPeriodic: \n" << updateMatrixPeriodic << std::endl
    //         << "constraintsVectorPeriodic: \n" << constraintVectorPeriodic << std::endl;

    // set the results in differentialCorrection
    for (int i = 0; i < numberOfPatchPoints; i++ ){

        differentialCorrection.segment(i*11,3) = corrections.segment(i*4,3);
        differentialCorrection((i+1)*10+(i)) = corrections((i+1)*3+i);
    }

   //std::cout << "Corrections: \n" << corrections << std::endl;
   // std::cout << "differentialCorrection: \n" << differentialCorrection << std::endl;

//std::cout << "====DEBUGGING LII CORRECTOR FINISHED===" << std::endl;

    return differentialCorrection;
}

