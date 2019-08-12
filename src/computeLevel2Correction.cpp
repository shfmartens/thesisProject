#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/LU>
#include <iostream>
#include <math.h>

#include "Tudat/Astrodynamics/Gravitation/librationPoint.h"
#include "Tudat/Astrodynamics/BasicAstrodynamics/celestialBodyConstants.h"
#include "computeLevel2Correction.h"
#include "stateDerivativeModelAugmented.h"
#include "propagateOrbitAugmented.h"
#include <Eigen/Eigenvalues>


Eigen::VectorXd computeLevel2Correction( const Eigen::VectorXd deviationVector, const Eigen::MatrixXd forwardPropagatedStatesInclSTM, const Eigen::VectorXd initialGuess, const Eigen::VectorXd unitOffsetVector, const int numberOfPatchPoints, const double massParameter )
{

    // define matrices and vectors
    Eigen::VectorXd differentialCorrection( 11*numberOfPatchPoints );
    Eigen::VectorXd corrections ( 4*numberOfPatchPoints );
    Eigen::VectorXd constraintVector (3*(numberOfPatchPoints-2));
    Eigen::MatrixXd updateMatrix(3*(numberOfPatchPoints-2), 4*numberOfPatchPoints );

    Eigen::VectorXd constraintVectorPeriodic(3*numberOfPatchPoints);
    Eigen::VectorXd correctionsPeriodic ( 4*numberOfPatchPoints );
    Eigen::MatrixXd updateMatrixPeriodic((3*numberOfPatchPoints), 4*numberOfPatchPoints );

    Eigen::MatrixXd periodicityJacobianRow1 (3, 4*numberOfPatchPoints);
    Eigen::MatrixXd periodicityJacobianRow2 (3, 4*numberOfPatchPoints);

//    Eigen::VectorXd constraintVectorPhase(3*numberOfPatchPoints+2);
//    Eigen::VectorXd correctionsPhase ( 4*numberOfPatchPoints );
//    Eigen::MatrixXd updateMatrixPhase((3*numberOfPatchPoints)+2, 4*numberOfPatchPoints );
//    Eigen::MatrixXd phaseJacobianRow (1, 4*numberOfPatchPoints);
//    Eigen::MatrixXd phaseJacobianRow2 (1, 4*numberOfPatchPoints);



    differentialCorrection.setZero();
    corrections.setZero();
    correctionsPeriodic.setZero();

    constraintVector.setZero();
    updateMatrix.setZero();

    constraintVectorPeriodic.setZero();
    updateMatrixPeriodic.setZero();
    periodicityJacobianRow1.setZero();
    periodicityJacobianRow2.setZero();

//    constraintVectorPhase.setZero();
//    correctionsPhase.setZero();
//    updateMatrixPhase.setZero();
//    phaseJacobianRow.setZero();
//    phaseJacobianRow2.setZero();



    for (int k = 1; k < (numberOfPatchPoints-1); k++)
    {
        // Compute the constraintVector
        constraintVector.segment( 3*( k-1 ),3 ) = -1.0*deviationVector.segment( ( 11*( k-1 )+3 ),3) ;
        constraintVectorPeriodic.segment( 3*( k-1 ),3 ) = -1.0*deviationVector.segment( ( 11*( k-1 )+3 ),3) ;
        //constraintVectorPhase.segment( 3*( k-1 ),3 ) = -1.0*deviationVector.segment( ( 11*( k-1 )+3 ),3) ;

        // compute the states, state derivatives and relevant accelerations and velocities
        Eigen::VectorXd stateVectorPrevious = initialGuess.segment(11*(k-1),10);
        Eigen::VectorXd stateVectorPresentMinus =  forwardPropagatedStatesInclSTM.block(10*(k-1),0,10,1);
        Eigen::VectorXd stateVectorPresentPlus =   initialGuess.segment(11*k,10);
        Eigen::VectorXd stateVectorFuturePlus =   initialGuess.segment(11*(k+1),10);
        Eigen::VectorXd stateVectorFutureMinus =  forwardPropagatedStatesInclSTM.block(10*k,0,10,1);

        Eigen::VectorXd velocityPreviousPlus = stateVectorPrevious.segment(3,3);
        Eigen::VectorXd velocityPresentMinus = stateVectorPresentMinus.segment(3,3);
        Eigen::VectorXd velocityPresentPlus = stateVectorPresentPlus.segment(3,3);
        Eigen::VectorXd velocityFutureMinus = stateVectorFutureMinus.segment(3,3);
        Eigen::VectorXd velocityFuturePlus = stateVectorFuturePlus.segment(3,3);

        Eigen::MatrixXd stateDerivativePrevious = computeStateDerivativeAugmented( 0.0, getFullInitialStateAugmented( stateVectorPrevious ) );
        Eigen::MatrixXd stateDerivativePresentMinus = computeStateDerivativeAugmented( 0.0, getFullInitialStateAugmented( stateVectorPresentMinus ) );
        Eigen::MatrixXd stateDerivativePresentPlus = computeStateDerivativeAugmented( 0.0, getFullInitialStateAugmented( stateVectorPresentPlus ) );
        Eigen::MatrixXd stateDerivativeFutureMinus = computeStateDerivativeAugmented( 0.0, getFullInitialStateAugmented( stateVectorFutureMinus ) );
        Eigen::MatrixXd stateDerivativeFuturePlus = computeStateDerivativeAugmented( 0.0, getFullInitialStateAugmented( stateVectorFuturePlus ) );

        Eigen::VectorXd accelerationPreviousPlus = stateDerivativePrevious.block(3,0,3,1);
        Eigen::VectorXd accelerationPresentMinus = stateDerivativePresentMinus.block(3,0,3,1);
        Eigen::VectorXd accelerationPresentPlus = stateDerivativePresentPlus.block(3,0,3,1);
        Eigen::VectorXd accelerationFutureMinus = stateDerivativeFutureMinus.block(3,0,3,1);
        Eigen::VectorXd accelerationFuturePlus = stateDerivativeFuturePlus.block(3,0,3,1);

//        if (k == ( numberOfPatchPoints - 2 ) )
//        {

//            std::cout << "=== check stateVectors, stateDerivatives, Velocities, Accelerations at N-1,N ==="<< std::endl
//                      << "initialGuess: \n" << initialGuess << std::endl
//                      << "propagatedStatesInclSTM: \n" << forwardPropagatedStatesInclSTM << std::endl
//                      << "stateVectorPrevious: \n" << stateVectorPrevious << std::endl
//                      << "stateVectorPresentPlus: \n" << stateVectorPresentPlus << std::endl
//                     << "stateVectorPresentMinus: \n" << stateVectorPresentMinus << std::endl
//                      << "stateVectorFutureMinus: \n" << stateVectorFutureMinus << std::endl
//                      << "stateVectorFuturePlus: \n" << stateVectorFuturePlus << std::endl
//                      << "velocityPrevious: \n" << velocityPreviousPlus << std::endl
//                      << "velocityPresentPlus: \n" << velocityPresentPlus << std::endl
//                      << "velocityPresentMinus: \n" << velocityPresentMinus << std::endl
//                      << "velocityFutureMinus: \n" << velocityFutureMinus << std::endl
//                      << "velocityFuturePlus: \n" << velocityFuturePlus << std::endl

//    //                  << "constraint verification: \n" << velocityPresentMinus - velocityPresentPlus << std::endl
//                      << "stateDerivativePrevious: \n" << stateDerivativePrevious << std::endl
//                      << "stateDerivativePresentPlus: \n" << stateDerivativePresentPlus << std::endl
//                      << "stateDerivativePresentMinus: \n" << stateDerivativePresentMinus << std::endl
//                      << "stateDerivativeFutureMinus: \n" << stateDerivativeFutureMinus << std::endl
//                      << "stateDerivativeFutureMinus: \n" << stateDerivativeFuturePlus << std::endl
//                      << "accelerationPreviousPlus: \n" << accelerationPreviousPlus << std::endl
//                      << "accelerationPresentMinus: \n" << accelerationPresentMinus << std::endl
//                      << "accelerationPresentPlus: \n" << accelerationPresentPlus << std::endl
//                      << "accelerationFutureMinus: \n" << accelerationFutureMinus << std::endl

//                      << "accelerationFutureMinus: \n" << accelerationFuturePlus << std::endl

//                      << "=== states check finished === finished " << std::endl;

//        }



        // compute the state transition matrices and submatrices
        Eigen::MatrixXd stateTransitionMatrixPO = forwardPropagatedStatesInclSTM.block(10*(k-1),1,6,6);
        Eigen::MatrixXd stateTransitionMatrixFP = forwardPropagatedStatesInclSTM.block(10*(k),1,6,6);

        Eigen::MatrixXd stateTransitionMatrixPOInverse = stateTransitionMatrixPO.inverse();
        Eigen::MatrixXd stateTransitionMatrixFPInverse = stateTransitionMatrixFP.inverse();

        Eigen::MatrixXd identityMatrix(6,6);
        identityMatrix.setIdentity();
        Eigen::MatrixXd stateTransitionMatrixPOSolve = stateTransitionMatrixPO.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(identityMatrix);

        // compute submatrices
        Eigen::MatrixXd A_PO = stateTransitionMatrixPO.block(0,0,3,3);
        Eigen::MatrixXd A_OP = stateTransitionMatrixPOInverse.block(0,0,3,3);
        Eigen::MatrixXd A_FP = stateTransitionMatrixFP.block(0,0,3,3);
        Eigen::MatrixXd A_PF = stateTransitionMatrixFPInverse.block(0,0,3,3);

        Eigen::MatrixXd B_PO = stateTransitionMatrixPO.block(0,3,3,3);
        Eigen::MatrixXd B_OP = stateTransitionMatrixPOInverse.block(0,3,3,3);
        Eigen::MatrixXd B_FP = stateTransitionMatrixFP.block(0,3,3,3);
        Eigen::MatrixXd B_PF = stateTransitionMatrixFPInverse.block(0,3,3,3);

        Eigen::MatrixXd C_PO = stateTransitionMatrixPO.block(3,0,3,3);
        Eigen::MatrixXd C_OP = stateTransitionMatrixPOInverse.block(3,0,3,3);
        Eigen::MatrixXd C_FP = stateTransitionMatrixFP.block(3,0,3,3);
        Eigen::MatrixXd C_PF = stateTransitionMatrixFPInverse.block(3,0,3,3);

        Eigen::MatrixXd D_PO = stateTransitionMatrixPO.block(3,3,3,3);
        Eigen::MatrixXd D_OP = stateTransitionMatrixPOInverse.block(3,3,3,3);
        Eigen::MatrixXd D_FP = stateTransitionMatrixFP.block(3,3,3,3);
        Eigen::MatrixXd D_PF = stateTransitionMatrixFPInverse.block(3,3,3,3);

//        std::cout << "======== Check STM, inverses and submatrices computation ======" << std::endl
//                  << "stateTransitionMatrix PO: \n" << stateTransitionMatrixPO << std::endl
//                  << "stateTransitionMatrix FP: \n" << stateTransitionMatrixFP << std::endl
//                  //<< "identity Matrix: \n" << identityMatrix << std::endl
//                  << "stateTransitionMatrix PO INVERSE (OP): \n" << stateTransitionMatrixPOInverse << std::endl
//                  << "stateTransitionMatrix FP INVERSE (PF): \n" << stateTransitionMatrixFPInverse << std::endl
//                  //<< "state Transition Matrix PO SOLVE: \n" << stateTransitionMatrixPOSolve << std::endl
//                  //<< "difference INVERSE - SOLVE: \n" << stateTransitionMatrixPOInverse - stateTransitionMatrixPOSolve << std::endl;
//                  << "A_PO: \n" << A_PO << std::endl
//                  << "B_PO: \n" << B_PO << std::endl
//                  << "C_PO: \n" << C_PO << std::endl
//                  << "D_PO: \n" << D_PO << std::endl
//                  << "A_FP: \n" << A_FP << std::endl
//                  << "B_FP: \n" << B_FP << std::endl
//                  << "C_FP: \n" << C_FP << std::endl
//                  << "D_FP: \n" << D_FP << std::endl
//                  << "A_OP: \n" << A_OP << std::endl
//                  << "B_OP: \n" << B_OP << std::endl
//                  << "C_OP: \n" << C_OP << std::endl
//                  << "D_OP: \n" << D_OP << std::endl
//                  << "A_PF: \n" << A_PF << std::endl
//                  << "B_PF: \n" << B_PF << std::endl
//                  << "C_PF: \n" << C_PF << std::endl
//                  << "D_PF: \n" << D_PF << std::endl;
//std::cout << "======== COMPLETED Check STM, inverses and submatrices computation ======" << std::endl;
//                Eigen::MatrixXd B_PO_marchand = (C_OP-D_OP*(B_OP.inverse())*A_OP).inverse();
//std::cout << "======== Inverse OF STM CALCULATION CHECKS ======" << std::endl
//          << "B_PO: \n" << B_PO << std::endl
//          << "B_OP.inverse(): " << B_PO.inverse() << std::endl
//          << "B_PO marchand: \n" << B_PO_marchand << std::endl
//          << "difference in B_PO calculation: (marchand \n" << B_PO - B_PO_marchand << std::endl
//          << "difference in B_PO calculation: (OP.inv()) \n" << B_PO - B_OP.inverse() << std::endl;
//std::cout << "======== COMPLETED Inverse OF STM CALCULATION CHECKS ======" << std::endl;

        // compute the partial derivatives
        Eigen::MatrixXd derivative1_marchand = -1.0*(B_OP.inverse());
        Eigen::MatrixXd derivative1_pernicka = D_PO*(B_PO.inverse())*A_PO - C_PO;

//        std::cout << "=== Check derivative1 derivatives: === "<<std::endl
//                  << "derivative1_marchand: \n" << derivative1_marchand << std::endl
//                  << "derivative1_pernicka: \n" << derivative1_pernicka << std::endl
//                  << "derivative1_difference: \n" << derivative1_marchand - derivative1_pernicka << std::endl
//                  << "=== Check derivative1 derivatives completed ===" << std::endl;

        Eigen::MatrixXd derivative2_marchand = (B_OP.inverse())*velocityPreviousPlus;
        Eigen::MatrixXd derivative2_pernicka = accelerationPresentMinus - D_PO*(B_PO.inverse())*velocityPresentMinus;

//        std::cout << "=== Check derivative2 derivatives: === "<<std::endl
//                  << "derivative2_marchand: \n" << derivative2_marchand << std::endl
//                  << "derivative2_pernicka: \n" << derivative2_pernicka << std::endl
//                  << "derivative2_difference: \n" << derivative2_marchand - derivative2_pernicka << std::endl
//                  << "=== Check derivative2 derivatives completed ===" << std::endl;

        Eigen::MatrixXd derivative3_marchand = (B_OP.inverse())*A_OP - (B_FP.inverse())*A_FP;
        Eigen::MatrixXd derivative3_pernicka = D_PF*(B_PF.inverse()) - D_PO*(B_PO.inverse());
//        std::cout << "=== Check derivative3 derivatives: === "<<std::endl
//                  << "derivative3_marchand: \n" << derivative3_marchand << std::endl
//                  << "derivative3_pernicka: \n" << derivative3_pernicka << std::endl
//                  << "derivative3_difference: \n" << derivative3_marchand - derivative3_pernicka << std::endl
//                  << "=== Check derivative3 derivatives completed ===" << std::endl;

        Eigen::MatrixXd derivative4_marchand = accelerationPresentPlus - accelerationPresentMinus
                + D_PO*(B_PO.inverse())*velocityPresentMinus - D_PF*(B_PF.inverse())*velocityPresentPlus;

        Eigen::MatrixXd derivative4_marchand_ALT = accelerationPresentPlus - accelerationPresentMinus
                -(B_OP.inverse())*A_OP*velocityPresentMinus + (B_FP.inverse())*A_FP*velocityPresentPlus;

        Eigen::MatrixXd derivative4_pernicka = D_PO*(B_PO.inverse())*velocityPresentMinus
                - D_PF*(B_PF.inverse())*velocityPresentPlus + accelerationPresentPlus - accelerationPresentMinus;

//        std::cout << "=== Check derivative4 derivatives: === "<<std::endl
//                  << "derivative4_marchand: \n" << derivative4_marchand << std::endl
//                  << "derivative4_marchand_ALT: \n" << derivative4_marchand_ALT << std::endl
//                  << "derivative4_pernicka: \n" << derivative4_pernicka << std::endl
//                  << "derivative4_difference: \n" << derivative4_marchand - derivative4_pernicka << std::endl
//                  << "derivative4_differenceMARCHAND: \n" << derivative4_marchand - derivative4_marchand_ALT << std::endl
//                  << "derivative4_differenceALT - PER: \n" << derivative4_marchand_ALT - derivative4_pernicka << std::endl
//                  << "=== Check derivative4 derivatives completed ===" << std::endl;


        Eigen::MatrixXd derivative5_marchand = B_FP.inverse();
        Eigen::MatrixXd derivative5_pernicka = C_PF - D_PF * (B_PF.inverse()) * A_PF;

//        std::cout << "=== Check derivative5 derivatives: === "<<std::endl
//                  << "derivative5_marchand: \n" << derivative5_marchand << std::endl
//                  << "derivative5_pernicka: \n" << derivative5_pernicka << std::endl
//                  << "derivative5_difference: \n" << derivative5_marchand - derivative5_pernicka << std::endl
//                  << "=== Check derivative5 derivatives completed ===" << std::endl;

        Eigen::MatrixXd derivative6_marchand = -1.0* (B_FP.inverse())*velocityFutureMinus;
        Eigen::MatrixXd derivative6_pernicka = D_PF*(B_PF.inverse())*velocityPresentPlus - accelerationPresentPlus;

//        std::cout << "=== Check derivative6 derivatives: === "<<std::endl
//                  << "derivative6_marchand: \n" << derivative6_marchand << std::endl
//                  << "derivative6_pernicka: \n" << derivative6_pernicka << std::endl
//                  << "derivative6_difference: \n" << derivative6_marchand - derivative6_pernicka << std::endl
//                  << "=== Check derivative6 derivatives completed ===" << std::endl;

        // assemble the updateMatrix for the particular patch point and store it in the updateMatrix
        Eigen::MatrixXd updateMatrixAtPatchPoint(3,12);

        updateMatrixAtPatchPoint.setZero();
        updateMatrixAtPatchPoint.block(0,0,3,3)  = derivative1_marchand;
        updateMatrixAtPatchPoint.block(0,3,3,1)  = derivative2_marchand;
        updateMatrixAtPatchPoint.block(0,4,3,3)  = derivative3_marchand;
        updateMatrixAtPatchPoint.block(0,7,3,1)  = derivative4_marchand;
        updateMatrixAtPatchPoint.block(0,8,3,3)  = derivative5_marchand;
        updateMatrixAtPatchPoint.block(0,11,3,1) = derivative6_marchand;


        updateMatrix.block(3*(k-1),4*(k-1),3,12) = updateMatrixAtPatchPoint;
        updateMatrixPeriodic.block(3*(k-1),4*(k-1),3,12) = updateMatrixAtPatchPoint;
        //updateMatrixPhase.block(3*(k-1),4*(k-1),3,12) = updateMatrixAtPatchPoint;


        if (k == 1)
        {
            Eigen::MatrixXd identityMatrix(3,3);
            identityMatrix.setIdentity();


            //Compute periodicity  partial derivatives at point 0 (past, 1 in literature) and 1 (present, 2 in literature)
            periodicityJacobianRow1.block(0,0,3,3) = identityMatrix;

            periodicityJacobianRow2.block(0,0,3,3) = -(B_PO.inverse())*A_PO;
            periodicityJacobianRow2.block(0,3,3,1) = accelerationPreviousPlus - D_OP*(B_OP.inverse())*velocityPreviousPlus;
            periodicityJacobianRow2.block(0,4,3,3) = (B_PO.inverse());
            periodicityJacobianRow2.block(0,7,3,1) = -(B_PO.inverse())*velocityPresentMinus;

            // Compute the phase condition partial derivatives at point 0 (past, 1 in literature) and 1 (present, 2 in literature)
            double accelerationVector = unitOffsetVector.transpose() * accelerationPreviousPlus;

//            phaseJacobianRow.block(0,0,1,3) = unitOffsetVector.transpose() * (-1.0*(B_PO.inverse()) * A_PO);
//            phaseJacobianRow(0,3) = accelerationVector + unitOffsetVector.transpose() * (accelerationPreviousPlus - D_OP*(B_OP.inverse())*velocityPreviousPlus );
//            phaseJacobianRow.block(0,4,1,3) = unitOffsetVector.transpose() * (B_OP.inverse());
//            phaseJacobianRow(0,7) = unitOffsetVector.transpose() * (-1.0 * (B_PO.inverse()) *velocityFutureMinus );

            // Set the phase constraintVector
            //constraintVectorPhase(3*numberOfPatchPoints) = -1.0 * unitOffsetVector.transpose() * initialGuess.segment(3,3);
            //constraintVectorPhase(3*numberOfPatchPoints+1) = -1.0 * unitOffsetVector.transpose() * initialGuess.segment(11*(numberOfPatchPoints-1)+3,3);
        }

        if (k == (numberOfPatchPoints - 2))
        {
            Eigen::MatrixXd identityMatrix(3,3);
            identityMatrix.setIdentity();

            //Compute position partial derivatives at point 0 (past, 1 in literature) and 1 (present, 2 in literature)
            periodicityJacobianRow1.block(0,4*(k+1),3,3) += -identityMatrix;

            periodicityJacobianRow2.block(0,4*(k),3,3) += -(B_PF.inverse());
            periodicityJacobianRow2.block(0,(4*(k))+3,3,1) += (B_PF.inverse())*velocityPresentPlus;
            periodicityJacobianRow2.block(0,4*(k+1),3,3) += (B_PF.inverse())*A_PF;
            periodicityJacobianRow2.block(0,(4*(k+1))+3,3,1) += -(accelerationFutureMinus - D_FP*(B_FP.inverse())*velocityPresentMinus );

            constraintVectorPeriodic.segment(3*k,3) = -1.0*(initialGuess.segment(0,3) - stateVectorFutureMinus.segment(0,3));
            constraintVectorPeriodic.segment(3*(k+1),3) = -1.0*(initialGuess.segment(3,3) - stateVectorFutureMinus.segment(3,3));

            double accelerationVector = unitOffsetVector.transpose() * accelerationFutureMinus;


//            phaseJacobianRow2.block(0,4*(k),1,3) = unitOffsetVector.transpose() * (B_PF.inverse());
//            phaseJacobianRow2(0,(4*(k))+3) = unitOffsetVector.transpose() * -1.0*(B_PF.inverse())*velocityPresentPlus;
//            phaseJacobianRow2.block(0,4*(k+1),3,3) = unitOffsetVector.transpose() * -1.0*(B_PF.inverse())*A_PF;
//            phaseJacobianRow2(0,(4*(k+1))+3) = accelerationVector + unitOffsetVector.transpose() * (accelerationFutureMinus - D_FP*(B_FP.inverse())*velocityPresentMinus );

//            constraintVectorPhase.segment(3*k,3) = -1.0*(initialGuess.segment(0,3) - stateVectorFutureMinus.segment(0,3));
//            constraintVectorPhase.segment(3*(k+1),3) = -1.0*(initialGuess.segment(3,3) - stateVectorFutureMinus.segment(3,3));



        }


    }
    updateMatrixPeriodic.block(3*(numberOfPatchPoints-2),0,3,4*numberOfPatchPoints) = periodicityJacobianRow1;
    updateMatrixPeriodic.block(3*(numberOfPatchPoints-1),0,3,4*numberOfPatchPoints) = periodicityJacobianRow2;

//    updateMatrixPhase.block(3*(numberOfPatchPoints-2),0,3,4*numberOfPatchPoints) = periodicityJacobianRow1;
//    updateMatrixPhase.block(3*(numberOfPatchPoints-1),0,3,4*numberOfPatchPoints) = periodicityJacobianRow2;
//    updateMatrixPhase.block(3*numberOfPatchPoints,0,1,4*numberOfPatchPoints) = phaseJacobianRow;
//    updateMatrixPhase.block(3*numberOfPatchPoints+1,0,1,4*numberOfPatchPoints) = phaseJacobianRow2;




    corrections =             1.0*(updateMatrix.transpose())*(updateMatrix*(updateMatrix.transpose())).inverse()*constraintVector;
    correctionsPeriodic =     1.0*(updateMatrixPeriodic.transpose())*(updateMatrixPeriodic*(updateMatrixPeriodic.transpose())).inverse()*constraintVectorPeriodic;
    //correctionsPhase =        1.0*(updateMatrixPhase.transpose())*(updateMatrixPhase*(updateMatrixPhase.transpose())).inverse()*constraintVectorPhase;

//    std::cout.precision(5);
//    std::cout << "== Checking Phases Implementation =="<< std::endl
//              << "updateMatrixPeriodic: \n" <<  updateMatrixPeriodic << std::endl
//              << "updateMatrixPhase: \n" <<  updateMatrixPhase << std::endl
//              << "constraintVectorPeriodic: \n" << constraintVectorPeriodic << std::endl
//              << "constraintVectorPhase: \n" << constraintVectorPhase << std::endl
//              << "correctionsPeriodic: \n" <<  correctionsPeriodic << std::endl
//              << "correctionsPhase: \n" <<  correctionsPhase << std::endl;


    // Store corrections in the differentialCorrection Vector
    for (int s = 0; s < numberOfPatchPoints; s++)
    {

         differentialCorrection.segment(s*11,3) = correctionsPeriodic.segment(s*4,3);
         differentialCorrection( s*11+10 ) = correctionsPeriodic( s*4+3 );

    }


    return differentialCorrection;

}

