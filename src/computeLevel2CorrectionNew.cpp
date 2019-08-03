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



Eigen::VectorXd computeLevel2Correction( const Eigen::VectorXd deviationVector, const Eigen::MatrixXd forwardPropagatedStatesInclSTM, const Eigen::VectorXd initialGuess, const int numberOfPatchPoints, const double massParameter )
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

    differentialCorrection.setZero();
    corrections.setZero();
    correctionsPeriodic.setZero();

    constraintVector.setZero();
    updateMatrix.setZero();

    constraintVectorPeriodic.setZero();
    updateMatrixPeriodic.setZero();
    periodicityJacobianRow1.setZero();
    periodicityJacobianRow2.setZero();

    for (int k = 1; k < (numberOfPatchPoints-1); k++)
    {
        // Compute the constraintVector
        constraintVector.segment( 3*( k-1 ),3 ) = -1.0*deviationVector.segment( ( 11*( k-1 )+3 ),3) ;
        constraintVectorPeriodic.segment( 3*( k-1 ),3 ) = -1.0*deviationVector.segment( ( 11*( k-1 )+3 ),3) ;


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


        // compute the partial derivatives
        Eigen::MatrixXd derivative1_marchand = -1.0*(B_OP.inverse());
        Eigen::MatrixXd derivative1_pernicka = D_PO*(B_PO.inverse())*A_PO - C_PO;

        Eigen::MatrixXd derivative2_marchand = (B_OP.inverse())*velocityPreviousPlus;
        Eigen::MatrixXd derivative2_pernicka = accelerationPresentMinus - D_PO*(B_PO.inverse())*velocityPresentMinus;


        Eigen::MatrixXd derivative3_marchand = (B_OP.inverse())*A_OP - (B_FP.inverse())*A_FP;
        Eigen::MatrixXd derivative3_pernicka = D_PF*(B_PF.inverse()) - D_PO*(B_PO.inverse());

        Eigen::MatrixXd derivative4_marchand = accelerationPresentPlus - accelerationPresentMinus
                + D_PO*(B_PO.inverse())*velocityPresentMinus - D_PF*(B_PF.inverse())*velocityPresentPlus;

        Eigen::MatrixXd derivative4_marchand_ALT = accelerationPresentPlus - accelerationPresentMinus
                -(B_OP.inverse())*A_OP*velocityPresentMinus + (B_FP.inverse())*A_FP*velocityPresentPlus;

        Eigen::MatrixXd derivative4_pernicka = D_PO*(B_PO.inverse())*velocityPresentMinus
                - D_PF*(B_PF.inverse())*velocityPresentPlus + accelerationPresentPlus - accelerationPresentMinus;


        Eigen::MatrixXd derivative5_marchand = B_FP.inverse();
        Eigen::MatrixXd derivative5_pernicka = C_PF - D_PF * (B_PF.inverse()) * A_PF;

        Eigen::MatrixXd derivative6_marchand = -1.0* (B_FP.inverse())*velocityFutureMinus;
        Eigen::MatrixXd derivative6_pernicka = D_PF*(B_PF.inverse())*velocityPresentPlus - accelerationPresentPlus;

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


        }


    }
    updateMatrixPeriodic.block(3*(numberOfPatchPoints-2),0,3,4*numberOfPatchPoints) = periodicityJacobianRow1;
    updateMatrixPeriodic.block(3*(numberOfPatchPoints-1),0,3,4*numberOfPatchPoints) = periodicityJacobianRow2;


    corrections =             1.0*(updateMatrix.transpose())*(updateMatrix*(updateMatrix.transpose())).inverse()*constraintVector;
    correctionsPeriodic =     1.0*(updateMatrixPeriodic.transpose())*(updateMatrixPeriodic*(updateMatrixPeriodic.transpose())).inverse()*constraintVectorPeriodic;


    // Store corrections in the differentialCorrection Vector
    for (int s = 0; s < numberOfPatchPoints; s++)
    {

        differentialCorrection.segment(s*11,3) = correctionsPeriodic.segment(s*4,3);
        differentialCorrection( s*11+10 ) = correctionsPeriodic( s*4+3 );

    }

    return differentialCorrection;

}

