#include <Eigen/Core>
#include <Eigen/LU>
#include <iostream>

#include "computeLevel2TLTCorrection.h"
#include "stateDerivativeModelAugmentedVaryingMass.h"
#include "propagateMassVaryingOrbitAugmented.h"
#include "propagateOrbitAugmented.h"
#include "stateDerivativeModelAugmentedVaryingMass.h"
#include "stateDerivativeModelAugmentedTLT.h"


Eigen::VectorXd computeLevel2TLTCorrection( const Eigen::VectorXd deviationVector, const Eigen::MatrixXd propagatedStatesInclSTMThrust, const Eigen::MatrixXd propagatedStatesInclSTMCoast, const Eigen::VectorXd initialGuess, const int numberOfPatchPoints, double massParameter )
{
    // Initialize variables and set vectors and matrices to zero
    Eigen::VectorXd correctedGuess(numberOfPatchPoints*12);

    Eigen::VectorXd corrections ( 4*numberOfPatchPoints );
    Eigen::VectorXd constraintVector (3*(numberOfPatchPoints-2));
    Eigen::MatrixXd updateMatrix(3*(numberOfPatchPoints-2), 4*numberOfPatchPoints );

    Eigen::VectorXd correctionsPeriodic ( 4*numberOfPatchPoints );
    Eigen::VectorXd constraintVectorPeriodic (3*(numberOfPatchPoints));
    Eigen::MatrixXd updateMatrixPeriodic(3*(numberOfPatchPoints), 4*numberOfPatchPoints );

    Eigen::MatrixXd periodicityJacobianRow1 (3, 4*numberOfPatchPoints);
    Eigen::MatrixXd periodicityJacobianRow2 (3, 4*numberOfPatchPoints);

    correctedGuess.setZero();
    corrections.setZero();
    correctionsPeriodic.setZero();


    constraintVector.setZero();
    updateMatrix.setZero();
    constraintVectorPeriodic.setZero();
    updateMatrixPeriodic.setZero();

    periodicityJacobianRow1.setZero();
    periodicityJacobianRow2.setZero();

    for (int i = 1; i < (numberOfPatchPoints-1); i++)
    {
        // Assemble the constraint vector
        Eigen::VectorXd localDeviationVector = deviationVector.segment((i-1)*12,12);
        constraintVector.segment(3*(i-1),3) = -1.0* localDeviationVector.segment(3,3);
        constraintVectorPeriodic.segment(3*(i-1),3) = -1.0* localDeviationVector.segment(3,3);

        // Construct a matrix which holds the entries for the partial derivatives at patch point i
        Eigen::MatrixXd jacobianRow(3,12);
        jacobianRow.setZero();

        // Extract the necessary state (derivative) information from propagatedStates and initialGuess
            // Define stateVectors 0, p- and p+ and f-
            Eigen::VectorXd stateVectorO = initialGuess.segment((i-1)*12,10);
            Eigen::VectorXd stateVectorPMinus = propagatedStatesInclSTMCoast.block((i-1)*10,0,10,1);
            Eigen::VectorXd stateVectorPPlus = initialGuess.segment(i*12,10);
            Eigen::VectorXd stateVectorF = propagatedStatesInclSTMCoast.block(i*10,0,10,1);

            double identityQuantity = (propagatedStatesInclSTMCoast.block((i-1)*10,0,6,1) - propagatedStatesInclSTMThrust.block((i-1)*10,0,6,1)).norm();

            if ( identityQuantity < 1.0E-12  )
            {
                std::cout << "identity Quantity REached" << std::endl;
              stateVectorPMinus = propagatedStatesInclSTMThrust.block((i-1)*10,0,10,1);
              stateVectorF = propagatedStatesInclSTMThrust.block(i*10,0,10,1);

           }




            // Define velocities
            Eigen::VectorXd velocityO = stateVectorO.segment(3,3);
            Eigen::VectorXd velocityPMinus = stateVectorPMinus.segment(3,3);
            Eigen::VectorXd velocityPPlus = stateVectorPPlus.segment(3,3);
            Eigen::VectorXd velocityF = stateVectorF.segment(3,3);

            // define stateDerivative p- and p+ for accelerations
            Eigen::VectorXd stateDerivativeO = computeStateDerivativeAugmentedTLT( 0.0, getFullInitialStateAugmented(stateVectorO)).block(0,0,10,1);
            Eigen::VectorXd stateDerivativePMinus = computeStateDerivativeAugmentedTLT( 0.0, getFullInitialStateAugmented(stateVectorPMinus)).block(0,0,10,1);
            Eigen::VectorXd stateDerivativePPlus = computeStateDerivativeAugmentedTLT( 0.0, getFullInitialStateAugmented(stateVectorPPlus)).block(0,0,10,1);
            Eigen::VectorXd stateDerivativeF = computeStateDerivativeAugmentedTLT( 0.0, getFullInitialStateAugmented(stateVectorF)).block(0,0,10,1);
            // Compute accelerations
            Eigen::VectorXd accelerationsO = stateDerivativeO.segment(3,3);
            Eigen::VectorXd accelerationsPMinus = stateDerivativePMinus.segment(3,3);
            Eigen::VectorXd accelerationsPPlus = stateDerivativePPlus.segment(3,3);
            Eigen::VectorXd accelerationsF = stateDerivativeF.segment(3,3);


            // Define Mass Rates
            double massRateO = stateDerivativeO(9);
            double massRatePMinus = stateDerivativePMinus(9);
            double massRatePPlus = stateDerivativePPlus(9);
            double massRateF = stateDerivativeF(9);

        // define the state transitionMatrices for coast and thrust arcs
           // Define PO and PO_BAR and compute OP and OP_BAR
            Eigen::MatrixXd StateTransitionMatrixPO = propagatedStatesInclSTMThrust.block((i-1)*10,1,10,10);
            Eigen::MatrixXd StateTransitionMatrixPO_BAR = propagatedStatesInclSTMCoast.block((i-1)*10,1,10,10);

            Eigen::MatrixXd StateTransitionMatrixOP = StateTransitionMatrixPO.inverse();
            Eigen::MatrixXd StateTransitionMatrixOP_BAR = StateTransitionMatrixPO_BAR.inverse();

            //Define FP and FP_BAR and compute PF and PF_BAR
            Eigen::MatrixXd StateTransitionMatrixFP = propagatedStatesInclSTMThrust.block(i*10,1,10,10);
            Eigen::MatrixXd StateTransitionMatrixFP_BAR = propagatedStatesInclSTMCoast.block(i*10,1,10,10);

            Eigen::MatrixXd StateTransitionMatrixPF = StateTransitionMatrixFP.inverse();
            Eigen::MatrixXd StateTransitionMatrixPF_BAR = StateTransitionMatrixFP_BAR.inverse();

            // Compute submatrices
                // forward in time

                Eigen::MatrixXd A_PO = StateTransitionMatrixPO.block(0,0,3,3);
                Eigen::MatrixXd B_PO = StateTransitionMatrixPO.block(0,3,3,3);
                Eigen::MatrixXd C_PO = StateTransitionMatrixPO.block(3,0,3,3);
                Eigen::MatrixXd D_PO = StateTransitionMatrixPO.block(3,3,3,3);
                Eigen::MatrixXd E_PO = StateTransitionMatrixPO.block(0,9,3,1);
                Eigen::MatrixXd I_PO = StateTransitionMatrixPO.block(3,9,3,1);

                Eigen::MatrixXd A_PO_BAR = StateTransitionMatrixPO_BAR.block(0,0,3,3);
                Eigen::MatrixXd B_PO_BAR = StateTransitionMatrixPO_BAR.block(0,3,3,3);
                Eigen::MatrixXd C_PO_BAR = StateTransitionMatrixPO_BAR.block(3,0,3,3);
                Eigen::MatrixXd D_PO_BAR = StateTransitionMatrixPO_BAR.block(3,3,3,3);
                Eigen::MatrixXd E_PO_BAR = StateTransitionMatrixPO_BAR.block(0,9,3,1);
                Eigen::MatrixXd I_PO_BAR = StateTransitionMatrixPO_BAR.block(3,9,3,1);

                Eigen::MatrixXd A_FP = StateTransitionMatrixFP.block(0,0,3,3);
                Eigen::MatrixXd B_FP = StateTransitionMatrixFP.block(0,3,3,3);
                Eigen::MatrixXd C_FP = StateTransitionMatrixFP.block(3,0,3,3);
                Eigen::MatrixXd D_FP = StateTransitionMatrixFP.block(3,3,3,3);
                Eigen::MatrixXd E_FP = StateTransitionMatrixFP.block(0,9,3,1);
                Eigen::MatrixXd I_FP = StateTransitionMatrixFP.block(3,9,3,1);

                Eigen::MatrixXd A_FP_BAR = StateTransitionMatrixFP_BAR.block(0,0,3,3);
                Eigen::MatrixXd B_FP_BAR = StateTransitionMatrixFP_BAR.block(0,3,3,3);
                Eigen::MatrixXd C_FP_BAR = StateTransitionMatrixFP_BAR.block(3,0,3,3);
                Eigen::MatrixXd D_FP_BAR = StateTransitionMatrixFP_BAR.block(3,3,3,3);
                Eigen::MatrixXd E_FP_BAR = StateTransitionMatrixFP_BAR.block(0,9,3,1);
                Eigen::MatrixXd I_FP_BAR = StateTransitionMatrixFP_BAR.block(3,9,3,1);

                // backward in time
                Eigen::MatrixXd A_OP = StateTransitionMatrixOP.block(0,0,3,3);
                Eigen::MatrixXd B_OP = StateTransitionMatrixOP.block(0,3,3,3);
                Eigen::MatrixXd C_OP = StateTransitionMatrixOP.block(3,0,3,3);
                Eigen::MatrixXd D_OP = StateTransitionMatrixOP.block(3,3,3,3);
                Eigen::MatrixXd E_OP = StateTransitionMatrixOP.block(0,9,3,1);
                Eigen::MatrixXd I_OP = StateTransitionMatrixOP.block(3,9,3,1);

                Eigen::MatrixXd A_OP_BAR = StateTransitionMatrixOP_BAR.block(0,0,3,3);
                Eigen::MatrixXd B_OP_BAR = StateTransitionMatrixOP_BAR.block(0,3,3,3);
                Eigen::MatrixXd C_OP_BAR = StateTransitionMatrixOP_BAR.block(3,0,3,3);
                Eigen::MatrixXd D_OP_BAR = StateTransitionMatrixOP_BAR.block(3,3,3,3);
                Eigen::MatrixXd E_OP_BAR = StateTransitionMatrixOP_BAR.block(0,9,3,1);
                Eigen::MatrixXd I_OP_BAR = StateTransitionMatrixOP_BAR.block(3,9,3,1);

                Eigen::MatrixXd A_PF = StateTransitionMatrixPF.block(0,0,3,3);
                Eigen::MatrixXd B_PF = StateTransitionMatrixPF.block(0,3,3,3);
                Eigen::MatrixXd C_PF = StateTransitionMatrixPF.block(3,0,3,3);
                Eigen::MatrixXd D_PF = StateTransitionMatrixPF.block(3,3,3,3);
                Eigen::MatrixXd E_PF = StateTransitionMatrixPF.block(0,9,3,1);
                Eigen::MatrixXd I_PF = StateTransitionMatrixPF.block(3,9,3,1);

                Eigen::MatrixXd A_PF_BAR = StateTransitionMatrixPF_BAR.block(0,0,3,3);
                Eigen::MatrixXd B_PF_BAR = StateTransitionMatrixPF_BAR.block(0,3,3,3);
                Eigen::MatrixXd C_PF_BAR = StateTransitionMatrixPF_BAR.block(3,0,3,3);
                Eigen::MatrixXd D_PF_BAR = StateTransitionMatrixPF_BAR.block(3,3,3,3);
                Eigen::MatrixXd E_PF_BAR = StateTransitionMatrixPF_BAR.block(0,9,3,1);
                Eigen::MatrixXd I_PF_BAR = StateTransitionMatrixPF_BAR.block(3,9,3,1);

        // Compute partial derivatives

                // Partials w.r.t to velocityPminus
                Eigen::MatrixXd partialVelocityMinusPositionO = (A_OP*B_OP_BAR + B_OP*D_OP_BAR).inverse();
                Eigen::VectorXd partialVelocityMinusTimeO = ( (A_OP*B_OP_BAR + B_OP*D_OP_BAR).inverse() )*(E_OP*massRateO-velocityO);
                Eigen::MatrixXd partialVelocityMinusPositionP = -1.0 * ( (A_OP*B_OP_BAR + B_OP*D_OP_BAR).inverse() )* ( A_OP*A_OP_BAR + B_OP*C_OP_BAR );
                Eigen::VectorXd partialVelocityMinusTimeP = ( (A_OP*B_OP_BAR + B_OP*D_OP_BAR).inverse() )* ( A_OP*A_OP_BAR + B_OP*C_OP_BAR )*velocityPMinus + accelerationsPMinus;

                // Partials w.r.t to velocityPPlus
                Eigen::MatrixXd partialVelocityPlusPositionP = -1.0 * ( (A_FP_BAR*B_FP + B_FP_BAR * D_FP ).inverse() )* ( A_FP_BAR * A_FP + B_FP_BAR * C_FP );
                Eigen::VectorXd partialVelocityPlusTimeP = ( (A_FP_BAR*B_FP + B_FP_BAR * D_FP ).inverse() )*( ( A_FP_BAR * A_FP + B_FP_BAR * C_FP )*velocityPPlus + ( A_FP_BAR * E_FP + B_FP_BAR* I_FP )*massRatePPlus )+accelerationsPPlus;
                Eigen::MatrixXd partialVelocityPlusPositionF = (A_FP_BAR*B_FP + B_FP_BAR * D_FP ).inverse();
                Eigen::VectorXd partialVelocityPlusTimeF = -1.0*( (A_FP_BAR*B_FP + B_FP_BAR * D_FP ).inverse() )*velocityF;

        // Compute SRM Entries and store into jacobianRow

                // SRM entry w.r.t P0
                Eigen::MatrixXd partialVelocityPositionO = partialVelocityMinusPositionO;
                jacobianRow.block(0,0,3,3) = partialVelocityPositionO;

                // SRM entry w.r.t t0
                Eigen::VectorXd partialVelocityTimeO = partialVelocityMinusTimeO;
                jacobianRow.block(0,3,3,1) = partialVelocityTimeO;

                // SRM entry partial w.r.t Pp
                Eigen::MatrixXd partialVelocityPositionP = partialVelocityMinusPositionP - partialVelocityPlusPositionP;
                jacobianRow.block(0,4,3,3) = partialVelocityPositionP;

                // SRM entry w.r.t tp
                Eigen::VectorXd partialVelocityTimeP = partialVelocityMinusTimeP - partialVelocityPlusTimeP;
                jacobianRow.block(0,7,3,1) = partialVelocityTimeP;


                // SRM entry w.r.t Pf
                Eigen::MatrixXd partialVelocityPositionF = -1.0*partialVelocityPlusPositionF;
                jacobianRow.block(0,8,3,3) = partialVelocityPositionF;

                // SRM entry w.r.t tf
                Eigen::VectorXd partialVelocityTimeF = -1.0*partialVelocityPlusTimeF;
                jacobianRow.block(0,11,3,1) = partialVelocityTimeF;

        // Put the computed Jacobian row into the updateMatrix
        updateMatrix.block((i-1)*3,(i-1)*4,3,12) = jacobianRow;
        updateMatrixPeriodic.block((i-1)*3,(i-1)*4,3,12) = jacobianRow;

        if (i == 1)
        {
            // Create identity Matrix
            Eigen::MatrixXd identityMatrix(3,3);
            identityMatrix.setIdentity();

            //Compute periodicity  partial derivatives at point 0 (past, 1 in literature) and 1 (present, 2 in literature)
            periodicityJacobianRow1.block(0,0,3,3) = -identityMatrix;
            periodicityJacobianRow2.block(0,0,3,3) = ( (A_PO_BAR*B_PO + B_PO_BAR * D_PO ).inverse() )* ( A_PO_BAR * A_PO + B_PO_BAR * C_PO );
            periodicityJacobianRow2.block(0,3,3,1) = -1.0*((A_PO_BAR*B_PO + B_PO_BAR * D_PO ).inverse()*(( A_PO_BAR * A_PO + B_PO_BAR * C_PO )*velocityO + (A_PO_BAR *E_PO+B_PO_BAR*I_PO)*massRateO)+accelerationsO);
            periodicityJacobianRow2.block(0,4,3,3) = -1.0 * (A_PO_BAR*B_PO + B_PO_BAR * D_PO ).inverse();
            periodicityJacobianRow2.block(0,7,3,1) = ( (A_PO_BAR*B_PO + B_PO_BAR * D_PO ).inverse() )*velocityPMinus;



        }

        if (i == numberOfPatchPoints-2 )
        {
            Eigen::MatrixXd identityMatrix(3,3);
            identityMatrix.setIdentity();

            //Compute position partial derivatives at point 0 (past, 1 in literature) and 1 (present, 2 in literature)
            periodicityJacobianRow1.block(0,4*(i+1),3,3) += identityMatrix;
            periodicityJacobianRow2.block(0,4*(i),3,3) = (A_PF*B_PF_BAR + B_PF*D_PF_BAR).inverse();
            periodicityJacobianRow2.block(0,(4*(i))+3,3,1) = ( (A_PF*B_PF_BAR + B_PF*D_PF_BAR).inverse() )*(E_PF*massRatePPlus-velocityPPlus);
            periodicityJacobianRow2.block(0,4*(i+1),3,3) = -1.0 * ( (A_PF*B_PF_BAR + B_PF*D_PF_BAR).inverse() )* ( A_PF * A_PF_BAR + B_PF * C_PF_BAR );
            periodicityJacobianRow2.block(0,(4*(i+1))+3,3,1) = (A_PF*B_PF_BAR + B_PF*D_PF_BAR).inverse() * ( A_PF * A_PF_BAR + B_PF * C_PF_BAR ) * velocityPMinus + accelerationsF;
            // CORRECT ACCORDING TO LITERATURE: periodicityJacobianRow2.block(0,(4*(i+1))+3,3,1) = (A_PF*B_PF_BAR + B_PF*D_PF_BAR).inverse() * ( A_PF * A_PF_BAR + B_PF * C_PF_BAR ) * velocityPMinus + accelerationsF;



            constraintVectorPeriodic.segment(3*i,3) =  stateVectorF.segment(0,3) - initialGuess.segment(0,3);
            constraintVectorPeriodic.segment(3*(i+1),3) =  stateVectorF.segment(3,3) - initialGuess.segment(3,3);




        }

    }

    updateMatrixPeriodic.block(3*(numberOfPatchPoints-2),0,3,4*numberOfPatchPoints) = periodicityJacobianRow1;
    updateMatrixPeriodic.block(3*(numberOfPatchPoints-1),0,3,4*numberOfPatchPoints) = periodicityJacobianRow2;

    //std::cout.precision(5);
    //std::cout << "\nupdateMatrix TLTLT: " << updateMatrixPeriodic << std::endl;


    // Compute required corrections
    corrections = -1.0 * ( updateMatrix.transpose() ) * ( updateMatrix * ( updateMatrix.transpose() ) ).inverse() * constraintVector;

    // Add corrections to the state Vector
    correctedGuess = initialGuess;

//    std::cout << "initialGuess: \n" << initialGuess << std::endl;
//    std::cout << "corrections: \n" << corrections << std::endl;


    for (int s = 0; s < ( numberOfPatchPoints ); s++ )
    {
        Eigen::VectorXd localCorrectionVector = corrections.segment(s*4,4);
        correctedGuess.segment(12*s,3) = correctedGuess.segment(12*s,3) + localCorrectionVector.segment(0,3);
        correctedGuess(12*s+10) = correctedGuess(12*s+10) + localCorrectionVector(3);

    }

    std::cout << "correcteGuess: " << correctedGuess << std::endl;


    return correctedGuess;
}
