#include <Eigen/Core>
#include <Eigen/LU>
#include <iostream>

#include "computeLevel2MassRefinementCorrection.h"
#include "stateDerivativeModelAugmentedVaryingMass.h"
#include "propagateMassVaryingOrbitAugmented.h"
#include "propagateOrbitAugmented.h"
#include "stateDerivativeModelAugmentedVaryingMass.h"




Eigen::VectorXd computeLevel2MassRefinementCorrection( const Eigen::VectorXd deviationVector, const Eigen::MatrixXd propagatedStatesInclSTM, const Eigen::VectorXd initialGuess, const int numberOfPatchPoints, double massParameter )
{
//std::cout << "\n MASS RATE CURRENTLY SET TO ZERO !!!!!!!!!!!!!" << std::endl;
    // Initialize variables and set vectors and matrices to zero
    Eigen::VectorXd correctedGuess(numberOfPatchPoints*11);

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

    //std::cout << "REMOVEd -1.0* FROM CONSTRAINT VECTOR " << std::endl;
    for (int i = 1; i < (numberOfPatchPoints-1); i++)
    {
        // Assemble the constraint vector
        Eigen::VectorXd localDeviationVector = deviationVector.segment((i-1)*11,11);
        constraintVector.segment(3*(i-1),3) = -1.0* localDeviationVector.segment(3,3);
        constraintVectorPeriodic.segment(3*(i-1),3) = -1.0* localDeviationVector.segment(3,3);

        // Construct a matrix which holds the entries for the partial derivatives at patch point i
        Eigen::MatrixXd jacobianRow(3,12);
        jacobianRow.setZero();


        // Extract the necessary state (derivative) information from propagatedStates and initialGuess
            // Define stateVectors 0, p- and p+ and f-
            Eigen::VectorXd stateVectorO = initialGuess.segment((i-1)*11,10);
            Eigen::VectorXd stateVectorPMinus = propagatedStatesInclSTM.block((i-1)*10,0,10,1);
            Eigen::VectorXd stateVectorPPlus = initialGuess.segment(i*11,10);
            Eigen::VectorXd stateVectorF = propagatedStatesInclSTM.block(i*10,0,10,1);

            // Define velocities
            Eigen::VectorXd velocityO = stateVectorO.segment(3,3);
            Eigen::VectorXd velocityPMinus = stateVectorPMinus.segment(3,3);
            Eigen::VectorXd velocityPPlus = stateVectorPPlus.segment(3,3);
            Eigen::VectorXd velocityF = stateVectorF.segment(3,3);

            // define stateDerivative p- and p+ for accelerations
            Eigen::VectorXd stateDerivativeO = computeStateDerivativeAugmentedVaryingMass( 0.0, getFullInitialStateAugmented(stateVectorO)).block(0,0,10,1);
            Eigen::VectorXd stateDerivativePMinus = computeStateDerivativeAugmentedVaryingMass( 0.0, getFullInitialStateAugmented(stateVectorPMinus)).block(0,0,10,1);
            Eigen::VectorXd stateDerivativePPlus = computeStateDerivativeAugmentedVaryingMass( 0.0, getFullInitialStateAugmented(stateVectorPPlus)).block(0,0,10,1);
            Eigen::VectorXd stateDerivativeF = computeStateDerivativeAugmentedVaryingMass( 0.0, getFullInitialStateAugmented(stateVectorF)).block(0,0,10,1);

            //std::cout.precision(5);
            //std::cout << "stateDerivativePMinus: " << stateDerivativePMinus << std::endl;

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

//            massRateO = 0.0;
//            massRatePMinus = 0.0;
//            massRatePPlus = 0.0;
//            massRateF = 0.0;

            // define the state transitionMatrices PO and FP and compute OP and PF
            Eigen::MatrixXd StateTransitionMatrixPO = propagatedStatesInclSTM.block((i-1)*10,1,10,10);
            Eigen::MatrixXd StateTransitionMatrixFP = propagatedStatesInclSTM.block(i*10,1,10,10);
            Eigen::MatrixXd StateTransitionMatrixOP = StateTransitionMatrixPO.inverse();
            Eigen::MatrixXd StateTransitionMatrixPF = StateTransitionMatrixFP.inverse();

            // Compute submatrices
                // forward in time
                Eigen::MatrixXd A_PO = StateTransitionMatrixPO.block(0,0,3,3);
                Eigen::MatrixXd B_PO = StateTransitionMatrixPO.block(0,3,3,3);
                Eigen::MatrixXd C_PO = StateTransitionMatrixPO.block(3,0,3,3);
                Eigen::MatrixXd D_PO = StateTransitionMatrixPO.block(3,3,3,3);
                Eigen::MatrixXd E_PO = StateTransitionMatrixPO.block(0,9,3,1);
                Eigen::MatrixXd I_PO = StateTransitionMatrixPO.block(3,9,3,1);



                Eigen::MatrixXd A_FP = StateTransitionMatrixFP.block(0,0,3,3);
                Eigen::MatrixXd B_FP = StateTransitionMatrixFP.block(0,3,3,3);
                Eigen::MatrixXd C_FP = StateTransitionMatrixFP.block(3,0,3,3);
                Eigen::MatrixXd D_FP = StateTransitionMatrixFP.block(3,3,3,3);
                Eigen::MatrixXd E_FP = StateTransitionMatrixFP.block(0,9,3,1);
                Eigen::MatrixXd I_FP = StateTransitionMatrixFP.block(3,9,3,1);


                // backward in time
                Eigen::MatrixXd A_OP = StateTransitionMatrixOP.block(0,0,3,3);
                Eigen::MatrixXd B_OP = StateTransitionMatrixOP.block(0,3,3,3);
                Eigen::MatrixXd C_OP = StateTransitionMatrixOP.block(3,0,3,3);
                Eigen::MatrixXd D_OP = StateTransitionMatrixOP.block(3,3,3,3);
                Eigen::MatrixXd E_OP = StateTransitionMatrixOP.block(0,9,3,1);
                Eigen::MatrixXd I_OP = StateTransitionMatrixOP.block(3,9,3,1);

                Eigen::MatrixXd A_PF = StateTransitionMatrixPF.block(0,0,3,3);
                Eigen::MatrixXd B_PF = StateTransitionMatrixPF.block(0,3,3,3);
                Eigen::MatrixXd C_PF = StateTransitionMatrixPF.block(3,0,3,3);
                Eigen::MatrixXd D_PF = StateTransitionMatrixPF.block(3,3,3,3);
                Eigen::MatrixXd E_PF = StateTransitionMatrixPF.block(0,9,3,1);
                Eigen::MatrixXd I_PF = StateTransitionMatrixPF.block(3,9,3,1);

        // Compute partial derivatives

                // Partials w.r.t to velocityPminus
                Eigen::MatrixXd partialVelocityMinusPositionO = B_OP.inverse();
                Eigen::VectorXd partialVelocityMinusTimeO = (B_OP.inverse())*(E_OP*massRateO-velocityO);
                Eigen::MatrixXd partialVelocityMinusPositionP = -1.0 * (B_OP.inverse())*A_OP;
                Eigen::VectorXd partialVelocityMinusTimeP = (B_OP.inverse())*A_OP*velocityPMinus + accelerationsPMinus;

                // Partials w.r.t to velocityPPlus
                Eigen::MatrixXd partialVelocityPlusPositionP = -1.0 * (B_FP.inverse())*A_FP;
                Eigen::VectorXd partialVelocityPlusTimeP = (B_FP.inverse())*(A_FP*velocityPPlus + E_FP*massRatePPlus)+accelerationsPPlus;
                Eigen::MatrixXd partialVelocityPlusPositionF = (B_FP.inverse());
                Eigen::VectorXd partialVelocityPlusTimeF = -1.0*(B_FP.inverse())*velocityF;


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
            periodicityJacobianRow2.block(0,0,3,3) = B_PO.inverse()*(A_PO);
            periodicityJacobianRow2.block(0,3,3,1) = -1.0*((B_PO.inverse())*(A_PO*velocityO + E_PO*massRateO)+accelerationsO);
            periodicityJacobianRow2.block(0,4,3,3) = -1.0 * (B_PO.inverse());
            periodicityJacobianRow2.block(0,7,3,1) = (B_PO.inverse() ) * velocityPMinus;



        }

        if (i == numberOfPatchPoints-2 )
        {

            Eigen::MatrixXd identityMatrix(3,3);
            identityMatrix.setIdentity();

            //Compute position partial derivatives at point 0 (past, 1 in literature) and 1 (present, 2 in literature)
            periodicityJacobianRow1.block(0,4*(i+1),3,3) += identityMatrix;

            periodicityJacobianRow2.block(0,4*(i),3,3) = B_PF.inverse();
            periodicityJacobianRow2.block(0,(4*(i))+3,3,1) = ( B_PF.inverse() ) * (E_PF * massRatePPlus - velocityPPlus);
            periodicityJacobianRow2.block(0,4*(i+1),3,3) = -1.0*( B_PF.inverse() )*(A_PF);
            periodicityJacobianRow2.block(0,(4*(i+1))+3,3,1) =  B_PF.inverse() * A_PF * velocityPMinus + accelerationsF; //SHOULD BE B_PF.inverse() * A_PF * velocityF + accelerationsF;
            //periodicityJacobianRow2.block(0,(4*(i+1))+3,3,1) =  B_PF.inverse() * A_PF * velocityF + accelerationsF; //SHOULD BE B_PF.inverse() * A_PF * velocityF + accelerationsF;


            constraintVectorPeriodic.segment(3*i,3) =  stateVectorF.segment(0,3) - initialGuess.segment(0,3);
            constraintVectorPeriodic.segment(3*(i+1),3) =  stateVectorF.segment(3,3) - initialGuess.segment(3,3);

            //Compute periodicity  partial derivatives at point 0 (past, 1 in literature) and 1 (present, 2 in literature)

        }

    }

    updateMatrixPeriodic.block(3*(numberOfPatchPoints-2),0,3,4*numberOfPatchPoints) = periodicityJacobianRow1;
    updateMatrixPeriodic.block(3*(numberOfPatchPoints-1),0,3,4*numberOfPatchPoints) = periodicityJacobianRow2;


    // Compute required corrections
    corrections = -1.0 * ( updateMatrix.transpose() ) * ( updateMatrix * ( updateMatrix.transpose() ) ).inverse() * constraintVector;
    correctionsPeriodic = -1.0 * ( updateMatrixPeriodic.transpose() ) * ( updateMatrixPeriodic * ( updateMatrixPeriodic.transpose() ) ).inverse() * constraintVectorPeriodic;


    // Add corrections to the state Vector
    correctedGuess = initialGuess;

    for (int s = 0; s < ( numberOfPatchPoints ); s++ )
    {
        Eigen::VectorXd localCorrectionVector = correctionsPeriodic.segment(s*4,4);
        correctedGuess.segment(11*s,3) = correctedGuess.segment(11*s,3) + localCorrectionVector.segment(0,3);
        correctedGuess(11*s+10) = correctedGuess(11*s+10) + localCorrectionVector(3);
    }


    //std::cout.precision(5);
    //std::cout << "updateMatrix MV: " << updateMatrixPeriodic << std::endl;



    return correctedGuess;
}
