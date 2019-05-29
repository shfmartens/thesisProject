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

Eigen::MatrixXd computeHamiltonianPositionPartials(const double massParameter, const Eigen::VectorXd stateVector )
{

    Eigen::MatrixXd outputVector(1,3);
    double xDistanceToPrimary   = stateVector(0) + massParameter;
    double xDistanceToSecondary = stateVector(0) + massParameter -1.0;
    double yDistance = stateVector(1);
    double zDistance = stateVector(2);

    double distanceToPrimary = sqrt( xDistanceToPrimary * xDistanceToPrimary + yDistance * yDistance + zDistance * zDistance );
    double distanceToSecondary= sqrt( xDistanceToSecondary * xDistanceToSecondary + yDistance * yDistance + zDistance * zDistance );

    double distanceToPrimaryCubed =   distanceToPrimary * distanceToPrimary * distanceToPrimary;
    double distanceToSecondaryCubed = distanceToSecondary * distanceToSecondary * distanceToSecondary;

    double termRelatedToPrimary = (1.0 - massParameter) / distanceToPrimaryCubed;
    double termRelatedToSecondary = (massParameter) / distanceToSecondaryCubed;

    double partialXPosition = -stateVector(0) + termRelatedToPrimary * (stateVector(0) + massParameter)
                                              + termRelatedToSecondary * (stateVector(0) + massParameter - 1.0)
                                              - stateVector(6) * cos(stateVector(8) * tudat::mathematical_constants::PI / 180.0 ) * cos(stateVector(7) * tudat::mathematical_constants::PI / 180.0 );

    double partialYPosition = -stateVector(1) + termRelatedToPrimary * (stateVector(1))
                                              + termRelatedToSecondary * (stateVector(1))
                                              - stateVector(6) * cos(stateVector(8) * tudat::mathematical_constants::PI / 180.0 ) * sin(stateVector(7) * tudat::mathematical_constants::PI / 180.0 );

    double partialZPosition =                   termRelatedToPrimary * (stateVector(2))
                                              + termRelatedToSecondary * (stateVector(2))
                                              - stateVector(6) * sin(stateVector(8) * tudat::mathematical_constants::PI / 180.0 );

    outputVector.setZero();
    outputVector(0,0) = partialXPosition;
    outputVector(0,1) = partialYPosition;
    outputVector(0,2) = partialZPosition;

    return outputVector;
}

Eigen::MatrixXd computeLLECorrection(const Eigen::MatrixXd pastStateTransitionMatrix, const::Eigen::MatrixXd futurestateTransitionMatrix, const double pastTime, const double futureTime, const bool exteriorPoint){

Eigen::MatrixXd stateTransitionMatrixSquared(10,10);
Eigen::MatrixXd identityMatrix(4,4);
double horizonTime;
double maximumEigenvalue;
double correctionFactor;
Eigen::MatrixXd outputMatrix(4,4);


// Construct 4x4 identity Matrix
identityMatrix.setIdentity();

// Compute Time horizon
horizonTime = std::abs( futureTime - pastTime );

// Compute the norm of the STM
if ( exteriorPoint ) {
    stateTransitionMatrixSquared = futurestateTransitionMatrix.transpose() * pastStateTransitionMatrix;

} else{
    stateTransitionMatrixSquared = futurestateTransitionMatrix * pastStateTransitionMatrix;

}

// compute the maximum eigenvalue
Eigen::EigenSolver< Eigen::MatrixXd > eig( stateTransitionMatrixSquared );
eig.eigenvalues().real();
maximumEigenvalue = 0.0;

for (int i = 0; i <= 5; i++) {

    if (eig.eigenvalues().real()(i) > maximumEigenvalue )
    {
        maximumEigenvalue = eig.eigenvalues().real()(i);
    }
}

//std::cout << "show eigenvalues: \n" << eig.eigenvalues() << std::endl;
//std::cout << "print max eigenvalue: \n" << maximumEigenvalue << std::endl;

// compute the outputMatrix and return
correctionFactor = (1.0 / horizonTime) * log( std::sqrt( maximumEigenvalue) );
outputMatrix = correctionFactor * identityMatrix;


return outputMatrix;
}

Eigen::VectorXd computeLevel2Correction( const Eigen::VectorXd deviationVector, const Eigen::MatrixXd forwardPropagatedStatesInclSTM, const Eigen::VectorXd initialGuess, const int numberOfPatchPoints, const double massParameter, const bool hamiltonianConstraint, Eigen::VectorXd hamiltonianDeviationVector )
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

    Eigen::VectorXd constraintVectorHamiltonian(4*numberOfPatchPoints-1);
    Eigen::VectorXd correctionsHamiltonian ( 4*numberOfPatchPoints );
    Eigen::MatrixXd updateMatrixHamiltonian(4*numberOfPatchPoints-1, 4*numberOfPatchPoints );

    Eigen::MatrixXd hamiltonianJacobian (numberOfPatchPoints-1, 4*numberOfPatchPoints);

    Eigen::MatrixXd weightingMatrix(4*numberOfPatchPoints, 4* numberOfPatchPoints);


    differentialCorrection.setZero();
    corrections.setZero();
    correctionsPeriodic.setZero();

    constraintVector.setZero();
    updateMatrix.setZero();

    constraintVectorPeriodic.setZero();
    updateMatrixPeriodic.setZero();
    periodicityJacobianRow1.setZero();
    periodicityJacobianRow2.setZero();

    constraintVectorHamiltonian.setZero();
    updateMatrixHamiltonian.setZero();
    hamiltonianJacobian.setZero();

    weightingMatrix.setZero();

    for (int k = 1; k < (numberOfPatchPoints-1); k++)
    {
        // Compute the constraintVector
        constraintVector.segment( 3*( k-1 ),3 ) = -1.0*deviationVector.segment( ( 11*( k-1 )+3 ),3) ;
        constraintVectorPeriodic.segment( 3*( k-1 ),3 ) = -1.0*deviationVector.segment( ( 11*( k-1 )+3 ),3) ;
        constraintVectorHamiltonian.segment( 3*( k-1 ),3 ) = -1.0*deviationVector.segment( ( 11*( k-1 )+3 ),3) ;


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

        // Compute Jacobian constraints for the SRM at patch point k and store in hamiltonianJacobian

        Eigen::MatrixXd partialHamiltonianVelocityPresent(1,3);
        Eigen::MatrixXd partialHamiltonianPositionPresent(1,3);

        partialHamiltonianVelocityPresent << velocityPresentPlus(0), velocityPresentPlus(1), velocityPresentPlus(2);
        partialHamiltonianPositionPresent = computeHamiltonianPositionPartials(massParameter, stateVectorPresentPlus );

        Eigen::MatrixXd derivativeHamiltonianPositionPresent = partialHamiltonianPositionPresent + partialHamiltonianVelocityPresent * (-1.0*(B_FP.inverse())*A_FP) ;
        Eigen::MatrixXd derivativeHamiltonianTimePresent     = partialHamiltonianVelocityPresent * ( accelerationPresentPlus - D_PF*(B_PF.inverse())*velocityPresentPlus );
        Eigen::MatrixXd derivativeHamiltonianPositionFuture  = partialHamiltonianVelocityPresent * ( B_FP.inverse() );
        Eigen::MatrixXd derivativeHamiltonianTimeFuture      = partialHamiltonianVelocityPresent * (-1.0*( B_FP.inverse() ) * velocityFutureMinus);

        hamiltonianJacobian.block(k,4*k,1,3) = derivativeHamiltonianPositionPresent;
        hamiltonianJacobian.block(k,4*k+3,1,1) = derivativeHamiltonianTimePresent;
        hamiltonianJacobian.block(k,4*(k+1),1,3) = derivativeHamiltonianPositionFuture;
        hamiltonianJacobian.block(k,4*(k+1)+3,1,1) = derivativeHamiltonianTimeFuture;

        // Compute the Weigthing matrix and add to large matrix
        double pastPatchPointTime = initialGuess((11*k)-1);
        double finalPatchPointTime = initialGuess((11*(k+1))+10);

        Eigen::MatrixXd weightingMatrixAtPatchPoint = computeLLECorrection( stateTransitionMatrixPO, stateTransitionMatrixFP, pastPatchPointTime, finalPatchPointTime, false);
        weightingMatrix.block(4*k,4*k,4,4) = weightingMatrixAtPatchPoint;

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

            partialHamiltonianVelocityPresent << velocityPreviousPlus(0), velocityPreviousPlus(1), velocityPreviousPlus(2);
            partialHamiltonianPositionPresent = computeHamiltonianPositionPartials(massParameter, stateVectorPrevious );

            //std::cout << "=== TESTING HAMILTONIAN CONSTRAINTS === " << std::endl
            //          << "stateVectorCheck: \n" << stateVectorPrevious << std::endl
            //          << "partialHamiltonianVelocityPresent: \n" << partialHamiltonianVelocityPresent << std::endl
            //          << "partialHamiltonianPositionPresent: \n" << partialHamiltonianPositionPresent << std::endl
            //          << "=== TEST COMPLETE ===" << std::endl;

            derivativeHamiltonianPositionPresent = partialHamiltonianPositionPresent + partialHamiltonianVelocityPresent * (-1.0*(B_PO.inverse())*A_PO) ;
            derivativeHamiltonianTimePresent     = partialHamiltonianVelocityPresent * ( accelerationPreviousPlus - D_OP*(B_OP.inverse())*velocityPreviousPlus );
            derivativeHamiltonianPositionFuture  = partialHamiltonianVelocityPresent * ( B_PO.inverse() );
            derivativeHamiltonianTimeFuture      = partialHamiltonianVelocityPresent * (-1.0*( B_PO.inverse() ) * velocityPresentMinus);

            hamiltonianJacobian.block(0,0,1,3) = derivativeHamiltonianPositionPresent;
            hamiltonianJacobian.block(0,3,1,1) = derivativeHamiltonianTimePresent;
            hamiltonianJacobian.block(0,4,1,3) = derivativeHamiltonianPositionFuture;
            hamiltonianJacobian.block(0,7,1,1) = derivativeHamiltonianTimeFuture;

            //std::cout << "=== TESTING HAMILTONIAN JACOBIAN ROW === " << std::endl
            //          << "derivativeHamiltonianPositionPresent: \n" << derivativeHamiltonianPositionPresent << std::endl
            //          << "derivativeHamiltonianTimePresent : \n" << derivativeHamiltonianTimePresent  << std::endl
            //          << "derivativeHamiltonianPositionFuture: \n" << derivativeHamiltonianPositionFuture << std::endl
            //          << "derivativeHamiltonianTimeFuture : \n" << derivativeHamiltonianTimeFuture  << std::endl
            //          << "hamiltonianJacobianRow1 : \n" << hamiltonianJacobianRow1  << std::endl
            //          << "=== TESTING HAMILTONIAN JACOBIAN ROW ===" << std::endl;

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

            constraintVectorHamiltonian.segment( 3*( k ),3 ) = -1.0*(initialGuess.segment(0,3) - stateVectorFutureMinus.segment(0,3));
            constraintVectorHamiltonian.segment( 3*(k+1),3 ) = -1.0*(initialGuess.segment(3,3) - stateVectorFutureMinus.segment(3,3));

            constraintVectorHamiltonian.segment(3*(k+2), numberOfPatchPoints-1 ) = hamiltonianDeviationVector.segment(0, numberOfPatchPoints - 1);

            //Compute energy constraint partial derivatives at terminal state (N-, in literature)

//            partialHamiltonianVelocityPresent << velocityFutureMinus(0), velocityFutureMinus(1), velocityFutureMinus(2);
//            partialHamiltonianPositionPresent = computeHamiltonianPositionPartials(massParameter, stateVectorFutureMinus );

//            Eigen::MatrixXd derivativeHamiltonianPositionPast = partialHamiltonianVelocityPresent * (B_PF.inverse()) ;
//            Eigen::MatrixXd derivativeHamiltonianTimePast     = partialHamiltonianVelocityPresent * (-1.0* ( ( B_PF.inverse() ) * velocityPresentPlus ) ) ;
//            derivativeHamiltonianPositionPresent              = partialHamiltonianPositionPresent + partialHamiltonianVelocityPresent * (-1.0 * ( ( B_PF.inverse()  ) * A_PF ) );
//            derivativeHamiltonianTimePresent                  = partialHamiltonianVelocityPresent * ( accelerationFutureMinus - D_FP * ( B_FP.inverse() ) * velocityFutureMinus );

//            hamiltonianJacobian.block(k+1,4*(k),1,3) = derivativeHamiltonianPositionPast;
//            hamiltonianJacobian.block(k+1,4*(k)+3,1,1) = derivativeHamiltonianTimePast;
//            hamiltonianJacobian.block(k+1,4*(k+1),1,3) = derivativeHamiltonianPositionPresent;
//            hamiltonianJacobian.block(k+1,4*(k+1)+3,1,1) = derivativeHamiltonianTimePresent;


        }


    }
    updateMatrixPeriodic.block(3*(numberOfPatchPoints-2),0,3,4*numberOfPatchPoints) = periodicityJacobianRow1;
    updateMatrixPeriodic.block(3*(numberOfPatchPoints-1),0,3,4*numberOfPatchPoints) = periodicityJacobianRow2;

    updateMatrixHamiltonian.block(0,0,3*(numberOfPatchPoints),4*numberOfPatchPoints) = updateMatrixPeriodic;
    updateMatrixHamiltonian.block(3*numberOfPatchPoints,0,numberOfPatchPoints-1,4*numberOfPatchPoints) = hamiltonianJacobian;

    //std::cout.precision(2);
    //std::cout << "periodicityJacobianRow1: \n" << periodicityJacobianRow1 << std::endl;
    //std::cout << "periodicityJacobianRow2: \n" << periodicityJacobianRow2 << std::endl;
    //std::cout << "constraintVector: \n" << constraintVector << std::endl;
    //std::cout << "initialGuess: \n" << initialGuess << std::endl;

    if (hamiltonianConstraint == true)
    {
        std::cout << "constraintVectorPeriodic: \n" << constraintVectorPeriodic << std::endl;
        std::cout << "constraintVectorHamiltonian: \n" << constraintVectorHamiltonian << std::endl;

        //std::cout.precision(3);
        std::cout << "updateMatrixPeriodic: \n" << updateMatrixPeriodic << std::endl;
        std::cout << "updateMatrixHamiltonian: \n" << updateMatrixHamiltonian << std::endl;

    }

    //std::cout << "updateMatrix: \n" << updateMatrix << std::endl;
    //std::cout << "updateMatrixPeriodic: \n" << updateMatrixPeriodic << std::endl;

    // compute corrections
    corrections =             1.0*(updateMatrix.transpose())*(updateMatrix*(updateMatrix.transpose())).inverse()*constraintVector;
    correctionsPeriodic =     1.0*(updateMatrixPeriodic.transpose())*(updateMatrixPeriodic*(updateMatrixPeriodic.transpose())).inverse()*constraintVectorPeriodic;
    correctionsHamiltonian =  1.0*(updateMatrixHamiltonian.transpose())*(updateMatrixHamiltonian*(updateMatrixHamiltonian.transpose())).inverse()*constraintVectorHamiltonian;
    //correctionsHamiltonian =  updateMatrixHamiltonian.inverse()*constraintVectorHamiltonian;

    //std::cout << "correctionsHamiltonian: \n " << correctionsHamiltonian << std::endl;

    // Store corrections in the differentialCorrection Vector
    for (int s = 0; s < numberOfPatchPoints; s++)
    {

        if (hamiltonianConstraint == false )
        {
            differentialCorrection.segment(s*11,3) = correctionsPeriodic.segment(s*4,3);
            differentialCorrection( s*11+10 ) = correctionsPeriodic( s*4+3 );

            //std::cout << "no constraint for energy " << std::endl;
        } else
        {

            differentialCorrection.segment(s*11,3) = correctionsHamiltonian.segment(s*4,3);
            differentialCorrection( s*11+10 ) = correctionsHamiltonian( s*4+3 );
            //std::cout << "CONSTRAINTS for energy INCLUDED " << std::endl;


        }

    }

//    Eigen::MatrixXd componentTranspose = updateMatrix.transpose();
//    Eigen::MatrixXd componentProduct = (updateMatrix* componentTranspose);
//    Eigen::MatrixXd componentInverse = componentProduct.inverse();
//    Eigen::MatrixXd componentLSQ = -1.0*componentTranspose * componentInverse;
//    Eigen::MatrixXd correctionsAlternative = componentLSQ * constraintVector;


//    std::cout << "=== check input ==="<< std::endl
//              << "numberOfPatchPoints: \n" << numberOfPatchPoints << std::endl
//              << "initialGuess: \n" << initialGuess << std::endl
//              << "deviationVector: \n" << deviationVector << std::endl
//              << "forwardPropagatedstatesInclSTM: \n " << forwardPropagatedStatesInclSTM << std::endl
//              << "=== input check finished === finished " << std::endl;


//     std::cout << "=== check elements of correction calculation ==="<< std::endl
//               << "numberOfPatchPoints: \n" << numberOfPatchPoints << std::endl
//               << "deviationVector: \n" << deviationVector << std::endl
//               << "constraintVector: \n" << constraintVector << std::endl
//               << "updateMatrix: \n" << updateMatrix << std::endl
//               << "corrections: \n" << corrections << std::endl
//               << "differentialCorrection: \n" << differentialCorrection << std::endl
//               << "=== element correction calculation check finished === finished " << std::endl;

//     std::cout << "====verify correction computation====" << std::endl
//               << "corrections: \n" << corrections << std::endl
//               << "correctionsAlternative: \n" << correctionsAlternative << std::endl
//               << "difference: " << corrections - correctionsAlternative << std::endl
//     std::cout << "====COMPLETED verify correction computation====" << std::endl;


//std::cout << " ===difference in corrections=== "<< std::endl
          //<< "corrections" << corrections << std::endl
          //<< "correctionsPeriodic " << correctionsPeriodic << std::endl
          //<< "differences (non - periodic) " << corrections - correctionsPeriodic << std::endl
          //<< " ===difference in correction Checked === "<< std::endl;


    return differentialCorrection;

}

