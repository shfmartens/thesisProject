#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/LU>
#include <iostream>

#include "computeLevel2Correction.h"
#include "stateDerivativeModelAugmented.h"
#include "propagateOrbitAugmented.h"
#include <Eigen/Eigenvalues>

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
    stateTransitionMatrixSquared = futurestateTransitionMatrix.transpose() * pastStateTransitionMatrix;

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
correctionFactor = (1.0 / horizonTime) * std::sqrt( maximumEigenvalue);
outputMatrix = correctionFactor * identityMatrix;


return outputMatrix;
}


Eigen::VectorXd computeLevel2Correction( const Eigen::VectorXd deviationVector, const Eigen::MatrixXd forwardPropagatedStatesInclSTM, const Eigen::VectorXd initialGuess, const int numberOfPatchPoints )
{
    std::cout.precision(6);

//    std::cout << "====LII INPUT ====: " << std::endl
//              << "initialStateVectors: \n" << initialGuess << std::endl
//              << "propagatedStatesInclSTM: \n" << forwardPropagatedStatesInclSTM << std::endl
//              << "deviationVector: \n" << deviationVector << std::endl
//              << "numberOfPatchPoints: " << numberOfPatchPoints << std::endl;

//    std::cout << "=== Debugging LII Correction ====" << std::endl
//              << "numberOfPatchPoints: " << numberOfPatchPoints << std::endl
//              << "forwardPropagatedStatesInclSTM: \n" << forwardPropagatedStatesInclSTM << std::endl
//    //          << "backwardPropagatedStatesInclSTM: \n" << backwardPropagatedStatesInclSTM << std::endl
//             << "initialGuess: \n" << initialGuess << std::endl;

// Initialize variables and set vectors and matrices to zero
Eigen::VectorXd differentialCorrection(11*numberOfPatchPoints);
Eigen::VectorXd corrections(4*numberOfPatchPoints);
Eigen::MatrixXd updateMatrix(3*(numberOfPatchPoints-2),4*numberOfPatchPoints);
Eigen::MatrixXd updateMatrixPeriodic(3*(numberOfPatchPoints),4*numberOfPatchPoints);
Eigen::MatrixXd periodicityConstraintRow1(3,4*numberOfPatchPoints);
Eigen::MatrixXd periodicityConstraintRow2(3,4*numberOfPatchPoints);


Eigen::VectorXd constraintVector(3*(numberOfPatchPoints-2));
Eigen::VectorXd constraintVectorPeriodic(3*numberOfPatchPoints);

Eigen::MatrixXd weightingMatrix(4*numberOfPatchPoints, 4* numberOfPatchPoints);

weightingMatrix.setZero();
updateMatrix.setZero();
updateMatrixPeriodic.setZero();
periodicityConstraintRow1.setZero();
periodicityConstraintRow2.setZero();
constraintVector.setZero();
constraintVectorPeriodic.setZero();
corrections.setZero();

for (int k = 1; k < (numberOfPatchPoints-1); k++){

    //Initialize update matrix and constraint vectors per patch points
    Eigen::MatrixXd updateMatrixAtPatchPoint(3,12);
    Eigen::VectorXd constraintVectorAtPatchPoint(3);
    Eigen::MatrixXd weightingMatrixAtPatchPoint(4,4);


    // Initialize Vectors/Matrices for the state (derivatives) and deviations
    Eigen::VectorXd stateVectorKpreviousPlus(10);
    Eigen::VectorXd stateVectorKpresentMinus(10);
    Eigen::VectorXd stateVectorKpresentPlus(10);
    Eigen::VectorXd stateVectorKfutureMinus(10);
    Eigen::VectorXd stateDerivativepreviousPlus(10);
    Eigen::VectorXd stateDerivativepresentPlus(10);
    Eigen::VectorXd stateDerivativepresentMinus(10);
    Eigen::VectorXd stateDerivativefutureMinus(10);



    stateVectorKpreviousPlus = initialGuess.segment((k-1)*11,10);
    stateVectorKpresentMinus = forwardPropagatedStatesInclSTM.block(10*(k-1),0,10,1);
    stateVectorKpresentPlus =  initialGuess.segment(k*11,10);
    stateVectorKfutureMinus =  forwardPropagatedStatesInclSTM.block(10*k,0,10,1);

    stateDerivativepreviousPlus = computeStateDerivativeAugmented(0.0, getFullInitialStateAugmented(stateVectorKpreviousPlus)).block(0,0,10,1);
    stateDerivativepresentMinus = computeStateDerivativeAugmented(0.0, getFullInitialStateAugmented(stateVectorKpresentMinus)).block(0,0,10,1);
    stateDerivativepresentPlus  = computeStateDerivativeAugmented(0.0, getFullInitialStateAugmented(stateVectorKpresentPlus)).block(0,0,10,1) ;
    stateDerivativefutureMinus  = computeStateDerivativeAugmented(0.0, getFullInitialStateAugmented(stateVectorKfutureMinus)).block(0,0,10,1) ;


//    std::cout << "stateVectorKPreviousPlus: \n" << stateVectorKpreviousPlus << std::endl
//              << "stateVectorKPresentMinus: \n" << stateVectorKpresentMinus << std::endl
//              << "stateVectorKPresentPlus: \n" << stateVectorKpresentPlus << std::endl
//              << "stateVectorKFutureMinus: \n" << stateVectorKfutureMinus << std::endl;


    // Initialize The State transition Matrices
    Eigen::MatrixXd stateTransitionMatrix_PO(6,6);
    Eigen::MatrixXd stateTransitionMatrix_FP(6,6);
    Eigen::MatrixXd stateTransitionMatrix_OP(6,6);
    Eigen::MatrixXd stateTransitionMatrix_PF(6,6);

    Eigen::MatrixXd identityMatrix(6,6);
    identityMatrix.setIdentity();

    // FORWARDS PROP
    stateTransitionMatrix_PO = forwardPropagatedStatesInclSTM.block(10*(k-1),1,6,6);
    stateTransitionMatrix_FP = forwardPropagatedStatesInclSTM.block(10*k,1,6,6);
    stateTransitionMatrix_OP = stateTransitionMatrix_PO.bdcSvd( Eigen::ComputeThinU | Eigen::ComputeThinV ).solve(identityMatrix);
    stateTransitionMatrix_PF = stateTransitionMatrix_FP.bdcSvd( Eigen::ComputeThinU | Eigen::ComputeThinV ).solve(identityMatrix);

//    std::cout << "stateTransitionMatrix_PO: \n" << stateTransitionMatrix_PO << std::endl
//              << "stateTransitionMatrix_FP: \n" << stateTransitionMatrix_FP << std::endl
//              << "stateTransitionMatrix_OP: \n" << stateTransitionMatrix_OP << std::endl
//              << "stateTransitionMatrix_PF: \n" << stateTransitionMatrix_PF << std::endl;
//        std::cout << "stateTransitionMatrix_OP.inv: \n" << stateTransitionMatrix_OP << std::endl
//                  << "stateTransitionMAtrix_OP.colPIV: \n" << stateTransitionMatrix_PO.colPivHouseholderQr().solve(identityMatrix)<< std::endl
//                  << "stateTransitionMAtrix_OP.bdscv: \n" << stateTransitionMatrix_PO.bdcSvd( Eigen::ComputeThinU | Eigen::ComputeThinV ).solve(identityMatrix)<< std::endl
//                  << "difference: inv - colPiv: "<< stateTransitionMatrix_OP - stateTransitionMatrix_PO.colPivHouseholderQr().solve(identityMatrix)  << std::endl
//                  << "difference: inv - bdscvd: " << stateTransitionMatrix_OP - stateTransitionMatrix_PO.bdcSvd( Eigen::ComputeThinU | Eigen::ComputeThinV ).solve(identityMatrix) << std::endl
//                  << "difference: colpiv - bdscvd: " << stateTransitionMatrix_PO.colPivHouseholderQr().solve(identityMatrix) - stateTransitionMatrix_PO.bdcSvd( Eigen::ComputeThinU | Eigen::ComputeThinV ).solve(identityMatrix) << std::endl;





    // Initialize the sub_elements
    Eigen::MatrixXd A_PO(3,3);Eigen::MatrixXd B_PO(3,3); Eigen::MatrixXd C_PO(3,3);Eigen::MatrixXd D_PO(3,3);
    Eigen::MatrixXd A_FP(3,3);Eigen::MatrixXd B_FP(3,3);Eigen::MatrixXd C_FP(3,3);Eigen::MatrixXd D_FP(3,3);
    Eigen::MatrixXd A_OP(3,3);Eigen::MatrixXd B_OP(3,3);Eigen::MatrixXd C_OP(3,3);Eigen::MatrixXd D_OP(3,3);
    Eigen::MatrixXd A_PF(3,3);Eigen::MatrixXd B_PF(3,3);Eigen::MatrixXd C_PF(3,3);Eigen::MatrixXd D_PF(3,3);
    Eigen::VectorXd velocityOPlus(3); Eigen::VectorXd velocityPMinus(3); Eigen::VectorXd velocityPPlus(3); Eigen::VectorXd velocityFMinus(3);
    Eigen::VectorXd accelerationsOPlus(3); Eigen::VectorXd accelerationsPMinus(3); Eigen::VectorXd accelerationsPPlus(3); Eigen::VectorXd accelerationsFMinus(3);

    A_PO = stateTransitionMatrix_PO.block(0,0,3,3);
    B_PO = stateTransitionMatrix_PO.block(0,3,3,3);
    C_PO = stateTransitionMatrix_PO.block(3,0,3,3);
    D_PO = stateTransitionMatrix_PO.block(3,3,3,3);

    A_FP = stateTransitionMatrix_FP.block(0,0,3,3);
    B_FP = stateTransitionMatrix_FP.block(0,3,3,3);
    C_FP = stateTransitionMatrix_FP.block(3,0,3,3);
    D_FP = stateTransitionMatrix_FP.block(3,3,3,3);

    A_OP = stateTransitionMatrix_OP.block(0,0,3,3);
    B_OP = stateTransitionMatrix_OP.block(0,3,3,3);
    C_OP = stateTransitionMatrix_OP.block(3,0,3,3);
    D_OP = stateTransitionMatrix_OP.block(3,3,3,3);

    A_PF = stateTransitionMatrix_PF.block(0,0,3,3);
    B_PF = stateTransitionMatrix_PF.block(0,3,3,3);
    C_PF = stateTransitionMatrix_PF.block(3,0,3,3);
    D_PF = stateTransitionMatrix_PF.block(3,3,3,3);

//    std::cout << "SUBMATRICES" << std::endl
//              << "A_PO: \n" << A_PO << std::endl
//              << "B_PO: \n" << B_PO << std::endl
//              << "C_PO: \n" << C_PO << std::endl
//              << "D_PO: \n" << D_PO << std::endl
//              << "A_FP: \n" << A_FP << std::endl
//              << "B_FP: \n" << B_FP << std::endl
//              << "C_FP: \n" << C_FP << std::endl
//              << "D_FP: \n" << D_FP << std::endl
//              << "A_OP: \n" << A_OP << std::endl
//              << "B_OP: \n" << B_OP << std::endl
//              << "C_OP: \n" << C_OP << std::endl
//              << "D_OP: \n" << D_OP << std::endl
//              << "A_PF: \n" << A_PF << std::endl
//              << "B_PF: \n" << B_PF << std::endl
//              << "C_PF: \n" << C_PF << std::endl
//              << "D_PF: \n" << D_PF << std::endl;


    velocityOPlus = stateVectorKpreviousPlus.segment(3,3);
    velocityPMinus = stateVectorKpresentMinus.segment(3,3);
    velocityPPlus = stateVectorKpresentPlus.segment(3,3);
    velocityFMinus = stateVectorKfutureMinus.segment(3,3);

    accelerationsOPlus = stateDerivativepreviousPlus.segment(3,3);
    accelerationsPMinus = stateDerivativepresentMinus.segment(3,3);
    accelerationsPPlus = stateDerivativepresentPlus.segment(3,3);
    accelerationsFMinus = stateDerivativefutureMinus.segment(3,3);

//    std::cout << "VelocityOplus: \n"<< velocityOPlus <<std::endl
//              << "VelocityPminus: \n" << velocityPMinus << std::endl
//              << "VelocityPPlus: \n" << velocityPPlus << std::endl
//              << "VelocityFminus: \n" << velocityFMinus << std::endl
//              << "stateDerivativepastPlus" << stateDerivativepreviousPlus << std::endl
//              << "stateDerivativepresentMinus: \n" << stateDerivativepresentMinus << std::endl
//              << "stateDerivativepresentPlus: \n" << stateDerivativepresentPlus << std::endl
//              << "stateDerivativefutureMinus: \n" << stateDerivativefutureMinus << std::endl

//              << "accelerationsOPlus : \n" << accelerationsOPlus << std::endl
//              << "accelerationsPMinus: \n" << accelerationsPMinus << std::endl
//              << "accelerationsPPlus: \n" << accelerationsPPlus << std::endl
//              << "accelerationsFMinus: \n" << accelerationsFMinus << std::endl;




    // Compute derivatives
    Eigen::MatrixXd derivative1(3,3);
    Eigen::MatrixXd derivative2(3,1);
    Eigen::MatrixXd derivative3(3,3);
    Eigen::MatrixXd derivative4(3,1);
    Eigen::MatrixXd derivative5(3,3);
    Eigen::MatrixXd derivative6(3,1);

    Eigen::MatrixXd derivative1_pernicka(3,3);
    Eigen::MatrixXd derivative2_pernicka(3,1);
    Eigen::MatrixXd derivative3_pernicka(3,3);
    Eigen::MatrixXd derivative4_pernicka(3,1);
    Eigen::MatrixXd derivative5_pernicka(3,3);
    Eigen::MatrixXd derivative6_pernicka(3,1);

    Eigen::MatrixXd derivative1_marchand(3,3);
    Eigen::MatrixXd derivative2_marchand(3,1);
    Eigen::MatrixXd derivative3_marchand(3,3);
    Eigen::MatrixXd derivative4_marchand(3,1);
    Eigen::MatrixXd derivative4_marchand_alternative(3,1);
    Eigen::MatrixXd derivative5_marchand(3,3);
    Eigen::MatrixXd derivative6_marchand(3,1);

//    // method Wilson & Howell 1998
    derivative1 = -1.0*(B_OP.inverse());
    derivative2 = (B_OP.inverse())*velocityOPlus;
    derivative3 = -1.0*(B_FP.inverse())*A_FP + (B_OP.inverse())*A_OP;
    derivative4 = (B_FP.inverse())*A_FP*velocityPPlus - (B_OP.inverse())*A_OP*velocityPMinus;
    derivative5 = (B_FP.inverse());
    derivative6 = -1.0*(B_FP.inverse())*velocityFMinus;

    // method Pernicka & Howell 1974
    derivative1_pernicka = D_PO*(B_PO.inverse())*A_PO-C_PO;
    derivative2_pernicka = accelerationsPMinus - D_PO*(B_PO.inverse())*velocityPMinus;
    derivative3_pernicka = D_PF*(B_PF.inverse()) - D_PO*(B_PO.inverse());
    derivative4_pernicka = D_PO*(B_PO.inverse())*velocityPMinus - D_PF*(B_PF.inverse())*velocityPPlus
            + accelerationsPPlus - accelerationsPMinus;
    derivative5_pernicka = C_PF - D_PF*(B_PF.inverse())*A_PF;
    derivative6_pernicka = D_PF*(B_PF.inverse())*velocityPPlus - accelerationsPPlus;

    //std::cout << "D_PF: " << D_PF   << std::endl;
    //std::cout << "B_PF.inverse(): " << B_PF.inverse()   << std::endl;
   // std::cout << "B_PO.inverse(): " << B_PO.inverse()   << std::endl;
    //std::cout << "D_PO: " << D_PO.inverse()   << std::endl;

    // Method Marchand & Howell 2010
    derivative1_marchand = -1.0*(B_OP.inverse());
    derivative2_marchand = (B_OP.inverse())*velocityOPlus;
    derivative3_marchand = (B_OP.inverse())*A_OP - (B_FP.inverse())*A_FP;
    derivative4_marchand = accelerationsPPlus - accelerationsPMinus - (B_OP.inverse())*A_OP.inverse()*velocityPMinus + (B_FP.inverse())*A_FP*velocityPPlus;
    derivative4_marchand_alternative = accelerationsPPlus - accelerationsPMinus + D_PO*(B_PO.inverse())*velocityPMinus - D_PF*(B_PF.inverse())*velocityPPlus;
    derivative5_marchand = B_FP.inverse();
    derivative6_marchand = -1.0*(B_FP.inverse())*velocityFMinus;

    //std::cout << "derivative1: \n" << derivative1_pernicka << std::endl
    //          << "derivative2: \n" << derivative2_pernicka << std::endl
    //          << "derivative3: \n" << derivative3_pernicka << std::endl
     //         << "derivative4: \n" << derivative4_pernicka << std::endl
     //         << "derivative5: \n" << derivative5_pernicka << std::endl
     //         << "derivative6: \n" << derivative6_pernicka << std::endl;


    // compute the update Matrix
    updateMatrixAtPatchPoint.block(0,0,3,3) = derivative1_marchand;
    updateMatrixAtPatchPoint.block(0,3,3,1) = derivative2_marchand;
    updateMatrixAtPatchPoint.block(0,4,3,3) = derivative3_marchand;
    updateMatrixAtPatchPoint.block(0,7,3,1) = derivative4_marchand;
    updateMatrixAtPatchPoint.block(0,8,3,3) = derivative5_marchand;
    updateMatrixAtPatchPoint.block(0,11,3,1) = derivative6_marchand;
     if (k == 1000 )
     {

         std::cout << "accelerationsPMinus: \n" << accelerationsPMinus << std::endl;
         std::cout << "accelerationsPPlus: \n" << accelerationsPPlus << std::endl;
         std::cout << "alternative MArchand derivative  difference (imp - alt): \n" << derivative4_marchand - derivative4_marchand_alternative << std::endl;
         std::cout << "derivative 4 pernicka - marchand_ALT: \n" << derivative4_pernicka- derivative4_marchand_alternative << std::endl;
         std::cout << "derivative 4 marchand_ALT - wilson: \n" << derivative4_marchand_alternative- derivative4 << std::endl;




         std::cout << "dervative 1: pernicka - Marchand: \n" << derivative1_pernicka - derivative1_marchand << std::endl
                   << "dervative 2: pernicka - Marchand: \n" << derivative2_pernicka - derivative2_marchand << std::endl
                   << "dervative 3: pernicka - Marchand: \n" << derivative3_pernicka - derivative3_marchand << std::endl
                   << "dervative 4: pernicka - Marchand: \n" << derivative4_pernicka - derivative4_marchand << std::endl
                   << "dervative 5: pernicka - Marchand: \n" << derivative5_pernicka - derivative5_marchand << std::endl
                   << "dervative 6: pernicka - Marchand: \n" << derivative6_pernicka - derivative6_marchand << std::endl;

         std::cout << "dervative 1: pernicka - Wilson: \n" << derivative1_pernicka - derivative1 << std::endl
                   << "dervative 2: pernicka - Wilson: \n" << derivative2_pernicka - derivative2 << std::endl
                   << "dervative 3: pernicka - Wilson: \n" << derivative3_pernicka - derivative3 << std::endl
                   << "dervative 4: pernicka - Wilson: \n" << derivative4_pernicka - derivative4 << std::endl
                   << "dervative 5: pernicka - Wilson: \n" << derivative5_pernicka - derivative5 << std::endl
                   << "dervative 6: pernicka - Wilson: \n" << derivative6_pernicka - derivative6 << std::endl;

         std::cout << "dervative 1: Marchand - Wilson: \n" << derivative1_marchand - derivative1 << std::endl
                   << "dervative 2: Marchand - Wilson: \n" << derivative2_marchand - derivative2 << std::endl
                   << "dervative 3: Marchand - Wilson: \n" << derivative3_marchand - derivative3 << std::endl
                   << "dervative 4: Marchand - Wilson: \n" << derivative4_marchand - derivative4 << std::endl
                   << "dervative 5: Marchand - Wilson: \n" << derivative5_marchand - derivative5 << std::endl
                   << "dervative 6: Marchand - Wilson: \n" << derivative6_marchand - derivative6 << std::endl;

     }

    // Compute the constraintVector
    // Wilson And Howell 1998
    constraintVectorAtPatchPoint = ( velocityPPlus - velocityPMinus);

    // Compute the Weigthing matrix
    double pastPatchPointTime = initialGuess((11*k)-1);
    double finalPatchPointTime = initialGuess((11*(k+1))+10);

    weightingMatrixAtPatchPoint = computeLLECorrection( stateTransitionMatrix_PO, stateTransitionMatrix_FP, pastPatchPointTime, finalPatchPointTime, false);

    //Put the update and constraint matrices in the end matrices
    updateMatrix.block(3*(k-1),4*(k-1),3,12) = updateMatrixAtPatchPoint;
    constraintVector.segment(3*(k-1),3) = deviationVector.segment(((k-1)*11)+3,3); // OR +?
    weightingMatrix.block(4*k,4*k,4,4) = weightingMatrixAtPatchPoint;

    //std::cout << "updateMatrixAtPatchPoint: \n" <<updateMatrixAtPatchPoint << std::endl;
    //std::cout << "constraintVectorAtPatchPoint: \n" << constraintVectorAtPatchPoint << std::endl;
    std::cout << "deviationVector: \n" << deviationVector << std::endl;

    if (k == 1)
    {
        // Add Periodicity constraints
        Eigen::MatrixXd identityMatrix(3,3);
        identityMatrix.block(0,0,3,3).setIdentity();

        // fill first row
        periodicityConstraintRow1.block(0,0,3,3) = identityMatrix;
        periodicityConstraintRow1.block(0,4*(numberOfPatchPoints-1),3,3) = -1.0 * identityMatrix;

        periodicityConstraintRow2.block(0,0,3,3) = -1.0*(B_PO.inverse())*A_PO;
        periodicityConstraintRow2.block(0,3,3,1) = accelerationsOPlus - D_OP*(B_PO.inverse())*velocityOPlus;
        periodicityConstraintRow2.block(0,4,3,3) = (B_PO.inverse());
        periodicityConstraintRow2.block(0,7,3,1) = -1.0*(B_PO.inverse())*velocityPMinus;

        // Add patch point weigthing matrices to weigthing matrices
        Eigen::MatrixXd weightingAtFirstPatchPoint(4,4);
        weightingAtFirstPatchPoint = computeLLECorrection(stateTransitionMatrix_PO, stateTransitionMatrix_PO, initialGuess(10), initialGuess(21), true);
        weightingMatrix.block(0,0,4,4) = weightingAtFirstPatchPoint;

        //std::cout << "weighting At First Patch Point: \n" << weightingAtFirstPatchPoint << std::endl;

    }

    if (k == (numberOfPatchPoints-2))
    {
        Eigen::MatrixXd identityWeightingMatrix(4,4);
        identityWeightingMatrix.block(0,0,4,4).setIdentity();

        // Add Periodicity constraints
        periodicityConstraintRow2.block(0,4*k,3,3) = -1.0*(B_PF.inverse());
        periodicityConstraintRow2.block(0,4*k+3,3,1) = (B_PF.inverse())*initialGuess.segment(3,3);
        periodicityConstraintRow2.block(0,4*(k+1),3,3) = (B_PF.inverse())*A_PF;
        periodicityConstraintRow2.block(0,4*(k+1)+3,3,1) = -1.0*(accelerationsFMinus - D_FP*(B_FP.inverse()* velocityFMinus));
        constraintVectorPeriodic.segment(3*(numberOfPatchPoints-2),6) = deviationVector.segment((numberOfPatchPoints-2)*11,6);

        //std::cout << "constraintVectorPeriodic positions"<< constraintVectorPeriodic.segment(3*(numberOfPatchPoints-2),3) <<std::endl;
        //std::cout << "constraintVectorPeriodic velocities"<< constraintVectorPeriodic.segment(3*(numberOfPatchPoints-1),3) <<std::endl;

        // Add patch point weigthing matrices to weigthing matrices
        Eigen::MatrixXd weightingAtFinalPatchPoint(4,4);
        weightingAtFinalPatchPoint = computeLLECorrection(stateTransitionMatrix_FP, stateTransitionMatrix_FP, initialGuess(11*(numberOfPatchPoints-2)+10), initialGuess(11*(numberOfPatchPoints-1)+10), true );
        weightingMatrix.block(4*(numberOfPatchPoints-1),4*(numberOfPatchPoints-1),4,4) = weightingAtFinalPatchPoint;


        //std::cout << "weighting At First Patch Point: \n" << weightingAtFinalPatchPoint << std::endl;


    }


}

//std::cout << "weighting Matrix complete: \n" << weightingMatrix << std::endl;
//std::cout << "initialGuess: \n" << initialGuess << std::endl;
//std::cout << "propagatedStatesInclSTM: \n" << forwardPropagatedStatesInclSTM << std::endl;



// construct the constraint vector with periodicities included
constraintVectorPeriodic.segment(0,3*(numberOfPatchPoints-2)) = constraintVector;

// construct the updateMatrix with periodicities included
updateMatrixPeriodic.block(0,0,3*(numberOfPatchPoints-2),4*numberOfPatchPoints) = updateMatrix;
updateMatrixPeriodic.block(3*(numberOfPatchPoints-2),0,3,4*numberOfPatchPoints) = periodicityConstraintRow1;
updateMatrixPeriodic.block(3*(numberOfPatchPoints-1),0,3,4*numberOfPatchPoints) = periodicityConstraintRow2;
std::cout.precision(4);

//std::cout << "PERIODICITY TEST: " << periodicityConstraintRow1 << std::endl;
//std::cout << "PERIODICITY TEST ROW 2: " << periodicityConstraintRow2 << std::endl;
//std::cout << "constraintVector: " << constraintVector << std::endl;
std::cout << "constraintVectorPeriodic: " << constraintVectorPeriodic << std::endl;

//std::cout << "updateMatrixPeriodic: " << updateMatrixPeriodic << std::endl;
//std::cout << "updateMatrix: " << updateMatrix << std::endl;



// Compute corrections
corrections = (weightingMatrix.inverse())*(updateMatrixPeriodic.transpose())*((updateMatrixPeriodic*(weightingMatrix.inverse())*(updateMatrixPeriodic.transpose())).inverse())*constraintVectorPeriodic; //For Vp+ - Vp-
//corrections = updateMatrix.bdcSvd( Eigen::ComputeThinU | Eigen::ComputeThinV).solve(constraintVector); //For Vp+ - Vp-

//std::cout.precision(4);
//std::cout << "updateMatrix: \n" <<updateMatrix << std::endl;
//std::cout << "updateMatrixTranspose: \n" <<updateMatrix.transpose() << std::endl;
//std::cout << "updateMatrixTransposeInPlace: \n" << updateMatrix.transpose() << std::endl;


//std::cout << "constraintVector: \n" << constraintVector << std::endl;
//std::cout << "corrections: \n" << corrections << std::endl;


differentialCorrection.setZero();

// Add corrections to outputVector
for(int i = 0; i < (numberOfPatchPoints); i++){

    differentialCorrection.segment(11*i,3) = corrections.segment((4*i),3);

    //differentialCorrection((10*(i+1)+(i))) = 0.0;

    differentialCorrection((10*(i+1)+(i))) = corrections(4*i+3);


}

//std::cout << "differentialCorrection: \n" << differentialCorrection << std::endl;

    return differentialCorrection;
}

