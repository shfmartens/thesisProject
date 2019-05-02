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
Eigen::MatrixXd updateMatrix(3*(numberOfPatchPoints-2),4*numberOfPatchPoints);
Eigen::VectorXd corrections(4*numberOfPatchPoints);

constraintVector.setZero();
updateMatrix.setZero();
differentialCorrection.setZero();
//std::cout.precision(8);
std::cout << "====DEBUGGING LII CORRECTOR===" << std::endl
//          << "DeviationVector: \n" << deviationVector << std::endl
//          << "PropagatedStatesInclSTM: \n" << propagatedStatesInclSTM << std::endl
          << "initialGuess: \n" << initialGuess << std::endl;
//          << "numberOfPatchPoints: " << numberOfPatchPoints << std::endl


// compute velocity continuity for all interior patch points
for(int k = 1; k < numberOfPatchPoints-1; k++){

    //std::cout << "CURRENT PATCH POINT: " << k << std::endl;

    // define the STM's from k-1 to k (past) and k to k+1 (future), accelerations at k, and propagatedStates
    Eigen::MatrixXd STMPast(10,10);
    Eigen::MatrixXd STMFuture(10,10);
    Eigen::MatrixXd STMPastReverse(10,10);
    Eigen::MatrixXd STMFutureReverse(10,10);
    Eigen::VectorXd accelerationsPresentPlus(3);
    Eigen::VectorXd accelerationsPresentMinus(3);
    Eigen::VectorXd velocityPastPlus(3);
    Eigen::VectorXd velocityPresentMinus(3);
    Eigen::VectorXd velocityPresentPlus(3);
    Eigen::VectorXd velocityFutureMinus(3);
    Eigen::MatrixXd Apast(3,3);
    Eigen::MatrixXd Afuture(3,3);
    Eigen::MatrixXd Bpast(3,3);
    Eigen::MatrixXd Bfuture(3,3);
    Eigen::MatrixXd Dpast(3,3);
    Eigen::MatrixXd Dfuture(3,3);
    Eigen::MatrixXd ApastReverse(3,3);
    Eigen::MatrixXd AfutureReverse(3,3);
    Eigen::MatrixXd BpastReverse(3,3);
    Eigen::MatrixXd BfutureReverse(3,3);
    Eigen::MatrixXd DpastReverse(3,3);
    Eigen::MatrixXd DfutureReverse(3,3);

    // Assign values to variables
    STMPast = propagatedStatesInclSTM.block(10*(k-1),1,10,10);
    STMFuture = propagatedStatesInclSTM.block(10*k,1,10,10);
    STMPastReverse = STMPast.inverse();
    STMFutureReverse = STMFuture.inverse();

    accelerationsPresentPlus = computeStateDerivativeAugmented( 0.0, getFullInitialStateAugmented(initialGuess.segment(k*11,10) ) ).block(3,0,3,1);
    accelerationsPresentMinus = computeStateDerivativeAugmented( 0.0, propagatedStatesInclSTM.block((k-1)*10,0,10,11)).block(3,0,3,1);

    velocityPastPlus = initialGuess.segment((11*(k-1))+3,3);
    velocityPresentMinus = propagatedStatesInclSTM.block((10*(k-1)+3),0,3,1);
    velocityPresentPlus = initialGuess.segment((11*k)+3,3);
    velocityFutureMinus = propagatedStatesInclSTM.block((10*(k)+3),0,3,1);

    Apast = STMPast.block(0,0,3,3);
    Afuture = STMFuture.block(0,0,3,3);
    Bpast = STMPast.block(0,3,3,3);
    Bfuture = STMFuture.block(0,3,3,3);
    Dpast = STMPast.block(3,3,3,3);
    Dfuture = STMFuture.block(3,3,3,3);

    ApastReverse = STMPastReverse.block(0,0,3,3);
    AfutureReverse = STMFutureReverse.block(0,0,3,3);
    BpastReverse = STMPast.block(0,3,3,3);
    BfutureReverse = STMFutureReverse.block(0,3,3,3);
    DpastReverse = STMPastReverse.block(3,3,3,3);
    DfutureReverse = STMFutureReverse.block(3,3,3,3);

//    std::cout << "STMPast: \n" << STMPast << std::endl
//              << "STMFuture \n " << STMFuture << std::endl
//              << "accelerationPresentPlus CHECK: \n" << getFullInitialStateAugmented(initialGuess.segment(k*11,10)) << std::endl
//              << "accelerationsPresentMinus: CHECK \n" << propagatedStatesInclSTM.block((k-1)*10,0,10,11) << std::endl
//              << "VelocityPastPlus: \n" << velocityPastPlus << std::endl
//              << "VelocityPresentMinus: \n" << velocityPresentMinus << std::endl
//              << "VelocityPresentPlus: \n" << velocityPresentPlus << std::endl
//              << "VelocityFutureMinus \n" << velocityFutureMinus << std::endl;
//    std::cout << "Apast: \n"  << Apast << std::endl
//              << "Afuture: \n"  << Afuture << std::endl
//              << "Bpast: \n"  << Bpast << std::endl
//              << "Bfuture: \n"  << Bfuture << std::endl
//              << "Dpast: \n"  << Dpast << std::endl
//              << "Dfuture: \n"  << Dfuture << std::endl;
//    std::cout << "VARIABLES FOR PARTIAL DERIVATIVES COMPUTED" << std::endl;


//    std::cout << "========TESTING INVERSIBILITY ======="<< std::endl
//              << "STMPast: \n" << STMPast.block(0,0,6,6) << std::endl
//              << "STMPast.inverse: \n" << (STMPast.block(0,0,6,6)).inverse() << std::endl
//              << "Bk,k-1: \n" << STMPast.block(0,3,3,3) << std::endl
//              << "Bk-1,k: \n" << ((STMPast.block(0,0,6,6)).inverse()).block(0,3,3,3) << std::endl
//              << "Bk-1,k.inverse: \n" << (((STMPast.block(0,0,6,6)).inverse()).block(0,3,3,3)).inverse() << std::endl
//              << "=====================================" << std::endl;

    // Compute partial derivatives
    Eigen::MatrixXd DeltaVelocityPastPositionDerivative(3,3);
    Eigen::MatrixXd DeltaVelocityPastTimeDerivative(3,1);
    Eigen::MatrixXd DeltaVelocityCurrentPositionDerivative(3,3);
    Eigen::MatrixXd DeltaVelocityCurrentTimeDerivative(3,1);
    Eigen::MatrixXd DeltaVelocityFuturePositionDerivative(3,3);
    Eigen::MatrixXd DeltaVelocityFutureTimeDerivative(3,1);

    DeltaVelocityPastPositionDerivative = -1.0*BpastReverse.inverse();
    DeltaVelocityPastTimeDerivative = BpastReverse.inverse() * velocityPastPlus;
    DeltaVelocityCurrentPositionDerivative = BpastReverse.inverse()*ApastReverse - Bfuture.inverse()*Afuture;
    DeltaVelocityCurrentTimeDerivative = accelerationsPresentPlus - accelerationsPresentMinus
            - BpastReverse.inverse()*ApastReverse*velocityPresentMinus + Bfuture.inverse() * Afuture * velocityPresentPlus;
    DeltaVelocityFuturePositionDerivative = Bfuture.inverse();
    DeltaVelocityFutureTimeDerivative = -1.0*Bfuture.inverse()*velocityFutureMinus;

//    std::cout << "derivative1: \n" << DeltaVelocityPastPositionDerivative << std::endl
//              << "derivative2: \n" << DeltaVelocityPastTimeDerivative << std::endl
//              << "derivative3: \n" << DeltaVelocityCurrentPositionDerivative << std::endl
//              << "derivative4: \n" << DeltaVelocityCurrentTimeDerivative << std::endl
//              << "derivative5: \n" << DeltaVelocityFuturePositionDerivative << std::endl
//              << "derivative6: \n" << DeltaVelocityFutureTimeDerivative << std::endl;

//    std::cout << "PARTIAL DERIVATIVES COMPUTED" << std::endl;

    // Construct the updateMatrix
    Eigen::MatrixXd updateMatrixAtPatchPoint(3,12);

    updateMatrixAtPatchPoint.block(0,0,3,3) = DeltaVelocityPastPositionDerivative;
    updateMatrixAtPatchPoint.block(0,3,3,1) = DeltaVelocityPastTimeDerivative;
    updateMatrixAtPatchPoint.block(0,4,3,3) = DeltaVelocityCurrentPositionDerivative;
    updateMatrixAtPatchPoint.block(0,7,3,1) = DeltaVelocityCurrentTimeDerivative;
    updateMatrixAtPatchPoint.block(0,8,3,3) = DeltaVelocityFuturePositionDerivative;
    updateMatrixAtPatchPoint.block(0,11,3,1) = DeltaVelocityFutureTimeDerivative;

//    std::cout << "updateMatrix: \n" << updateMatrix << std::endl;
//    std::cout << "UPDATE MATRIX CONSTRUCTED" << std::endl;

    // Construct the constraint Vector (deviations in position and velocity)
    Eigen::VectorXd constraintVectorAtPatchPoint(3);
    constraintVectorAtPatchPoint = deviationVector.segment(((k-1)*11+3),3);

    // Place the constraint and update matrices at K in the large matrices
    constraintVector.segment((k-1)*3,3) = constraintVectorAtPatchPoint;
    updateMatrix.block((k-1)*3,(k-1)*4,3,12)= updateMatrixAtPatchPoint;

}

    // Add the periodicity constraints to the velocity vectors
    //std::cout << "initial state of init guess: "<<  initialGuess.segment(0,6) << std::endl;
    //std::cout << "End state of propagated guess: "<<  propagatedStatesInclSTM.block(10*(numberOfPatchPoints-1),0,6,1) << std::endl;
    //constraintVector.segment((numberOfPatchPoints-1)*3,3) = initialGuess.segment(0,6) - propagatedStatesInclSTM.block(10*(numberOfPatchPoints-1),0,6,1);

    // compute the corrections
    corrections = updateMatrix.transpose()*(updateMatrix*updateMatrix.transpose()).inverse()*constraintVector;

    // set the results in differentialCorrection
    for (int i = 0; i < numberOfPatchPoints; i++ ){
        differentialCorrection.segment(i*11,3) = corrections.segment(i*4,3);
        differentialCorrection((i+1)*10+(i)) = corrections((i+1)*3+i);
    }

    //std::cout << "Corrections: \n" << corrections << std::endl;
    //std::cout << "differentialCorrection: \n" << differentialCorrection << std::endl;

    return differentialCorrection;
}
