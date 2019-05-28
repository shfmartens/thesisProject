#include <Eigen/Core>
#include <Eigen/LU>
#include <iostream>

#include "computeLevel1Correction.h"
#include "stateDerivativeModelAugmented.h"
#include "propagateOrbitAugmented.h"



Eigen::VectorXd computeLevel1Correction( const Eigen::VectorXd deviationVector, const Eigen::MatrixXd propagatedStatesInclSTM, const Eigen::VectorXd initialGuess, const int numberOfPatchPoints )
{

// Initialize variables and set vectors and matrices to zero
int numberOfArcs = numberOfPatchPoints - 1;
Eigen::VectorXd differentialCorrection(numberOfPatchPoints*11);
Eigen::MatrixXd updateMatrix(numberOfArcs*3,numberOfArcs*3);
Eigen::VectorXd constraintVector(numberOfArcs*3);
Eigen::VectorXd corrections(numberOfArcs*3);

Eigen::MatrixXd updateMatrixLSQ(numberOfArcs*3,numberOfArcs*4);
Eigen::VectorXd constraintVectorLSQ(numberOfArcs*4);
Eigen::VectorXd correctionsLSQ(numberOfArcs*4);


differentialCorrection.setZero();
updateMatrix.setZero();
constraintVector.setZero();

updateMatrixLSQ.setZero();


// store results into the outputVector

// Construct the constraint and update matrix
for(int i = 0; i <= numberOfPatchPoints-2; i++ ) {

 Eigen::VectorXd stateVectorP = propagatedStatesInclSTM.block(i*10,0,10,1);
 Eigen::MatrixXd statedervativeP = computeStateDerivativeAugmented( 0.0, getFullInitialStateAugmented( stateVectorP ) );
 Eigen::VectorXd accelerationsP = statedervativeP.block(3,0,3,1);

 constraintVector.segment(i*3,3) = deviationVector.segment(i*11,3);
 updateMatrix.block(i*3,i*3,3,3) = propagatedStatesInclSTM.block(i*10,4,3,3); // OR INVERSE MATRICES?

 updateMatrixLSQ.block(i*3,i*4,3,3) = propagatedStatesInclSTM.block(i*10,4,3,3);
 updateMatrixLSQ.block(i*3,i*4+3,3,1) = accelerationsP;

}

//std::cout.precision(6);
//std::cout << "updateMatrix: \n" << updateMatrix << std::endl;
//std::cout << "updateMatrixLSQ: \n" << updateMatrixLSQ << std::endl;

//std::cout << "updateMatrix.inverse(): \n" << updateMatrix.inverse() << std::endl;

// compute the correction
corrections = updateMatrix.inverse() * constraintVector;
correctionsLSQ = (updateMatrixLSQ.transpose())*((updateMatrixLSQ*(updateMatrixLSQ.transpose())).inverse())*constraintVector;

//std::cout << "correctionsLSQ: \n" << correctionsLSQ << std::endl;

// store results into the outputVector
for(int i = 0; i <= numberOfPatchPoints-2; i++ ) {

    differentialCorrection.segment(((11*i)+3),3) = corrections.segment(i*3,3);

    //differentialCorrection.segment(((11*i)+3),3) = correctionsLSQ.segment(i*4,3);
    //differentialCorrection(((11*i)+10)) = correctionsLSQ( (i*4) +3 );


}

//std::cout << "differentialCorrection: \n" << differentialCorrection << std::endl;

    return differentialCorrection;
}
