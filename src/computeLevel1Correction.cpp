#include <Eigen/Core>
#include <Eigen/LU>
#include <iostream>

#include "computeLevel1Correction.h"
#include "stateDerivativeModelAugmented.h"



Eigen::VectorXd computeLevel1Correction( const Eigen::VectorXd deviationVector, const Eigen::MatrixXd propagatedStatesInclSTM, const int numberOfPatchPoints )
{

// Initialize variables and set vectors and matrices to zero
int numberOfArcs = numberOfPatchPoints - 1;
Eigen::VectorXd differentialCorrection(numberOfPatchPoints*11);
Eigen::MatrixXd updateMatrix(numberOfArcs*3,numberOfArcs*3);
Eigen::VectorXd constraintVector(numberOfArcs*3);
Eigen::VectorXd corrections(numberOfArcs*3);

differentialCorrection.setZero();
updateMatrix.setZero();
constraintVector.setZero();

// store results into the outputVector

// Construct the constraint and update matrix
for(int i = 0; i <= numberOfPatchPoints-2; i++ ) {

 constraintVector.segment(i*3,3) = deviationVector.segment(i*11,3);
 updateMatrix.block(i*3,i*3,3,3) = propagatedStatesInclSTM.block(i*10,4,3,3); // OR INVERSE MATRICES?

}

// compute the correction
corrections = updateMatrix.inverse() * constraintVector;

// store results into the outputVector
for(int i = 0; i <= numberOfPatchPoints-2; i++ ) {

    differentialCorrection.segment(((11*i)+3),3) = corrections.segment(i*3,3);

}

    return differentialCorrection;
}
