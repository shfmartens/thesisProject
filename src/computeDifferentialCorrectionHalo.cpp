#include <Eigen/Core>
#include <Eigen/LU>
#include <iostream>

#include "computeDifferentialCorrectionHalo.h"
#include "stateDerivativeModel.h"

using namespace std;

// Function to compute the differential correction
Eigen::VectorXd computeDifferentialCorrectionHalo(Eigen::VectorXd cartesianState)
{
    // Initiate vectors, matrices etc.
    Eigen::VectorXd differentialCorrection(6);
    Eigen::VectorXd corrections(2);
    Eigen::MatrixXd updateMatrix(2,2);
    Eigen::MatrixXd accelerations(2,1);
    Eigen::MatrixXd velocities(2,1);
    Eigen::MatrixXd multiplicationMatrix(1,2);

    // Compute the accelerations and velocities (in X- and Z-direction) on the spacecraft and put them in a 2x1 vector.
    Eigen::VectorXd cartesianAccelerations = computeStateDerivative(0.0, cartesianState);
    accelerations << cartesianAccelerations(3),
            cartesianAccelerations(5);
    velocities << -cartesianState(3),
            -cartesianState(5);

    // Reshape the state vector to matrix form.
    Eigen::VectorXd stmPartOfStateVector = cartesianState.segment(6,36);
    Eigen::Map<Eigen::MatrixXd> stmPartOfStateVectorInMatrixForm = Eigen::Map<Eigen::MatrixXd>(stmPartOfStateVector.data(),6,6);

    // Check which deviation is larger: x-velocity or z-velocity.
//    if (fabs(cartesianState(5)) > fabs(cartesianState(3))) {

        // Set the correct multiplication matrix.
        multiplicationMatrix << stmPartOfStateVectorInMatrixForm(1,2), stmPartOfStateVectorInMatrixForm(1,4);

        // Compute the update matrix.
        updateMatrix << stmPartOfStateVectorInMatrixForm(3,2), stmPartOfStateVectorInMatrixForm(3,4),
                        stmPartOfStateVectorInMatrixForm(5,2), stmPartOfStateVectorInMatrixForm(5,4);
        updateMatrix = updateMatrix - (1.0 / cartesianState(4)) * accelerations * multiplicationMatrix;

        // Compute the necessary differential correction.
        corrections = updateMatrix.inverse() * velocities;

        // Put corrections in correct format.
        differentialCorrection.setZero();
        differentialCorrection(2) = corrections(0);
        differentialCorrection(4) = corrections(1);

//    }
//    else {
//
//        // Set the correct multiplication matrix.
//        multiplicationMatrix << stmPartOfStateVectorInMatrixForm(1,0), stmPartOfStateVectorInMatrixForm(1,4);
//
//        // Compute the update matrix.
//        updateMatrix << stmPartOfStateVectorInMatrixForm(3,0), stmPartOfStateVectorInMatrixForm(3,4),
//                        stmPartOfStateVectorInMatrixForm(5,0), stmPartOfStateVectorInMatrixForm(5,4);
//        updateMatrix = updateMatrix - (1.0 / cartesianState(4)) * accelerations * multiplicationMatrix;
//
//        // Compute the necessary differential correction.
//        corrections = updateMatrix.inverse() * velocities;
//
//        // Put corrections in correct format.
//        differentialCorrection.setZero();
//        differentialCorrection(0) = corrections(0);
//        differentialCorrection(4) = corrections(1);
//    }


    // Return differential correction.
    return differentialCorrection;

}
