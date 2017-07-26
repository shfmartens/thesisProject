#include <Eigen/Core>
#include <Eigen/LU>
#include <iostream>

#include "computeDifferentialCorrection.h"
#include "stateDerivativeModel.h"

using namespace std;

// Function to compute the differential correction
Eigen::VectorXd computeDifferentialCorrection(Eigen::VectorXd cartesianState)
{
    // Initiate vectors, matrices etc.
    Eigen::VectorXd differentialCorrection(7);
    Eigen::VectorXd corrections(3);
    Eigen::MatrixXd updateMatrix(3,3);
    Eigen::MatrixXd multiplicationMatrix(3,1);

    // Compute the accelerations and velocities (in X- and Z-direction) on the spacecraft and put them in a 2x1 vector.
    Eigen::VectorXd cartesianAccelerations = computeStateDerivative(0.0, cartesianState);

    // Reshape the state vector to matrix form.
    Eigen::VectorXd stmPartOfStateVector = cartesianState.segment(6,36);
    Eigen::Map<Eigen::MatrixXd> stmPartOfStateVectorInMatrixForm = Eigen::Map<Eigen::MatrixXd>(stmPartOfStateVector.data(),6,6);

    // Check which deviation is larger: x-velocity or z-velocity.
    if (fabs(cartesianState(3)) > fabs(cartesianState(5))) {
        // Set the correct multiplication matrix.
        multiplicationMatrix << cartesianState(1), cartesianState(3), cartesianState(5);

        // Compute the update matrix.
        updateMatrix << stmPartOfStateVectorInMatrixForm(1, 2), stmPartOfStateVectorInMatrixForm(1, 4), cartesianState(4),
                stmPartOfStateVectorInMatrixForm(3, 2), stmPartOfStateVectorInMatrixForm(3, 4), cartesianAccelerations(3),
                stmPartOfStateVectorInMatrixForm(5, 2), stmPartOfStateVectorInMatrixForm(5, 4), cartesianAccelerations(5);

        // Compute the necessary differential correction.
        corrections = updateMatrix.inverse() * multiplicationMatrix;

        // Put corrections in correct format.
        differentialCorrection.setZero();
        differentialCorrection(2) = -corrections(0);
        differentialCorrection(4) = -corrections(1);
        differentialCorrection(6) = -corrections(2);
    } else {
        // Set the correct multiplication matrix.
        multiplicationMatrix << cartesianState(1), cartesianState(3), cartesianState(5);

        // Compute the update matrix.
        updateMatrix << stmPartOfStateVectorInMatrixForm(1, 0), stmPartOfStateVectorInMatrixForm(1, 4), cartesianState(4),
                        stmPartOfStateVectorInMatrixForm(3, 0), stmPartOfStateVectorInMatrixForm(3, 4), cartesianAccelerations(3),
                        stmPartOfStateVectorInMatrixForm(5, 0), stmPartOfStateVectorInMatrixForm(5, 4), cartesianAccelerations(5);

        // Compute the necessary differential correction.
        corrections = updateMatrix.inverse() * multiplicationMatrix;

        // Put corrections in correct format.
        differentialCorrection.setZero();
        differentialCorrection(0) = -corrections(0);
        differentialCorrection(4) = -corrections(1);
        differentialCorrection(6) = -corrections(2);
    }


    // Return differential correction.
    return differentialCorrection;

}
