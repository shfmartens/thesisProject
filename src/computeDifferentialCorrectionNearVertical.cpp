#include <Eigen/Core>
#include <Eigen/LU>
#include <iostream>

#include "computeDifferentialCorrectionNearVertical.h"
#include "stateDerivativeModel.h"

using namespace std;

// Function to compute the differential correction
Eigen::VectorXd computeDifferentialCorrectionNearVertical(Eigen::VectorXd cartesianState)
{
    // Initiate vectors, matrices etc.
    Eigen::VectorXd differentialCorrection(6);
    Eigen::VectorXd corrections(2);
    Eigen::MatrixXd updateMatrix(2,2);
    Eigen::MatrixXd vectorOne(2,1);
    Eigen::MatrixXd vectorTwo(2,1);
    Eigen::MatrixXd multiplicationMatrix(1,2);

    // Compute the two vectors for differential correction [y_dot, x_dot_dot] and [y, x_dot].
    Eigen::VectorXd cartesianAccelerations = computeStateDerivative(0.0, cartesianState);
    vectorOne << -cartesianState(4),
            cartesianAccelerations(3);
    vectorTwo << -cartesianState(1),
            -cartesianState(3);

    // Reshape the state vector to matrix form.
    Eigen::VectorXd stmPartOfStateVector = cartesianState.segment(6,36);
    Eigen::Map<Eigen::MatrixXd> stmPartOfStateVectorInMatrixForm = Eigen::Map<Eigen::MatrixXd>(stmPartOfStateVector.data(),6,6);


    // Set the correct multiplication matrix.
    multiplicationMatrix << stmPartOfStateVectorInMatrixForm(2,4), stmPartOfStateVectorInMatrixForm(2,5);

    // Compute the update matrix.
    updateMatrix << stmPartOfStateVectorInMatrixForm(1,4), stmPartOfStateVectorInMatrixForm(1,5),
            stmPartOfStateVectorInMatrixForm(3,4), stmPartOfStateVectorInMatrixForm(3,5);
    updateMatrix = updateMatrix - (1.0 / cartesianState(5)) * vectorOne * multiplicationMatrix;

    // Compute the necessary differential correction.
    corrections = updateMatrix.inverse() * vectorTwo;

    // Put corrections in correct format.
    differentialCorrection.setZero();
    differentialCorrection(4) = corrections(0);
    differentialCorrection(5) = corrections(1);

    // Return differential correction.
    return differentialCorrection;

}
