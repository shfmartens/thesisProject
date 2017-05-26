#include <Eigen/Core>
#include <Eigen/LU>
#include <iostream>

#include "thesisProject/computeDifferentialCorrection.h"
#include "stateDerivativeModel2D.h"

using namespace std;

// Function to compute the differential correction
double computeDifferentialCorrection(Eigen::VectorXd cartesianState)
{

    // Initiate vectors, matrices etc.
    double differentialCorrection;

    // Compute the accelerations and velocities on the spacecraft.
    Eigen::VectorXd cartesianAccelerations = computeStateDerivative(0.0, cartesianState);
    double xdotdot = cartesianAccelerations(2);

    // Reshape the state vector to matrix form.
    Eigen::VectorXd stmPartOfStateVector = cartesianState.segment(4,16);
    Eigen::Map<Eigen::MatrixXd> stmPartOfStateVectorInMatrixForm = Eigen::Map<Eigen::MatrixXd>(stmPartOfStateVector.data(),4,4);

    // Compute the differential correction.
    differentialCorrection = cartesianState(2) / (stmPartOfStateVectorInMatrixForm(2,3) - xdotdot / cartesianState(3) * stmPartOfStateVectorInMatrixForm(1,3));

    // Return differential correction.
    return differentialCorrection;

}
