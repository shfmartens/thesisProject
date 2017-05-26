// Include statements.
#include <TudatCore/Basics/utilityMacros.h>
#include "stateDerivativeModel2D.h"

// Define function.
Eigen::VectorXd computeStateDerivative( const double time, const Eigen::VectorXd cartesianState) {

    // Time is not directly used in the function.
    TUDAT_UNUSED_PARAMETER( time );

    // Calculate mass parameter for system.
    const double massParameter = 0.012155092;//crtbp::computeMassParameter(GMEarth, GMMoon);

    // Declare state derivative vector with same length as the state.
    Eigen::VectorXd stateDerivative = Eigen::VectorXd::Zero(cartesianState.size());

    // Set the derivative of the position equal to the velocitities.
    stateDerivative(0) = cartesianState(2);
    stateDerivative(1) = cartesianState(3);

    // Compute distances to primaries.
    double distanceToPrimaryBody = sqrt( pow(cartesianState(0) + massParameter, 2.0) + pow(cartesianState(1), 2.0));
    double distanceToSecondaryBody = sqrt( pow(cartesianState(0) -1.0 + massParameter, 2.0) + pow(cartesianState(1), 2.0));

    // Set the derivative of the velocities to the accelerations.
    double termRelatedToPrimaryBody = ( 1.0 - massParameter ) / pow( distanceToPrimaryBody, 3.0 );
    double termRelatedToSecondaryBody = massParameter / pow( distanceToSecondaryBody, 3.0 );
    stateDerivative(2) = 2.0 * cartesianState(3) + cartesianState(0) - termRelatedToPrimaryBody * ( cartesianState(0) + massParameter ) - termRelatedToSecondaryBody * ( cartesianState(0) - 1.0 + massParameter );
    stateDerivative(3) = -2.0 * cartesianState(2) + cartesianState(1) - termRelatedToPrimaryBody * cartesianState(1) - termRelatedToSecondaryBody * cartesianState(1);

    // Compute partial derivatives of the potential.
    double Uxx = 1.0 - (1.0-massParameter)/pow(distanceToPrimaryBody,3.0) - massParameter/pow(distanceToSecondaryBody, 3.0) + 3.0*(1.0-massParameter)*pow(cartesianState(0) + massParameter, 2.0)/pow(distanceToPrimaryBody, 5.0) + 3.0 * massParameter * pow(cartesianState(0) - 1 + massParameter,2.0) / pow(distanceToSecondaryBody, 5.0);
    double Uxy = 3.0*(1.0 - massParameter) * (cartesianState(0) + massParameter) * cartesianState(1) / pow(distanceToPrimaryBody, 5.0) + 3.0 * massParameter * (cartesianState(0) - 1.0 + massParameter) * cartesianState(1) / pow(distanceToSecondaryBody, 5.0);
    double Uyx = Uxy;
    double Uyy = 1.0 - (1.0-massParameter)/pow(distanceToPrimaryBody, 3.0) - massParameter / pow(distanceToSecondaryBody, 3.0) + 3.0 * (1.0 - massParameter) * pow(cartesianState(1), 2.0) / pow(distanceToPrimaryBody, 5.0) + 3.0 * massParameter * pow(cartesianState(1),2.0) / pow(distanceToSecondaryBody, 5.0);

    // Create the STM-derivative matrix
    Eigen::MatrixXd stmDerivativeFunction (4,4);
    stmDerivativeFunction << 0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
            Uxx, Uxy, 0.0, 2.0,
            Uyx, Uyy, -2.0, 0.0;

    // Reshape the STM part of the state vector to a 4x4 matrix.
    Eigen::VectorXd stmPartOfStateVector = cartesianState.segment(4,16);
    Eigen::Map<Eigen::MatrixXd> stmPartOfStateVectorInMatrixForm = Eigen::Map<Eigen::MatrixXd>(stmPartOfStateVector.data(),4,4);

    // Differentiate the STM.
    Eigen::MatrixXd derivativeOfSTMInMatrixForm = stmDerivativeFunction * stmPartOfStateVectorInMatrixForm;

    // Reshape the derivative of the STM to a vector.
    stateDerivative.segment(4,16) = Eigen::Map<Eigen::MatrixXd>(derivativeOfSTMInMatrixForm.data(),16,1);

    // Return computed state derivative.
    return stateDerivative;
}
