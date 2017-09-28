#include "Tudat/Basics/utilityMacros.h"
#include "Tudat/Astrodynamics/Gravitation/librationPoint.h"

#include "stateDerivativeModel.h"



Eigen::VectorXd computeStateDerivative( const double time, const Eigen::VectorXd cartesianState )
{
    // Time is not directly used in the function.
    TUDAT_UNUSED_PARAMETER( time );

    // Declare mass parameter.
    extern double massParameter;

    // Declare state derivative vector with same length as the state.
    Eigen::VectorXd stateDerivative = Eigen::VectorXd::Zero(cartesianState.size());

    // Set the derivative of the position equal to the velocities.
    stateDerivative(0) = cartesianState(3);
    stateDerivative(1) = cartesianState(4);
    stateDerivative(2) = cartesianState(5);

    double xPositionScaledSquared = (cartesianState(0)+massParameter) * (cartesianState(0)+massParameter);
    double xPositionScaledSquared2 = (1.0-massParameter-cartesianState(0)) * (1.0-massParameter-cartesianState(0));
    double yPositionScaledSquared = (cartesianState(1) * cartesianState(1) );
    double zPositionScaledSquared = (cartesianState(2) * cartesianState(2) );

    // Compute distances to primaries.
    double distanceToPrimaryBody   = sqrt(xPositionScaledSquared     + yPositionScaledSquared + zPositionScaledSquared);
    double distanceToSecondaryBody = sqrt(xPositionScaledSquared2 + yPositionScaledSquared + zPositionScaledSquared);

    double distanceToPrimaryCubed = distanceToPrimaryBody * distanceToPrimaryBody * distanceToPrimaryBody;
    double distanceToSecondaryCubed = distanceToSecondaryBody * distanceToSecondaryBody * distanceToSecondaryBody;

    double distanceToPrimaryToFifthPower = distanceToPrimaryCubed * distanceToPrimaryBody * distanceToPrimaryBody;
    double distanceToSecondaryToFifthPower = distanceToSecondaryCubed * distanceToSecondaryBody * distanceToSecondaryBody;

    // Set the derivative of the velocities to the accelerations.
    double termRelatedToPrimaryBody   = (1.0-massParameter)/distanceToPrimaryCubed;
    double termRelatedToSecondaryBody = massParameter      /distanceToSecondaryCubed;
    stateDerivative(3) = -termRelatedToPrimaryBody*(massParameter+cartesianState(0)) + termRelatedToSecondaryBody*(1.0-massParameter-cartesianState(0)) + cartesianState(0) + 2.0*cartesianState(4);
    stateDerivative(4) = -termRelatedToPrimaryBody*cartesianState(1)                 - termRelatedToSecondaryBody*cartesianState(1)                     + cartesianState(1) - 2.0*cartesianState(3);
    stateDerivative(5) = -termRelatedToPrimaryBody*cartesianState(2)                 - termRelatedToSecondaryBody*cartesianState(2);

    // Compute partial derivatives of the potential.
    double Uxx = (3.0*(1.0-massParameter)*xPositionScaledSquared          )/distanceToPrimaryToFifthPower+ (3.0*massParameter*xPositionScaledSquared2           )/distanceToSecondaryToFifthPower - (1.0-massParameter)/distanceToPrimaryCubed - massParameter/distanceToSecondaryCubed + 1.0;
    double Uxy = (3.0*(1.0-massParameter)*(cartesianState(0)+massParameter)*cartesianState(1))/distanceToPrimaryToFifthPower- (3.0*massParameter*(1.0-massParameter-cartesianState(0))*cartesianState(1))/distanceToSecondaryToFifthPower;
    double Uxz = (3.0*(1.0-massParameter)*(cartesianState(0)+massParameter)*cartesianState(2))/distanceToPrimaryToFifthPower- (3.0*massParameter*(1.0-massParameter-cartesianState(0))*cartesianState(2))/distanceToSecondaryToFifthPower;
    double Uyx = Uxy;
    double Uyy = (3.0*(1.0-massParameter)*yPositionScaledSquared                         )/distanceToPrimaryToFifthPower+ (3.0*massParameter*yPositionScaledSquared                             )/distanceToSecondaryToFifthPower - (1.0-massParameter)/distanceToPrimaryCubed - massParameter/distanceToSecondaryCubed + 1.0 ;
    double Uyz = (3.0*(1.0-massParameter)*cartesianState(1)*cartesianState(2)                )/distanceToPrimaryToFifthPower+ (3.0*massParameter*cartesianState(1)*cartesianState(2)                    )/distanceToSecondaryToFifthPower;
    double Uzx = Uxz;
    double Uzy = Uyz;
    double Uzz = (3.0*(1.0-massParameter)*zPositionScaledSquared                         )/distanceToPrimaryToFifthPower+ (3.0*massParameter*zPositionScaledSquared                             )/distanceToSecondaryToFifthPower - (1.0-massParameter)/distanceToPrimaryCubed - massParameter/distanceToSecondaryCubed ;


    // Create the STM-derivative matrix
    Eigen::MatrixXd stmDerivativeFunction (6,6);
    stmDerivativeFunction << 0.0, 0.0, 0.0,  1.0, 0.0, 0.0,
                             0.0, 0.0, 0.0,  0.0, 1.0, 0.0,
                             0.0, 0.0, 0.0,  0.0, 0.0, 1.0,
                             Uxx, Uxy, Uxz,  0.0, 2.0, 0.0,
                             Uyx, Uyy, Uyz, -2.0, 0.0, 0.0,
                             Uzx, Uzy, Uzz,  0.0, 0.0, 0.0;

    // Reshape the STM part of the state vector to a 6x6 matrix.
    Eigen::VectorXd stmPartOfStateVector = cartesianState.segment(6,36);
    Eigen::Map<Eigen::MatrixXd> stmPartOfStateVectorInMatrixForm = Eigen::Map<Eigen::MatrixXd>(stmPartOfStateVector.data(),6,6);

    // Differentiate the STM.
    Eigen::MatrixXd derivativeOfSTMInMatrixForm = stmDerivativeFunction * stmPartOfStateVectorInMatrixForm;

    // Reshape the derivative of the STM to a vector.
    stateDerivative.segment(6,36) = Eigen::Map<Eigen::MatrixXd>(derivativeOfSTMInMatrixForm.data(),36,1);

    return stateDerivative;
}
