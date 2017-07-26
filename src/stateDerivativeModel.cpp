// Include statements.
#include "Tudat/Basics/utilityMacros.h"
#include "Tudat/Astrodynamics/Gravitation/librationPoint.h"

#include "stateDerivativeModel.h"

// Define function.
Eigen::VectorXd computeStateDerivative( const double time, const Eigen::VectorXd cartesianState) {

    // Time is not directly used in the function.
    TUDAT_UNUSED_PARAMETER( time );

    // Declare mass parameter.
    extern double massParameter;
//    extern Eigen::Vector3d thrustVector;
//    extern double thrustAcceleration;

    // Declare state derivative vector with same length as the state.
    Eigen::VectorXd stateDerivative = Eigen::VectorXd::Zero(cartesianState.size());

    // Set the derivative of the position equal to the velocities.
    stateDerivative(0) = cartesianState(3);
    stateDerivative(1) = cartesianState(4);
    stateDerivative(2) = cartesianState(5);

    // Compute distances to primaries.
    double distanceToPrimaryBody   = sqrt(pow((cartesianState(0)+massParameter),2.0)     + pow(cartesianState(1),2.0) + pow(cartesianState(2),2.0));
    double distanceToSecondaryBody = sqrt(pow((1.0-massParameter-cartesianState(0)),2.0) + pow(cartesianState(1),2.0) + pow(cartesianState(2),2.0));
// Error by Rohner:
//    double distanceToSecondaryBody = sqrt( pow(cartesianState(0) - 1.0 + massParameter, 2.0) + pow(cartesianState(1), 2.0) + pow(cartesianState(2), 2.0) );

    // Set the derivative of the velocities to the accelerations.
    double termRelatedToPrimaryBody   = (1.0-massParameter)/pow(distanceToPrimaryBody,3.0);
    double termRelatedToSecondaryBody = massParameter      /pow(distanceToSecondaryBody,3.0);
//    stateDerivative(3) = 2.0 * cartesianState(4) + cartesianState(0) - termRelatedToPrimaryBody * ( cartesianState(0) + massParameter ) - termRelatedToSecondaryBody * ( cartesianState(0) - 1.0 + massParameter ) + thrustVector(0) * thrustAcceleration;
//    stateDerivative(4) = -2.0 * cartesianState(3) + cartesianState(1) - termRelatedToPrimaryBody * cartesianState(1) - termRelatedToSecondaryBody * cartesianState(1)+thrustVector(1)*thrustAcceleration;
//    stateDerivative(5) = -termRelatedToPrimaryBody * cartesianState(2) - termRelatedToSecondaryBody * cartesianState(2)+thrustVector(2)*thrustAcceleration;
    stateDerivative(3) = -termRelatedToPrimaryBody*(massParameter+cartesianState(0)) + termRelatedToSecondaryBody*(1.0-massParameter-cartesianState(0)) + cartesianState(0) + 2.0*cartesianState(4);
    stateDerivative(4) = -termRelatedToPrimaryBody*cartesianState(1)                 - termRelatedToSecondaryBody*cartesianState(1)                     + cartesianState(1) - 2.0*cartesianState(3);
    stateDerivative(5) = -termRelatedToPrimaryBody*cartesianState(2)                 - termRelatedToSecondaryBody*cartesianState(2);

    // Compute partial derivatives of the potential.
//    double Uxx = 1.0 - (1.0-massParameter)/pow(distanceToPrimaryBody,3.0) - massParameter/pow(distanceToSecondaryBody, 3.0) + 3.0*(1.0-massParameter)*pow(cartesianState(0) + massParameter, 2.0)/pow(distanceToPrimaryBody, 5.0) + 3.0*massParameter*pow(cartesianState(0)-1+massParameter,2.0)/pow(distanceToSecondaryBody, 5.0);
//    double Uxy = 3.0*(1.0-massParameter)*(cartesianState(0)+massParameter)*cartesianState(1)/pow(distanceToPrimaryBody, 5.0) + 3.0*massParameter*(cartesianState(0)-1.0+massParameter)*cartesianState(1)/pow(distanceToSecondaryBody, 5.0);
//    double Uxz = 3.0*(1.0-massParameter)*(cartesianState(0)+massParameter)*cartesianState(2)/pow(distanceToPrimaryBody, 5.0) + 3.0*massParameter*(cartesianState(0)-1.0+massParameter)*cartesianState(2)/pow(distanceToSecondaryBody, 5.0);
//    double Uyx = Uxy;
//    double Uyy = 1.0 - (1.0-massParameter)/pow(distanceToPrimaryBody, 3.0) - massParameter/pow(distanceToSecondaryBody, 3.0) + 3.0*(1.0-massParameter)* pow(cartesianState(1), 2.0)/pow(distanceToPrimaryBody, 5.0) + 3.0*massParameter* pow(cartesianState(1),2.0)/pow(distanceToSecondaryBody, 5.0);
//    double Uyz = 3.0*(1.0-massParameter)* cartesianState(1)    *cartesianState(2)/pow(distanceToPrimaryBody, 5.0) + 3.0*massParameter* cartesianState(1)      *cartesianState(2)/pow(distanceToSecondaryBody, 5.0);
//    double Uzx = Uxz;
//    double Uzy = Uyz;
//    double Uzz =     (1.0-massParameter)/pow(distanceToPrimaryBody, 3.0) - massParameter/pow(distanceToSecondaryBody, 3.0) + 3.0*(1.0-massParameter)* pow(cartesianState(2), 2.0)/pow(distanceToPrimaryBody, 5.0) + 3.0*massParameter* pow(cartesianState(2), 2.0)/pow(distanceToSecondaryBody, 5.0);

    double Uxx = (3.0*(1.0-massParameter)*pow(cartesianState(0)+massParameter, 2.0)          )/pow(distanceToPrimaryBody,5.0) + (3.0*massParameter*pow(1.0-massParameter-cartesianState(0),2.0)           )/pow(distanceToSecondaryBody,5.0) - (1.0-massParameter)/pow(distanceToPrimaryBody,3.0) - massParameter/pow(distanceToSecondaryBody,3.0) + 1.0;
    double Uxy = (3.0*(1.0-massParameter)*(cartesianState(0)+massParameter)*cartesianState(1))/pow(distanceToPrimaryBody,5.0) - (3.0*massParameter*(1.0-massParameter-cartesianState(0))*cartesianState(1))/pow(distanceToSecondaryBody,5.0);
    double Uxz = (3.0*(1.0-massParameter)*(cartesianState(0)+massParameter)*cartesianState(2))/pow(distanceToPrimaryBody,5.0) - (3.0*massParameter*(1.0-massParameter-cartesianState(0))*cartesianState(2))/pow(distanceToSecondaryBody,5.0);
    double Uyx = Uxy;
    double Uyy = (3.0*(1.0-massParameter)*pow(cartesianState(1),2.0)                         )/pow(distanceToPrimaryBody,5.0) + (3.0*massParameter*pow(cartesianState(1),2.0)                             )/pow(distanceToSecondaryBody,5.0) - (1.0-massParameter)/pow(distanceToPrimaryBody,3.0) - massParameter/pow(distanceToSecondaryBody,3.0) + 1.0 ;
    double Uyz = (3.0*(1.0-massParameter)*cartesianState(1)*cartesianState(2)                )/pow(distanceToPrimaryBody,5.0) + (3.0*massParameter*cartesianState(1)*cartesianState(2)                    )/pow(distanceToSecondaryBody,5.0);
    double Uzx = Uxz;
    double Uzy = Uyz;
    double Uzz = (3.0*(1.0-massParameter)*pow(cartesianState(2),2.0)                         )/pow(distanceToPrimaryBody,5.0) + (3.0*massParameter*pow(cartesianState(2),2.0)                             )/pow(distanceToSecondaryBody,5.0) - (1.0-massParameter)/pow(distanceToPrimaryBody,3.0) - massParameter/pow(distanceToSecondaryBody,3.0) ;


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

    // Return computed state derivative.
    return stateDerivative;
}
