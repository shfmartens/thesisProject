// Include-statements.
#include "createStateVector2D.h"
#include "Tudat/Astrodynamics/Gravitation/jacobiEnergy.h"

//! Create a state vector.
Eigen::VectorXd createStateVector(double xCoordinateInNormalizedUnits, double massParameter, double jacobiEnergy) {

    // Calculate the distances to the two bodies. For the 2D-case Y0=Z0=0.
    double distanceToPrimaryBody = sqrt( pow( xCoordinateInNormalizedUnits + massParameter, 2.0 ) );
    double distanceToSecondaryBody = sqrt( pow( xCoordinateInNormalizedUnits - 1.0 + massParameter, 2.0 ) );

    // Compose the state vector, including the STM. The STM-part is a 4x4 identity-matrix reshaped to a vector.
    Eigen::VectorXd stateVector = Eigen::VectorXd::Zero(20);
    stateVector(0) = xCoordinateInNormalizedUnits;
    stateVector(3) = sqrt( pow( xCoordinateInNormalizedUnits, 2.0) + ( 2.0 * (1.0 - massParameter ) ) / distanceToPrimaryBody + ( 2.0 * massParameter ) / distanceToSecondaryBody - jacobiEnergy );
    stateVector(4) = 1.0;
    stateVector(9) = 1.0;
    stateVector(14) = 1.0;
    stateVector(19) = 1.0;

    // Return the state vector.
    return stateVector;
}
