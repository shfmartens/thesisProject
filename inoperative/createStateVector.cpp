#include "createStateVector.h"

//! Create a state vector
Eigen::VectorXd createStateVector(double xCoordinateInNormalizedUnits, double zCoordinateInNormalizedUnits, double massParameter, double jacobiEnergy) {

    // Calculate the distances to the two bodies. For the 3D-case Y0 = 0.
    double distanceToPrimaryBody = sqrt( pow( xCoordinateInNormalizedUnits + massParameter, 2.0 ) + pow(zCoordinateInNormalizedUnits, 2.0) );
    double distanceToSecondaryBody = sqrt( pow( xCoordinateInNormalizedUnits - 1.0 + massParameter, 2.0 ) + pow(zCoordinateInNormalizedUnits, 2.0) );

    // Check the sign of the y-velocity.
    double sign = 1.0;
    if ((zCoordinateInNormalizedUnits < 0.0 && xCoordinateInNormalizedUnits < 1.15568214778253*1.1) || (zCoordinateInNormalizedUnits > 0.0 && xCoordinateInNormalizedUnits > 1.15568214778253*1.1)) {
        sign = -1.0;
    }

    // Compose the state vector, including the STM. The STM-part is a 6x6 identity-matrix reshaped to a vector.
    Eigen::VectorXd stateVector = Eigen::VectorXd::Zero(42);
    stateVector(0) = xCoordinateInNormalizedUnits;
    stateVector(2) = zCoordinateInNormalizedUnits;
    stateVector(4) = sign * sqrt( pow( xCoordinateInNormalizedUnits, 2.0) + ( 2.0 * (1.0 - massParameter ) ) / distanceToPrimaryBody + ( 2.0 * massParameter ) / distanceToSecondaryBody - jacobiEnergy );
    stateVector(6) = 1.0;
    stateVector(13) = 1.0;
    stateVector(20) = 1.0;
    stateVector(27) = 1.0;
    stateVector(34) = 1.0;
    stateVector(41) = 1.0;

    // Return the state vector.
    return stateVector;
}
