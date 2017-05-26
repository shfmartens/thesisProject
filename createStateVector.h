#ifndef CREATESTATEVECTOR_H
#define CREATESTATEVECTOR_H

#include <Eigen/Core>
#include "createStateVector.cpp"

//! Create a state vector.
/*!
 * Returns the state vector.
 * \param xCoordinateInNormalizedUnits X-coordinate of the third body in normalized units.
 * \param zCoordinateInNormalizedUnits Z-coordinate of the third body in normalized units.
 * \param yVelocityInNormalizedUnits Y-velocity of the third body in normalized units.
 * \return State vector including the coordinates and velocities in three dimensions and the STM.
 */
Eigen::VectorXd createStateVector(double xCoordinateInNormalizedUnits, double zCoordinateInNormalizedUnits, double massParameter, double jacobiEnergy);

Eigen::VectorXd computeStateVector(double xCoordinateInNormalizedUnits, double zCoordinateInNormalizedUnits, double yVelocityInNormalizedUnits, double massParameter);

#endif // CREATESTATEVECTOR_H
