#ifndef CREATESTATEVECTOR_H
#define CREATESTATEVECTOR_H

// Include statements.
#include <Eigen/Core>
#include "createStateVector2D.cpp"

//! Create a state vector.
/*!
 * Returns the state vector.
 * \param xCoordinateInNormalizedUnits X-coordinate of the third body in normalized units.
 * \param zCoordinateInNormalizedUnits Z-coordinate of the third body in normalized units.
 * \param yVelocityInNormalizedUnits Y-velocity of the third body in normalized units.
 * \return State vector including the coordinates and velocities in three dimensions and the STM.
 */
Eigen::VectorXd createStateVector(double xCoordinateInNormalizedUnits, double massParameter, double jacobiEnergy);

#endif // CREATESTATEVECTOR_H
