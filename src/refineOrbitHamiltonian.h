#ifndef TUDATBUNDLE_REFINEORBITHAMILTONIAN_H
#define TUDATBUNDLE_REFINEORBITHAMILTONIAN_H


#include "Eigen/Core"
#include <map>

Eigen::VectorXd extractStatesContinuationVector(std::string referenceString, const double familyHamiltonian, const double accelerationMagnitude, const bool startFromAlpha, int& numberOfCollocationPoints);

std::string createReferenceString(const int librationPointNr, const std::string orbitType, const double accelerationMagnitude, const double accelerationAngle, const double accelerationAngle2, const double familyHamiltonian, bool startFromAlpha);

Eigen::MatrixXd refineOrbitHamiltonian (const int librationPointNr, const std::string orbitType, const double accelerationMagnitude, const double accelerationAngle, const double accelerationAngle2,
                                        const double familyHamiltonian, const double massParameter, const int continuationIndex, bool startFromAlpha, int& numberOfCollocationPoints);



#endif  // TUDATBUNDLE_APPLYCOLLOCATION_H
