#ifndef TUDATBUNDLE_INITIALISECONTINUATIONFROMTEXTFILE_H
#define TUDATBUNDLE_INITIALISECONTINUATIONFROMTEXTFILE_H


#include "Eigen/Core"
#include <map>

Eigen::VectorXd extractStatesContinuationVectorFromKnownOrbitNumber(std::string continuation_fileName, const int targetOrbitNumber);

Eigen::VectorXd extractStatesContinuationVectorFromKnownHamiltonian(std::string referenceString, const double familyHamiltonian, int& numberOfCollocationPoints, int& orbitNumber);

void initialiseContinuationFromTextFile(const int librationPointNr, const std::string orbitType, const double accelerationMagnitude, const double accelerationAngle, const double accelerationAngle2,
                                        const double hamiltonianFirstGuess, const double hamiltonianSecondGuess, const double ySign, const double massParameter,
                                        Eigen::VectorXd& statesContinuationVectorFirstGuess, Eigen::VectorXd& statesContinuationVectorSecondGuess,
                                        int& numberOfCollocationPointsFirstGuess, int& numberOfCollocationPointsSecondGuess,
                                        Eigen::VectorXd& adaptedIncrementVector, int& numberOfInitialConditions);


#endif  // TUDATBUNDLE_INITIALISECONTINUATIONFROMTEXTFILE_H
