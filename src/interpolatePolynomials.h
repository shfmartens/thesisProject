#ifndef TUDATBUNDLE_INTERPOLATEPOLYNOMIALS_H
#define TUDATBUNDLE_INTERPOLATEPOLYNOMIALS_H

#include "Tudat/Basics/basicTypedefs.h"

#include <Eigen/Core>
#include <string>
#include <vector>

void rewriteDesignVectorToFullFormatComplex(const Eigen::VectorXcd collocationDesignVector, const int numberOfCollocationPoints, const Eigen::VectorXcd thrustAndMassParameters, Eigen::VectorXcd& currentDesignVector );

void rewriteDesignVectorToFullFormat(const Eigen::MatrixXd collocationDesignVector, const int numberOfCollocationPoints, const Eigen::VectorXd thrustAndMassParameters, Eigen::VectorXd& currentDesignVector );

double computeIntegralPhaseConstraint(const Eigen::MatrixXd collocationDesignVector, const int numberOfCollocationPoints,const Eigen::VectorXd previousDesignVector );

void computeTimeAndSegmentInformationFromPhaseComplex(const Eigen::VectorXcd currentDesignVector, const Eigen::VectorXcd previousDesignVector, const int currentNumberOfCollocationPoints, const Eigen::MatrixXcd oddStates, const int previousNumberOfCollocationPoints,
                                                      Eigen::VectorXcd& oddPointTimesDimensional, Eigen::VectorXcd& oddPointTimesNormalized, Eigen::VectorXd& segmentVector);

void computeTimeAndSegmentInformationFromPhase(const Eigen::VectorXd currentDesignVector, const Eigen::VectorXd previousDesignVector, const int currentNumberOfCollocationPoints, const Eigen::MatrixXd oddStates, const int previousNumberOfCollocationPoints,
                                               Eigen::VectorXd& oddPointTimesDimensional, Eigen::VectorXd& oddPointTimesNormalized, Eigen::VectorXd& segmentVector);

void computeStateIncrementFromInterpolation (const Eigen::VectorXd previousGuess, Eigen::VectorXd currentGuess, Eigen::VectorXd& stateIncrement);

void computeTimeAndSegmentInformation(const Eigen::MatrixXd collocationDesignVector, const int oldNumberOfCollocationPoints, const int newNumberOfCollocationPoints, Eigen::VectorXd& oddPointTimesNormalized, Eigen::VectorXd& oddPointTimesDimensional, Eigen::VectorXd& segmentVector);

void interpolatePolynomials(const Eigen::MatrixXd collocationDesignVector, const int oldNumberOfCollocationPoints, Eigen::MatrixXd& collocationGuessStart, const int newNumberOfCollocationPoints, const Eigen::VectorXd thrustAndMassParameters, const double massParameter );




#endif  // TUDATBUNDLE_INTERPOLATEPOLYNOMIALS_H
