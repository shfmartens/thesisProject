#ifndef TUDATBUNDLE_APPLYCOLLOCATION_H
#define TUDATBUNDLE_APPLYCOLLOCATION_H


#include "Eigen/Core"
#include <map>

Eigen::VectorXd computeCollocationDeviationNorms(const Eigen::VectorXd collocationDefectVector, const int numberOfCollocationPoints);

Eigen::MatrixXd evaluateVectorFields(const Eigen::MatrixXd initialCollocationGuess, const int numberOfCollocationPoints);

void extractDurationAndDynamicsFromInput(const Eigen::MatrixXd initialCollocationGuess, const int numberOfCollocationPoints, Eigen::MatrixXd& oddPointsDynamics,  Eigen::VectorXd& timeIntervals);

void computeOddPoints(Eigen::VectorXd initialStateVector, Eigen::MatrixXd& internalPointsMatrix,int numberOfCollocationPoints, const double massParameter);

Eigen::VectorXd convertNodeTimes(Eigen::MatrixXd nodeTimesNormalized, double lowerBound, double upperBound);


void computeMeshStates(Eigen::VectorXd currentNodeAndTime, Eigen::VectorXd nextNodeAndTime, Eigen::MatrixXd& statesNodesAndInteriorPoints, Eigen::MatrixXd& statesDefectPoints);

void retrieveLegendreGaussLobattoConstaints(const std::string desiredQuantity, Eigen::MatrixXd& outputMatrix);

void computeCollocationDefects(Eigen::MatrixXd& collocationDefectVector, Eigen::MatrixXd& collocationDesignVector, const Eigen::MatrixXd oddStates, const Eigen::MatrixXd oddStatesDerivatives, Eigen::VectorXd timeIntervals, Eigen::VectorXd thrustAndMassParameters, const int numberOfCollocationPoints);


Eigen::VectorXd applyCollocation(const Eigen::MatrixXd initialCollocationGuesss, const double massParameter, const int numberOfCollocationPoints, Eigen::VectorXd& collocatedGuess,
                                                         const double maxPositionDeviationFromPeriodicOrbit, const double maxVelocityDeviationFromPeriodicOrbit, const double maxPeriodDeviationFromPeriodicOrbit, const int maxNumberOfIterations = 10 );



#endif  // TUDATBUNDLE_APPLYCOLLOCATION_H
