#ifndef TUDATBUNDLE_APPLYCOLLOCATION_H
#define TUDATBUNDLE_APPLYCOLLOCATION_H


#include "Eigen/Core"
#include <map>

void computeOddPoints(Eigen::VectorXd initialStateVector, Eigen::MatrixXd& internalPointsMatrix,int numberOfCollocationPoints, const double massParameter);

Eigen::VectorXd convertNodeTimes(Eigen::MatrixXd nodeTimesNormalized, double lowerBound, double upperBound);


void computeMeshStates(Eigen::VectorXd currentNodeAndTime, Eigen::VectorXd nextNodeAndTime, Eigen::MatrixXd& statesNodesAndInteriorPoints, Eigen::MatrixXd& statesDefectPoints);

void retrieveLegendreGaussLobattoConstaints(const std::string desiredQuantity, Eigen::MatrixXd& outputMatrix);

Eigen::VectorXd applyCollocation(const Eigen::MatrixXd initialCollocationGuesss, const double massParameter, const int numberOfCollocationPoints, Eigen::VectorXd& collocatedGuess,
                                                         const double maxPositionDeviationFromPeriodicOrbit, const double maxVelocityDeviationFromPeriodicOrbit, const double maxPeriodDeviationFromPeriodicOrbit, const int maxNumberOfIterations = 10 );



#endif  // TUDATBUNDLE_APPLYCOLLOCATION_H
