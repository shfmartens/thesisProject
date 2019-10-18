#ifndef TUDATBUNDLE_APPLYCOLLOCATION_H
#define TUDATBUNDLE_APPLYCOLLOCATION_H


#include "Eigen/Core"
#include <map>

void checkMeshTiming(const Eigen::MatrixXd collocationDesignVector, const int numberOfCollocationPoints, const Eigen::VectorXd thrustAndMassParameters);

Eigen::VectorXd computeProcedureTimeShifts(Eigen::VectorXd collocationDesignVectorInitial, Eigen::VectorXd collocationDesignVectorFinal, const int numberOfCollocationPoints);

void  writeTrajectoryErrorDataToFile(const int numberOfCollocationPoints, const Eigen::VectorXd fullPeriodDeviations, const Eigen::VectorXd defectVectorMS, const Eigen::VectorXd collocatedDefects, const Eigen::VectorXd collcationSegmentErrors, const int magnitudeNoiseOffset, const double amplitude );

Eigen::VectorXd rewriteOddPointsToVector(const Eigen::MatrixXd& oddNodesMatrix, const int numberOfCollocationPoints);

void shiftTimeOfConvergedCollocatedGuess(const Eigen::MatrixXd collocationDesignVector, Eigen::VectorXd& collocatedGuess, Eigen::VectorXd& collocatedNodes, const int numberOfCollocationPoints, Eigen::VectorXd thrustAndMassParameters);

void propagateAndSaveCollocationProcedure(const Eigen::MatrixXd oddPointsInput, Eigen::VectorXd timeIntervals, Eigen::VectorXd thrustAndMassParameters, const int numberOfCollocationPoints, const int typeOfInput, const double massParameter);

Eigen::VectorXd computeCollocationDeviationNorms(const Eigen::MatrixXd collocationDefectVector, const Eigen::MatrixXd collocationDesignVector, const int numberOfCollocationPoints);

Eigen::MatrixXd evaluateVectorFields(const Eigen::MatrixXd initialCollocationGuess, const int numberOfCollocationPoints);

void extractDurationAndDynamicsFromInput(const Eigen::MatrixXd initialCollocationGuess, const int numberOfCollocationPoints, Eigen::MatrixXd& oddPointsDynamics,  Eigen::VectorXd& timeIntervals);

void computeOddPoints(Eigen::VectorXd initialStateVector, Eigen::MatrixXd& internalPointsMatrix,int numberOfCollocationPoints, const double massParameter, bool firstCollocationGuess);

Eigen::VectorXd convertNodeTimes(Eigen::MatrixXd nodeTimesNormalized, double lowerBound, double upperBound);


void computeMeshStates(Eigen::VectorXd currentNodeAndTime, Eigen::VectorXd nextNodeAndTime, Eigen::MatrixXd& statesNodesAndInteriorPoints, Eigen::MatrixXd& statesDefectPoints);

void retrieveLegendreGaussLobattoConstaints(const std::string desiredQuantity, Eigen::MatrixXd& outputMatrix);

void computeCollocationDefects(Eigen::MatrixXd& collocationDefectVector, Eigen::MatrixXd& collocationDesignVector, const Eigen::MatrixXd oddStates, const Eigen::MatrixXd oddStatesDerivatives, Eigen::VectorXd timeIntervals, Eigen::VectorXd thrustAndMassParameters, const int numberOfCollocationPoints, const double initialTime, const int continuationIndex, const Eigen::VectorXd previousDesignVector, const int orbitNumber);


Eigen::VectorXd applyCollocation(const Eigen::MatrixXd initialCollocationGuesss, const double massParameter, int& numberOfCollocationPoints, Eigen::VectorXd& collocatedGuess, Eigen::VectorXd& collocatedNodes, Eigen::VectorXd& deviationNorms, Eigen::VectorXd& collocatedDefects, const int continuationIndex, const Eigen::VectorXd previousDesignVector, const int orbitNumber,
                                                          double maxPositionDeviationFromPeriodicOrbit,  double maxVelocityDeviationFromPeriodicOrbit,  double maxPeriodDeviationFromPeriodicOrbit, const int maxNumberOfCollocationIterations = 500, const double maximumErrorTolerance = 1.0E-9  );



#endif  // TUDATBUNDLE_APPLYCOLLOCATION_H
