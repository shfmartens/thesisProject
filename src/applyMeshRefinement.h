#ifndef TUDATBUNDLE_APPLYMESHREFINEMENT_H
#define TUDATBUNDLE_APPLYMESHREFINEMENT_H

#include "Tudat/Basics/basicTypedefs.h"

#include <Eigen/Core>
#include <string>
#include <vector>

void computeNewMesh(const Eigen::VectorXd collocationDesignVector, const Eigen::VectorXd thrustAndMassParameters, const Eigen::VectorXd nodeTimes, const Eigen::VectorXd newNodeTimes, const int numberOfCollocationPoints, Eigen::VectorXd& newDesignVector);

void computeTimeIntervals(const Eigen::VectorXd collocationDesignVector, const int numberOfCollocationPoints, Eigen::VectorXd& timeIntervals, Eigen::VectorXd& nodeTimes);

void computeSegmentDerivatives(Eigen::MatrixXd& segmentDerivatives, Eigen::MatrixXd& oddStates,Eigen::MatrixXd& oddStateDerivatives, const Eigen::VectorXd timeIntervals, const int numberOfCollocationPoints);

void computeSegmentProperties(const Eigen::VectorXd collocationDesignVector, const Eigen::VectorXd thrustAndMassParameters, const int numberOfCollocationPoints, Eigen::MatrixXd& oddStates, Eigen::MatrixXd& oddStateDervatives, Eigen::VectorXd& timeIntervals);

void computeSegmentErrors(Eigen::VectorXd collocationDesignVector, const Eigen::VectorXd thrustAndMassParameters, int numberOfCollocationPoints, Eigen::VectorXd& segmentErrors, Eigen::VectorXd& eightOrderDerivatives, const double computableConstant = 2.93579395141895E-9 );

void applyMeshRefinement(Eigen::MatrixXd& collocationDesignVector, const Eigen::VectorXd thrustAndMassParameters, int numberOfCollocationPoints );




#endif  // TUDATBUNDLE_APPLYMESHREFINEMENT_H
