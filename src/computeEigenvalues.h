#ifndef TUDATBUNDLE_COMPUTEEIGENVALUES_H
#define TUDATBUNDLE_COMPUTEEIGENVALUES_H



#include <Eigen/Core>


std::vector<double> computeEigenvalues( const Eigen::VectorXd& stateVectorInclSTM );



#endif  // TUDATBUNDLE_COMPUTEEIGENVALUES_H
