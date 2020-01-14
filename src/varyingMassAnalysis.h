#ifndef TUDATBUNDLE_VARYINGMASSANALYSIS_H
#define TUDATBUNDLE_VARYINGMASSANALYSIS_H



#include "Eigen/Core"

void saveVaryingMassResultsToTextFile(const int librationPointNr, const double accMag, const double accAngle, const int member, const int specificImpulse,
                                      const std::map<double, Eigen::VectorXd> constantStateHistory,
                                      const std::map<double, Eigen::VectorXd> varyingStateHistory);

void retrieveInitialState(int testCaseNumber, Eigen::VectorXd& initialStateVector, double& orbitalPeriod, double& hamiltonian, int& lagrangePointNr, int& member);
void varyingMassAnalysis(const int testCaseNumber, const double massParameter);


#endif  // TUDATBUNDLE_VARYINGMASSANALYSIS_H
