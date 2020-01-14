#include <fstream>
#include <iomanip>
#include <iostream>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>
#include <Eigen/Dense>
#include <boost/function.hpp>
#include <random>
#include <sstream>


#include "Tudat/Astrodynamics/BasicAstrodynamics/celestialBodyConstants.h"
#include "Tudat/Astrodynamics/Gravitation/librationPoint.h"
#include "Tudat/Astrodynamics/Gravitation/jacobiEnergy.h"
#include "Tudat/InputOutput/basicInputOutput.h"


#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>
#include <Eigen/Dense>
#include "refineOrbitHamiltonian.h"
#include "applyCollocation.h"
#include "createLowThrustInitialConditions.h"
#include "propagateOrbitAugmented.h"
#include "propagateMassVaryingOrbitAugmented.h"

void saveVaryingMassResultsToTextFile(const int librationPointNr, const double accMag, const double accAngle, const int member, const int specificImpulse,
                                      const std::map<double, Eigen::VectorXd> constantStateHistory,
                                      const std::map<double, Eigen::VectorXd> varyingStateHistory)
{
    std::ostringstream ssAccelerationMagnitude;
    ssAccelerationMagnitude << std::fixed <<std::setprecision(11) << accMag;
    std::string stringAccelerationMagnitude = ssAccelerationMagnitude.str();

    std::ostringstream ssAccelerationAngle1;
    ssAccelerationAngle1 << std::fixed <<  std::setprecision(11) << accAngle;
    std::string stringAccelerationAngle1 = ssAccelerationAngle1.str();


    std::string fileNameStringConstant;
    std::string fileNameStringVarying;

    std::string directoryString = "../data/raw/varying_mass/";

    fileNameStringConstant = ("L" +std::to_string(librationPointNr) + "_" + stringAccelerationMagnitude + "_" + stringAccelerationAngle1 + "_" + std::to_string(member) + "_"  + std::to_string(specificImpulse) +  "_constantMass.txt");
    fileNameStringVarying = ("L" +std::to_string(librationPointNr) + "_" + stringAccelerationMagnitude + "_" + stringAccelerationAngle1 + "_" + std::to_string(member)  + "_"  + std::to_string(specificImpulse) + "_varyingMass.txt");

    tudat::input_output::writeDataMapToTextFile( constantStateHistory, fileNameStringConstant, directoryString );
    tudat::input_output::writeDataMapToTextFile( varyingStateHistory, fileNameStringVarying, directoryString );


}

void retrieveInitialState(int testCaseNumber, Eigen::VectorXd& initialStateVector, double& orbitalPeriod, double& hamiltonian, int& lagrangePointNr, int& member)
{
    if (testCaseNumber == 1)
    {
            // L1, alt = 0.01, alpha = 0, HLT-var, member = 1938
                // HLT:    -1.462009914700207
                // Period:  6.537957262589215


            initialStateVector(0) = 6.412118120115954e-01;
            initialStateVector(1) = -1.174798166073575e-13;
            initialStateVector(2) = 0.0;
            initialStateVector(3) = 1.434118764075372e-13;
            initialStateVector(4) = 7.706858702952796e-01;
            initialStateVector(5) = 0.0;
            initialStateVector(6) = 1.000000000000000e-02;
            initialStateVector(7) = 0.0;
            initialStateVector(8) = 0.0;
            initialStateVector(9) = 1.0;

            orbitalPeriod = 6.499910706630720;
            hamiltonian = -1.462009914700207;
            lagrangePointNr = 1;
            member = 1938;
    }

    if (testCaseNumber == 2)
    {
        // L1, alt = 0.05, alpha = 0, HLT-var, member = 2113
            // HLT:    -1.487190282472479
            // Period:  6.347544478985032

            initialStateVector(0) = 6.200654481720653e-01;
            initialStateVector(1) = -1.839198263937863e-13;
            initialStateVector(2) = 0.000000000000000;
            initialStateVector(3) = 3.463392467039106e-14;
            initialStateVector(4) = 8.143827451200628e-01;
            initialStateVector(5) = 0.0;
            initialStateVector(6) = 5.000000000000000e-02;
            initialStateVector(7) = 0.0;
            initialStateVector(8) = 0.0;
            initialStateVector(9) = 1.0;

            orbitalPeriod = 6.347544478985032;
            hamiltonian = -1.487190282472479;
            lagrangePointNr = 1;
            member = 2113;

    }

    if (testCaseNumber == 3)
    {

        // L1, alt = 0.1, alpha = 0, HLT-var, member = 2167
            // HLT:    -1.522927333320343
            // Period:  6.062806853409906

            initialStateVector(0) = 6.099078591638476e-01;
            initialStateVector(1) = 1.445578058876212e-13;
            initialStateVector(2) = 0.000000000000000;
            initialStateVector(3) = -2.049406805393628e-14;
            initialStateVector(4) = 8.297466566445242e-01;
            initialStateVector(5) = 0.0;
            initialStateVector(6) = 1.000000000000000e-01;
            initialStateVector(7) = 0.0;
            initialStateVector(8) = 0.0;
            initialStateVector(9) = 1.0;

            orbitalPeriod = 6.062806853409906;
            hamiltonian = -1.522927333320343;
            lagrangePointNr = 1;
            member = 2167;

    }
}

void varyingMassAnalysis(const int testCaseNumber, const double massParameter)
{

    // Retrieve initialState, orbital period and HAmiltonian,
    Eigen::VectorXd initialStateVector(10); initialStateVector.setZero();
    double orbitalPeriod;
    double hamiltonian;
    int specificImpulse = 3000;
    int lagrangePointNr;
    int member;

    retrieveInitialState(testCaseNumber,initialStateVector, orbitalPeriod, hamiltonian, lagrangePointNr, member);
    Eigen::MatrixXd fullInitialState(10,11);
    fullInitialState.block(0,0,10,1) = initialStateVector;
    fullInitialState.block(0,1,10,10).setIdentity();

    std::cout << "\n === VARYING MASS ANALYSIS =====" << std::endl
              << "test case: " <<   testCaseNumber << std::endl
              << "initialStateVector: " <<   fullInitialState.block(0,0,10,1) << std::endl
              << "orbitalPeriod: " <<   orbitalPeriod << std::endl
              << "hamiltonian: " <<   hamiltonian << std::endl;


    std::cout << "\n == Propagating the constant mass trajectory == " << std ::endl;
    std::map<double, Eigen::VectorXd> stateHistoryConstant;
    std::pair< Eigen::MatrixXd, double > stateVectorInclSTMAndTime = propagateOrbitAugmentedToFinalCondition(fullInitialState, massParameter, orbitalPeriod, 1,
                                                                                        stateHistoryConstant, 1000, 0.0);

    Eigen::MatrixXd stateVectorInclSTMConstant = stateVectorInclSTMAndTime.first;
    double timeConstant = stateVectorInclSTMAndTime.second;
    Eigen::VectorXd stateVectorConstant = stateVectorInclSTMConstant.block(0,0,10,1);

    std::cout << "\n == Constant Mass trajectory propagated == " << std ::endl
              << "endTime: " << timeConstant << std::endl
              << "stateVectorConstant: " << stateVectorConstant << std::endl
              << "State vector discrepancy: " << initialStateVector - stateVectorConstant << std::endl;


    std::cout << "\n == Propagating the constant mass trajectory == " << std ::endl;
    std::map<double, Eigen::VectorXd> stateHistoryVarying;
    std::pair< Eigen::MatrixXd, double > stateVectorInclSTMAndTimeVarying = propagateMassVaryingOrbitAugmentedToFinalCondition(fullInitialState, massParameter, orbitalPeriod, 1,
                                                                                        stateHistoryVarying, 1000, 0.0);


    Eigen::MatrixXd stateVectorInclSTMVarying = stateVectorInclSTMAndTimeVarying.first;
    double timeVarying = stateVectorInclSTMAndTimeVarying.second;
    Eigen::VectorXd stateVectorVarying = stateVectorInclSTMVarying.block(0,0,10,1);

    std::cout << "\n == Varying Mass trajectory propagated == " << std ::endl
              << "endTime: " << timeVarying << std::endl
              << "stateVectorConstant: " << stateVectorVarying << std::endl
              << "State vector discrepancy: " << initialStateVector - stateVectorVarying << std::endl;

    // Saving information
    saveVaryingMassResultsToTextFile(lagrangePointNr, initialStateVector(6), initialStateVector(7), member, specificImpulse, stateHistoryConstant, stateHistoryVarying);

}


