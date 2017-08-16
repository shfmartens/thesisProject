#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>
#include <Eigen/Dense>

#include "applyDifferentialCorrection.h"
#include "checkEigenvalues.h"
#include "computeEigenvalues.h"
#include "computeManifolds.h"
#include "propagateOrbit.h"
#include "richardsonThirdOrderApproximation.h"
#include "writePeriodicOrbitToFile.h"



void createInitialConditions( int librationPointNr, std::string orbitType,
                              const double primaryGravitationalParameter = tudat::celestial_body_constants::EARTH_GRAVITATIONAL_PARAMETER,
                              const double secondaryGravitationalParameter = tudat::celestial_body_constants::MOON_GRAVITATIONAL_PARAMETER,
                              double maxPositionDeviationFromPeriodicOrbit = 1.0e-12, double maxVelocityDeviationFromPeriodicOrbit = 1.0e-12,
                              double maxEigenvalueDeviation = 1.0e-3 )
{
    std::cout << "\nCreate initial conditions:\n" << std::endl;

    // Set output maximum precision
    std::cout.precision(std::numeric_limits<double>::digits10);

    // Initialize state vectors and orbital periods
    double orbitalPeriod               = 0.0;
    double jacobiEnergy                = 0.0;
    double jacobiEnergyHalfPeriod      = 0.0;
    double pseudoArcLengthCorrection   = 0.0;
    bool continueNumericalContinuation = true;
    Eigen::VectorXd initialStateVector = Eigen::VectorXd::Zero(6);
    Eigen::VectorXd delta              = Eigen::VectorXd::Zero(7);
    Eigen::VectorXd stateVectorInclSTM = Eigen::VectorXd::Zero(42);
    std::vector< std::vector <double> > initialConditions;
    std::vector< std::vector <double> > differentialCorrections;
    std::vector<double>                 tempInitialCondition;
    std::vector<double>                 tempDifferentialCorrection;
    std::vector<double>                 eigenvalues;
    Eigen::VectorXd                     outputVector(43);
    Eigen::VectorXd                     differentialCorrectionResult;
    Eigen::VectorXd                     richardsonThirdOrderApproximationResult;

    // Define massParameter
    massParameter = tudat::gravitation::circular_restricted_three_body_problem::computeMassParameter( primaryGravitationalParameter, secondaryGravitationalParameter );

    if (orbitType == "horizontal") {
        if (librationPointNr == 1) {
            richardsonThirdOrderApproximationResult = richardsonThirdOrderApproximation("horizontal", 1, 1.0e-3);
        } else if (librationPointNr == 2) {
            richardsonThirdOrderApproximationResult = richardsonThirdOrderApproximation("horizontal", 2, 1.0e-4);
        }
    } else if (orbitType == "vertical"){
        if (librationPointNr == 1){
            richardsonThirdOrderApproximationResult = richardsonThirdOrderApproximation("vertical", 1, 1.0e-1);
        } else if (librationPointNr == 2){
            richardsonThirdOrderApproximationResult = richardsonThirdOrderApproximation("vertical", 2, 1.0e-1);
        }
    } else if (orbitType == "halo") {
        if (librationPointNr == 1) {
            richardsonThirdOrderApproximationResult = richardsonThirdOrderApproximation("halo", 1, 1.1e-1, 3.0);
        } else if (librationPointNr == 2) {
            richardsonThirdOrderApproximationResult = richardsonThirdOrderApproximation("halo", 2, 1.5e-1);
        }
    }

    // Split input parameters
    initialStateVector = richardsonThirdOrderApproximationResult.segment(0,6);
    orbitalPeriod      = richardsonThirdOrderApproximationResult(6);

    // Correct state vector guesses
    differentialCorrectionResult = applyDifferentialCorrection( librationPointNr, orbitType, initialStateVector, orbitalPeriod, massParameter, maxPositionDeviationFromPeriodicOrbit, maxVelocityDeviationFromPeriodicOrbit);
    initialStateVector           = differentialCorrectionResult.segment(0,6);
    orbitalPeriod                = differentialCorrectionResult(6);

//    computeManifolds( initialStateVector, orbitalPeriod, librationPointNr, orbitType, 0);

    // Save number of iterations, jacobi energy, time of integration and the half period state vector
    jacobiEnergyHalfPeriod       = tudat::gravitation::circular_restricted_three_body_problem::computeJacobiEnergy(massParameter, differentialCorrectionResult.segment(7,6));

    tempDifferentialCorrection.clear();
    tempDifferentialCorrection.push_back( differentialCorrectionResult(14) );  // numberOfIterations
    tempDifferentialCorrection.push_back( jacobiEnergyHalfPeriod );  // jacobiEnergyHalfPeriod
    tempDifferentialCorrection.push_back( differentialCorrectionResult(13) );  // currentTime
    for (int i = 7; i <= 12; i++){
        tempDifferentialCorrection.push_back( differentialCorrectionResult(i) );  // halfPeriodStateVector
    }
    differentialCorrections.push_back(tempDifferentialCorrection);

    // Propagate the initialStateVector for a full period and write output to file.
    stateVectorInclSTM = writePeriodicOrbitToFile( initialStateVector, librationPointNr, orbitType, 0, orbitalPeriod, massParameter);

    // Save jacobi energy, orbital period, initial condition, and eigenvalues
    tempInitialCondition.clear();

    // Add Jacobi energy and orbital period
    jacobiEnergy = tudat::gravitation::circular_restricted_three_body_problem::computeJacobiEnergy(massParameter, stateVectorInclSTM.segment(0,6));
    tempInitialCondition.push_back(jacobiEnergy);
    tempInitialCondition.push_back(orbitalPeriod);

    // Add initial condition of periodic solution
    for (int i = 0; i <= 5; i++){
        tempInitialCondition.push_back(initialStateVector(i));
    }

    // Add Monodromy matrix
    for (int i = 6; i <= 41; i++){
        tempInitialCondition.push_back(stateVectorInclSTM(i));
    }

    initialConditions.push_back(tempInitialCondition);

    // Define second state vector guess
    initialStateVector = Eigen::VectorXd::Zero(6);
    if (orbitType == "horizontal") {
        if (librationPointNr == 1) {
            richardsonThirdOrderApproximationResult = richardsonThirdOrderApproximation("horizontal", 1, 1.0e-4);
        } else if (librationPointNr == 2) {
            richardsonThirdOrderApproximationResult = richardsonThirdOrderApproximation("horizontal", 2, 1.0e-3);
        }
    } else if (orbitType == "vertical"){
        if (librationPointNr == 1){
            richardsonThirdOrderApproximationResult = richardsonThirdOrderApproximation("vertical", 1, 2.0e-1);
        } else if (librationPointNr == 2){
            richardsonThirdOrderApproximationResult = richardsonThirdOrderApproximation("vertical", 2, 2.0e-1);
        }
    } else if (orbitType == "halo") {
        if (librationPointNr == 1) {
            richardsonThirdOrderApproximationResult = richardsonThirdOrderApproximation("halo", 1, 1.2e-1, 3.0);
        } else if (librationPointNr == 2) {
            richardsonThirdOrderApproximationResult = richardsonThirdOrderApproximation("halo", 2, 1.6e-1);
        }
    }

    // Split input parameters
    initialStateVector = richardsonThirdOrderApproximationResult.segment(0,6);
    orbitalPeriod = richardsonThirdOrderApproximationResult(6);

    // Correct state vector guesses
    differentialCorrectionResult = applyDifferentialCorrection( librationPointNr, orbitType, initialStateVector, orbitalPeriod, massParameter, maxPositionDeviationFromPeriodicOrbit, maxVelocityDeviationFromPeriodicOrbit);
    initialStateVector           = differentialCorrectionResult.segment(0,6);
    orbitalPeriod                = differentialCorrectionResult(6);

    // Save number of iterations, jacobi energy, time of integration and the half period state vector
    jacobiEnergyHalfPeriod       = tudat::gravitation::circular_restricted_three_body_problem::computeJacobiEnergy(massParameter, differentialCorrectionResult.segment(7,6));

    tempDifferentialCorrection.clear();
    tempDifferentialCorrection.push_back( differentialCorrectionResult(14) );  // numberOfIterations
    tempDifferentialCorrection.push_back( jacobiEnergyHalfPeriod );  // jacobiEnergyHalfPeriod
    tempDifferentialCorrection.push_back( differentialCorrectionResult(13) );  // currentTime
    for (int i = 7; i <= 12; i++){
        tempDifferentialCorrection.push_back( differentialCorrectionResult(i) );  // halfPeriodStateVector
    }
    differentialCorrections.push_back(tempDifferentialCorrection);

    // Propagate the initialStateVector for a full period and write output to file.
    stateVectorInclSTM = writePeriodicOrbitToFile( initialStateVector, librationPointNr, orbitType, 1, orbitalPeriod, massParameter);

    // Save jacobi energy, orbital period, initial condition, and eigenvalues
    tempInitialCondition.clear();

    // Add Jacobi energy and orbital period
    jacobiEnergy = tudat::gravitation::circular_restricted_three_body_problem::computeJacobiEnergy(massParameter, stateVectorInclSTM.segment(0,6));
    tempInitialCondition.push_back(jacobiEnergy);
    tempInitialCondition.push_back(orbitalPeriod);

    // Add initial condition of periodic solution
    for (int i = 0; i <= 5; i++){
        tempInitialCondition.push_back(initialStateVector(i));
    }

    // Add Monodromy matrix
    for (int i = 6; i <= 41; i++){
        tempInitialCondition.push_back(stateVectorInclSTM(i));
    }

    initialConditions.push_back(tempInitialCondition);

    // Set exit parameters of continuation procedure
    int numberOfInitialConditions = 2;
    int maximumNumberOfInitialConditions = 4000;

    while (numberOfInitialConditions < maximumNumberOfInitialConditions and continueNumericalContinuation){

        continueNumericalContinuation = false;

        delta = Eigen::VectorXd::Zero(7);
        delta(0) = initialConditions[initialConditions.size()-1][0+2] - initialConditions[initialConditions.size()-2][0+2];
        delta(1) = initialConditions[initialConditions.size()-1][1+2] - initialConditions[initialConditions.size()-2][1+2];
        delta(2) = initialConditions[initialConditions.size()-1][2+2] - initialConditions[initialConditions.size()-2][2+2];
        delta(3) = initialConditions[initialConditions.size()-1][3+2] - initialConditions[initialConditions.size()-2][3+2];
        delta(4) = initialConditions[initialConditions.size()-1][4+2] - initialConditions[initialConditions.size()-2][4+2];
        delta(5) = initialConditions[initialConditions.size()-1][5+2] - initialConditions[initialConditions.size()-2][5+2];
        delta(6) = initialConditions[initialConditions.size()-1][1]   - initialConditions[initialConditions.size()-2][1];

        pseudoArcLengthCorrection = 1e-4 / sqrt(pow(delta(0),2) + pow(delta(1),2) + pow(delta(2),2));

        std::cout << "pseudoArcCorrection: " << pseudoArcLengthCorrection << std::endl;

        // Apply numerical continuation
        initialStateVector = Eigen::VectorXd::Zero(6);
        initialStateVector(0) = initialConditions[initialConditions.size()-1][0+2] + delta(0) * pseudoArcLengthCorrection;
        initialStateVector(1) = initialConditions[initialConditions.size()-1][1+2] + delta(1) * pseudoArcLengthCorrection;
        initialStateVector(2) = initialConditions[initialConditions.size()-1][2+2] + delta(2) * pseudoArcLengthCorrection;
        initialStateVector(3) = initialConditions[initialConditions.size()-1][3+2] + delta(3) * pseudoArcLengthCorrection;
        initialStateVector(4) = initialConditions[initialConditions.size()-1][4+2] + delta(4) * pseudoArcLengthCorrection;
        initialStateVector(5) = initialConditions[initialConditions.size()-1][5+2] + delta(5) * pseudoArcLengthCorrection;
        orbitalPeriod         = initialConditions[initialConditions.size()-1][1]   + delta(6) * pseudoArcLengthCorrection;

        // Correct state vector guesses
        differentialCorrectionResult = applyDifferentialCorrection( librationPointNr, orbitType, initialStateVector, orbitalPeriod, massParameter, maxPositionDeviationFromPeriodicOrbit, maxVelocityDeviationFromPeriodicOrbit);
        if (differentialCorrectionResult == Eigen::VectorXd::Zero(15)){
            continueNumericalContinuation = false;
            std::cout << "\n\nNUMERICAL CONTINUATION STOPPED DUE TO EXCEEDING MAXIMUM NUMBER OF ITERATIONS\n\n" << std::endl;
            break;
        }
        initialStateVector           = differentialCorrectionResult.segment(0,6);
        orbitalPeriod                = differentialCorrectionResult(6);

        // Save number of iterations, jacobi energy, time of integration and the half period state vector
        jacobiEnergyHalfPeriod       = tudat::gravitation::circular_restricted_three_body_problem::computeJacobiEnergy(massParameter, differentialCorrectionResult.segment(7,6));

        tempDifferentialCorrection.clear();
        tempDifferentialCorrection.push_back( differentialCorrectionResult(14) );  // numberOfIterations
        tempDifferentialCorrection.push_back( jacobiEnergyHalfPeriod );  // jacobiEnergyHalfPeriod
        tempDifferentialCorrection.push_back( differentialCorrectionResult(13) );  // currentTime
        for (int i = 7; i <= 12; i++){
            tempDifferentialCorrection.push_back( differentialCorrectionResult(i) );  // halfPeriodStateVector
        }
        differentialCorrections.push_back(tempDifferentialCorrection);

        // Propagate the initialStateVector for a full period and write output to file.
        stateVectorInclSTM = writePeriodicOrbitToFile( initialStateVector, librationPointNr, orbitType, numberOfInitialConditions, orbitalPeriod, massParameter);

        // Check eigenvalue condition (at least one pair equalling a real one)
        // Exception for the horizontal Lyapunov family in Earth-Moon L2: eigenvalue may be of module one instead of a real one to compute a more extensive family
        if ( librationPointNr == 2 and orbitType == "horizontal" ){
            continueNumericalContinuation = checkEigenvalues(stateVectorInclSTM, maxEigenvalueDeviation, true);
        } else {
            continueNumericalContinuation = checkEigenvalues(stateVectorInclSTM, maxEigenvalueDeviation, false);
        }

        // Save jacobi energy, orbital period, initial condition, and eigenvalues
        tempInitialCondition.clear();

        // Add Jacobi energy and orbital period
        jacobiEnergy = tudat::gravitation::circular_restricted_three_body_problem::computeJacobiEnergy(massParameter, initialStateVector);
        tempInitialCondition.push_back(jacobiEnergy);
        tempInitialCondition.push_back(orbitalPeriod);

        // Add initial condition of periodic solution
        for (int i = 0; i <= 5; i++){
            tempInitialCondition.push_back(initialStateVector(i));
        }

        // Add Monodromy matrix
        for (int i = 6; i <= 41; i++){
            tempInitialCondition.push_back(stateVectorInclSTM(i));
        }

        initialConditions.push_back(tempInitialCondition);
        numberOfInitialConditions += 1;
    }

    // Prepare file for initial conditions
    remove(("../data/raw/L" + std::to_string(librationPointNr) + "_" + orbitType + "_initial_conditions.txt").c_str());
    std::ofstream textFileInitialConditions;
    textFileInitialConditions.open(("../data/raw/L" + std::to_string(librationPointNr) + "_" + orbitType + "_initial_conditions.txt").c_str());
    textFileInitialConditions.precision(std::numeric_limits<double>::digits10);

    // Prepare file for differential correction
    remove(("../data/raw/L" + std::to_string(librationPointNr) + "_" + orbitType + "_differential_correction.txt").c_str());
    std::ofstream textFileDifferentialCorrection;
    textFileDifferentialCorrection.open(("../data/raw/L" + std::to_string(librationPointNr) + "_" + orbitType + "_differential_correction.txt").c_str());
    textFileDifferentialCorrection.precision(std::numeric_limits<double>::digits10);

    // Write initial conditions to file
    for (unsigned int i=0; i<initialConditions.size(); i++) {

        textFileInitialConditions << std::left << std::scientific                                          << std::setw(25)
                                  << initialConditions[i][0]  << std::setw(25) << initialConditions[i][1]  << std::setw(25)
                                  << initialConditions[i][2]  << std::setw(25) << initialConditions[i][3]  << std::setw(25)
                                  << initialConditions[i][4]  << std::setw(25) << initialConditions[i][5]  << std::setw(25)
                                  << initialConditions[i][6]  << std::setw(25) << initialConditions[i][7]  << std::setw(25)
                                  << initialConditions[i][8]  << std::setw(25) << initialConditions[i][9]  << std::setw(25)
                                  << initialConditions[i][10] << std::setw(25) << initialConditions[i][11] << std::setw(25)
                                  << initialConditions[i][12] << std::setw(25) << initialConditions[i][13] << std::setw(25)
                                  << initialConditions[i][14] << std::setw(25) << initialConditions[i][15] << std::setw(25)
                                  << initialConditions[i][16] << std::setw(25) << initialConditions[i][17] << std::setw(25)
                                  << initialConditions[i][18] << std::setw(25) << initialConditions[i][19] << std::setw(25)
                                  << initialConditions[i][20] << std::setw(25) << initialConditions[i][21] << std::setw(25)
                                  << initialConditions[i][22] << std::setw(25) << initialConditions[i][23] << std::setw(25)
                                  << initialConditions[i][24] << std::setw(25) << initialConditions[i][25] << std::setw(25)
                                  << initialConditions[i][26] << std::setw(25) << initialConditions[i][27] << std::setw(25)
                                  << initialConditions[i][28] << std::setw(25) << initialConditions[i][29] << std::setw(25)
                                  << initialConditions[i][30] << std::setw(25) << initialConditions[i][31] << std::setw(25)
                                  << initialConditions[i][32] << std::setw(25) << initialConditions[i][33] << std::setw(25)
                                  << initialConditions[i][34] << std::setw(25) << initialConditions[i][35] << std::setw(25)
                                  << initialConditions[i][36] << std::setw(25) << initialConditions[i][37] << std::setw(25)
                                  << initialConditions[i][38] << std::setw(25) << initialConditions[i][39] << std::setw(25)
                                  << initialConditions[i][40] << std::setw(25) << initialConditions[i][41] << std::setw(25)
                                  << initialConditions[i][42] << std::setw(25) << initialConditions[i][43] << std::endl;

        textFileDifferentialCorrection << std::left << std::scientific   << std::setw(25)
                                       << differentialCorrections[i][0]  << std::setw(25) << differentialCorrections[i][1]  << std::setw(25)
                                       << differentialCorrections[i][2]  << std::setw(25) << differentialCorrections[i][3]  << std::setw(25)
                                       << differentialCorrections[i][4]  << std::setw(25) << differentialCorrections[i][5]  << std::setw(25)
                                       << differentialCorrections[i][6]  << std::setw(25) << differentialCorrections[i][7]  << std::setw(25)
                                       << differentialCorrections[i][8]  << std::setw(25) << std::endl;
    }
}
