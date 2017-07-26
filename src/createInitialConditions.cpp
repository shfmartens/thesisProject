#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>
#include <Eigen/Dense>

#include "thesisProject/src/applyDifferentialCorrection.h"
#include "thesisProject/src/computeDifferentialCorrectionHalo.h"
#include "thesisProject/src/computeEigenvalues.h"
#include "thesisProject/src/propagateOrbit.h"
#include "thesisProject/src/richardsonThirdOrderApproximation.h"
#include "thesisProject/src/writePeriodicOrbitToFile.h"


void createInitialConditions( int lagrangePointNr, string orbitType,
                              const double primaryGravitationalParameter = tudat::celestial_body_constants::EARTH_GRAVITATIONAL_PARAMETER,
                              const double secondaryGravitationalParameter = tudat::celestial_body_constants::MOON_GRAVITATIONAL_PARAMETER,
                              double maxPositionDeviationFromPeriodicOrbit = 1.0e-12, double maxVelocityDeviationFromPeriodicOrbit = 1.0e-12,
                              double maxDeviationEigenvalue = 1.0e-2)
{
    cout << "\nCreate initial conditions:\n" << endl;

    // Set output maximum precision
//    std::cout.precision(14);
    std::cout.precision(std::numeric_limits<double>::digits10);

    // Initialize state vectors and orbital periods
    double orbitalPeriod               = 0.0;
    double jacobiEnergy                = 0.0;
    double pseudoArcLengthCorrection   = 0.0;
    bool continueNumericalContinuation = true;
    Eigen::VectorXd initialStateVector = Eigen::VectorXd::Zero(6);
    Eigen::VectorXd delta              = Eigen::VectorXd::Zero(7);
    Eigen::VectorXd stateVectorInclSTM = Eigen::VectorXd::Zero(42);
    std::vector< std::vector <double> > initialStateVectors;
    std::vector<double> tempStateVector;
    std::vector<double> eigenvalues;
    Eigen::VectorXd outputVector(43);
    Eigen::VectorXd differentialCorrectionResult;
    Eigen::VectorXd richardsonThirdOrderApproximationResult;

    // Define massParameter
    massParameter = tudat::gravitation::circular_restricted_three_body_problem::computeMassParameter( primaryGravitationalParameter, secondaryGravitationalParameter );

    // TODO extend horizontal, compute halo and vertical
    // Define first state vector guess (smaller in amplitude than second)
    if (orbitType == "horizontal") {
        if (lagrangePointNr == 1) { // Ax = 1e-3
//            initialStateVector(0) = 0.836764423217002;
//            initialStateVector(4) = 0.00126332151282969;
//            orbitalPeriod         = 2.69158429793537;
            richardsonThirdOrderApproximationResult = richardsonThirdOrderApproximation("horizontal", 1, 1.0e-3);

        } else if (lagrangePointNr == 2) { // Az = 1e-4
//            initialStateVector(0) = 1.15566536294209;
//            initialStateVector(4) = 9.10547312525634e-5;
//            orbitalPeriod         = 3.37325810239639;
            richardsonThirdOrderApproximationResult = richardsonThirdOrderApproximation("horizontal", 2, 1.0e-4);
        }
    } else if (orbitType == "vertical"){
        if (lagrangePointNr == 1){ // Az = 1e-1
//            initialStateVector(0) = 0.837371102093524;
//            initialStateVector(2) = 0.0150902314188015;
//            initialStateVector(4) = -0.000322734270054828;
//            orbitalPeriod         = 2.68935340498007;
            richardsonThirdOrderApproximationResult = richardsonThirdOrderApproximation("vertical", 1, 1.0e-1);
        } else if (lagrangePointNr == 2){ // Az = 1e-1
//            initialStateVector(0) = 1.15502585527639;
//            initialStateVector(2) = 0.0167803298688178;
//            initialStateVector(4) = 0.000482238467409511;
//            orbitalPeriod         = 3.36831131220139;
            richardsonThirdOrderApproximationResult = richardsonThirdOrderApproximation("vertical", 2, 1.0e-1);
        }
    } else if (orbitType == "halo") {
        if (lagrangePointNr == 1) { // Az = 1.1e-1
//            initialStateVector(0) = 0.823842368103342;
//            initialStateVector(2) = 0.0177441810419053;
//            initialStateVector(4) = 0.130143004917572;
//            orbitalPeriod         = 2.74399197221861;
            richardsonThirdOrderApproximationResult = richardsonThirdOrderApproximation("halo", 1, 1.1e-1);
        } else if (lagrangePointNr == 2) { // Az = 1.5e-1
//            initialStateVector(0) = 1.11822154781651;
//            initialStateVector(2) = 0.0218987160505483;
//            initialStateVector(4) = 0.183307010794364;
//            orbitalPeriod         = 3.40300397070175;
            richardsonThirdOrderApproximationResult = richardsonThirdOrderApproximation("halo", 2, 1.5e-1);
        }
    }

    // Split input parameters
    initialStateVector = richardsonThirdOrderApproximationResult.segment(0,6);
    orbitalPeriod = richardsonThirdOrderApproximationResult(6);

    // Correct state vector guesses
    differentialCorrectionResult = applyDifferentialCorrection( orbitType, initialStateVector, orbitalPeriod, massParameter, maxPositionDeviationFromPeriodicOrbit, maxVelocityDeviationFromPeriodicOrbit);
    initialStateVector           = differentialCorrectionResult.segment(0,6);
    orbitalPeriod                = differentialCorrectionResult(6);

    // Propagate the initialStateVector for a full period and write output to file.
    stateVectorInclSTM = writePeriodicOrbitToFile( initialStateVector, lagrangePointNr, orbitType, 0, orbitalPeriod, massParameter);

    // Save jacobi energy, orbital period, initial condition, and eigenvalues
    tempStateVector.clear();

    // Add Jacobi energy and orbital period
    jacobiEnergy = tudat::gravitation::circular_restricted_three_body_problem::computeJacobiEnergy(massParameter, stateVectorInclSTM.segment(0,6));
    tempStateVector.push_back(jacobiEnergy);
    tempStateVector.push_back(orbitalPeriod);

    // Add initial condition of periodic solution
    for (int i = 0; i <= 5; i++){
        tempStateVector.push_back(initialStateVector(i));
    }

    // Add Monodromy matrix
    for (int i = 6; i <= 41; i++){
        tempStateVector.push_back(stateVectorInclSTM(i));
    }

    // Add eigenvalues
//    eigenvalues = computeEigenvalues(stateVectorInclSTM);

    initialStateVectors.push_back(tempStateVector);

    // Define second state vector guess
    initialStateVector = Eigen::VectorXd::Zero(6);
    if (orbitType == "horizontal") {
        if (lagrangePointNr == 1) { // Az = 1e-4
//            initialStateVector(0) = 0.836900057031702;
//            initialStateVector(4) = 0.000126362886457397;
//            orbitalPeriod         = 2.69157963713831;
            richardsonThirdOrderApproximationResult = richardsonThirdOrderApproximation("horizontal", 1, 1.0e-4);
        } else if (lagrangePointNr == 2) { // Az = 1e-3
//            initialStateVector(0) = 1.15551415840187;
//            initialStateVector(4) = 0.000910813629368759;
//            orbitalPeriod         = 3.37325926346013;
            richardsonThirdOrderApproximationResult = richardsonThirdOrderApproximation("horizontal", 2, 1.0e-3);
        }
    } else if (orbitType == "vertical"){
        if (lagrangePointNr == 1){ // Az = 2e-1
//            initialStateVector(0) = 0.838738963267659;
//            initialStateVector(2) = 0.0301612889821024;
//            initialStateVector(4) = -0.00129414025304457;
//            orbitalPeriod         = 2.68269689018263;
            richardsonThirdOrderApproximationResult = richardsonThirdOrderApproximation("vertical", 1, 2.0e-1);
        } else if (lagrangePointNr == 2){ // Az = 2e-1
//            initialStateVector(0) = 1.15305697775794;
//            initialStateVector(2) = 0.0335430016706184;
//            initialStateVector(4) = 0.00193744012287736;
//            orbitalPeriod         = 3.35355764707046;
            richardsonThirdOrderApproximationResult = richardsonThirdOrderApproximation("vertical", 2, 2.0e-1);
        }
    } else if (orbitType == "halo") {
        if (lagrangePointNr == 1) { // Az = 1.2e-1
//            initialStateVector(0) = 0.823853706063225;
//            initialStateVector(2) = 0.0193676274048036;
//            initialStateVector(4) = 0.131122874393244;
//            orbitalPeriod         = 2.74440271607439;
            richardsonThirdOrderApproximationResult = richardsonThirdOrderApproximation("halo", 1, 1.2e-1);
        } else if (lagrangePointNr == 2) { // Az = 1.6e-1
//            initialStateVector(0) = 1.11775138258286;
//            initialStateVector(2) = 0.0233369306514276;
//            initialStateVector(4) = 0.184805075718017;
//            orbitalPeriod         = 3.40197707259568;
            richardsonThirdOrderApproximationResult = richardsonThirdOrderApproximation("halo", 2, 1.6e-1);
        }
    }

    // Split input parameters
    initialStateVector = richardsonThirdOrderApproximationResult.segment(0,6);
    orbitalPeriod = richardsonThirdOrderApproximationResult(6);

    // Correct state vector guesses
    differentialCorrectionResult = applyDifferentialCorrection( orbitType, initialStateVector, orbitalPeriod, massParameter, maxPositionDeviationFromPeriodicOrbit, maxVelocityDeviationFromPeriodicOrbit);
    initialStateVector           = differentialCorrectionResult.segment(0,6);
    orbitalPeriod                = differentialCorrectionResult(6);

    // Propagate the initialStateVector for a full period and write output to file.
    stateVectorInclSTM = writePeriodicOrbitToFile( initialStateVector, lagrangePointNr, orbitType, 1, orbitalPeriod, massParameter);

    // Save jacobi energy, orbital period, initial condition, and eigenvalues
    tempStateVector.clear();

    // Add Jacobi energy and orbital period
    jacobiEnergy = tudat::gravitation::circular_restricted_three_body_problem::computeJacobiEnergy(massParameter, stateVectorInclSTM.segment(0,6));
    tempStateVector.push_back(jacobiEnergy);
    tempStateVector.push_back(orbitalPeriod);

    // Add initial condition of periodic solution
    for (int i = 0; i <= 5; i++){
        tempStateVector.push_back(initialStateVector(i));
    }

    // Add Monodromy matrix
    for (int i = 6; i <= 41; i++){
        tempStateVector.push_back(stateVectorInclSTM(i));
    }

    // Add eigenvalues
//    eigenvalues = computeEigenvalues(stateVectorInclSTM);

    initialStateVectors.push_back(tempStateVector);

    // Set exit parameters of continuation procedure
    int numberOfInitialConditions = 2;
    int maximumNumberOfInitialConditions = 3000;

    while (numberOfInitialConditions <= maximumNumberOfInitialConditions and continueNumericalContinuation){

        continueNumericalContinuation = false;

        delta = Eigen::VectorXd::Zero(7);
        delta(0) = initialStateVectors[initialStateVectors.size()-1][0+2] - initialStateVectors[initialStateVectors.size()-2][0+2];
        delta(1) = initialStateVectors[initialStateVectors.size()-1][1+2] - initialStateVectors[initialStateVectors.size()-2][1+2];
        delta(2) = initialStateVectors[initialStateVectors.size()-1][2+2] - initialStateVectors[initialStateVectors.size()-2][2+2];
        delta(3) = initialStateVectors[initialStateVectors.size()-1][3+2] - initialStateVectors[initialStateVectors.size()-2][3+2];
        delta(4) = initialStateVectors[initialStateVectors.size()-1][4+2] - initialStateVectors[initialStateVectors.size()-2][4+2];
        delta(5) = initialStateVectors[initialStateVectors.size()-1][5+2] - initialStateVectors[initialStateVectors.size()-2][5+2];
        delta(6) = initialStateVectors[initialStateVectors.size()-1][1]   - initialStateVectors[initialStateVectors.size()-2][1];

        pseudoArcLengthCorrection = 1e-4 / sqrt(pow(delta(0),2) + pow(delta(1),2) + pow(delta(2),2));

        cout << "pseudoArcCorrection: " << pseudoArcLengthCorrection << endl;

        // Apply numerical continuation
        initialStateVector = Eigen::VectorXd::Zero(6);
        initialStateVector(0) = initialStateVectors[initialStateVectors.size()-1][0+2] + delta(0) * pseudoArcLengthCorrection;
        initialStateVector(1) = initialStateVectors[initialStateVectors.size()-1][1+2] + delta(1) * pseudoArcLengthCorrection;
        initialStateVector(2) = initialStateVectors[initialStateVectors.size()-1][2+2] + delta(2) * pseudoArcLengthCorrection;
        initialStateVector(3) = initialStateVectors[initialStateVectors.size()-1][3+2] + delta(3) * pseudoArcLengthCorrection;
        initialStateVector(4) = initialStateVectors[initialStateVectors.size()-1][4+2] + delta(4) * pseudoArcLengthCorrection;
        initialStateVector(5) = initialStateVectors[initialStateVectors.size()-1][5+2] + delta(5) * pseudoArcLengthCorrection;
        orbitalPeriod         = initialStateVectors[initialStateVectors.size()-1][1]   + delta(6) * pseudoArcLengthCorrection;

        // Correct state vector guesses
        differentialCorrectionResult = applyDifferentialCorrection( orbitType, initialStateVector, orbitalPeriod, massParameter, maxPositionDeviationFromPeriodicOrbit, maxVelocityDeviationFromPeriodicOrbit);
        initialStateVector           = differentialCorrectionResult.segment(0,6);
        orbitalPeriod                = differentialCorrectionResult(6);

        // Propagate the initialStateVector for a full period and write output to file.
        stateVectorInclSTM = writePeriodicOrbitToFile( initialStateVector, lagrangePointNr, orbitType, numberOfInitialConditions, orbitalPeriod, massParameter);

        // Check whether continuation procedure has already crossed the position of the second primary
        if (lagrangePointNr == 1){
            if (initialStateVector(0) < (1.0 - massParameter)){

                continueNumericalContinuation = true;

                if (orbitType == "horizontal") {
                    if (initialStateVector(0) > 0.95 * (1.0 - massParameter)) {
                        continueNumericalContinuation = false;
                    }
                }if (orbitType == "vertical") {
                if (initialStateVector(0) > 0.94 * (1.0 - massParameter)) {
                    continueNumericalContinuation = false;
                    }
                }if (orbitType == "halo") {
                    if (initialStateVector(0) > 0.99 * (1.0 - massParameter)) {
                        continueNumericalContinuation = false;
                    }
                }
            }
        } else if (lagrangePointNr == 2){
            if (initialStateVector(0) > (1.0 - massParameter)){
                continueNumericalContinuation = true;

                if (orbitType == "horizontal"){
                    if (initialStateVector(0) < 1.02*(1.0 - massParameter)){
                        continueNumericalContinuation = false;
                    }
                }
            }
        }

        // Save jacobi energy, orbital period, initial condition, and eigenvalues
        tempStateVector.clear();

        // Add Jacobi energy and orbital period
        jacobiEnergy = tudat::gravitation::circular_restricted_three_body_problem::computeJacobiEnergy(massParameter, initialStateVector);
        tempStateVector.push_back(jacobiEnergy);
        tempStateVector.push_back(orbitalPeriod);

        // Add initial condition of periodic solution
        for (int i = 0; i <= 5; i++){
            tempStateVector.push_back(initialStateVector(i));
        }

        // Add Monodromy matrix
        for (int i = 6; i <= 41; i++){
            tempStateVector.push_back(stateVectorInclSTM(i));
        }

        // Add eigenvalues
//        eigenvalues = computeEigenvalues(stateVectorInclSTM);

        initialStateVectors.push_back(tempStateVector);
        numberOfInitialConditions += 1;
    }

    // Prepare file for initial conditions
    remove(("../data/raw/" + orbitType + "_L" + to_string(lagrangePointNr) + "_initial_conditions.txt").c_str());
    ofstream textFileInitialConditions(("../data/raw/" + orbitType + "_L" + to_string(lagrangePointNr) + "_initial_conditions.txt").c_str());
    textFileInitialConditions.precision(std::numeric_limits<double>::digits10);

    // Write initial conditions to file
    for (unsigned int i=0; i<initialStateVectors.size(); i++) {
        cout << "row: " << i << endl;
        textFileInitialConditions << left << scientific         << setw(25) << i << setw(25)
                                  << initialStateVectors[i][0]  << setw(25) << initialStateVectors[i][1]  << setw(25)
                                  << initialStateVectors[i][2]  << setw(25) << initialStateVectors[i][3]  << setw(25)
                                  << initialStateVectors[i][4]  << setw(25) << initialStateVectors[i][5]  << setw(25)
                                  << initialStateVectors[i][6]  << setw(25) << initialStateVectors[i][7]  << setw(25)
                                  << initialStateVectors[i][8]  << setw(25) << initialStateVectors[i][9]  << setw(25)
                                  << initialStateVectors[i][10] << setw(25) << initialStateVectors[i][11] << setw(25)
                                  << initialStateVectors[i][12] << setw(25) << initialStateVectors[i][13] << setw(25)
                                  << initialStateVectors[i][14] << setw(25) << initialStateVectors[i][15] << setw(25)
                                  << initialStateVectors[i][16] << setw(25) << initialStateVectors[i][17] << setw(25)
                                  << initialStateVectors[i][18] << setw(25) << initialStateVectors[i][19] << setw(25)
                                  << initialStateVectors[i][20] << setw(25) << initialStateVectors[i][21] << setw(25)
                                  << initialStateVectors[i][22] << setw(25) << initialStateVectors[i][23] << setw(25)
                                  << initialStateVectors[i][24] << setw(25) << initialStateVectors[i][25] << setw(25)
                                  << initialStateVectors[i][26] << setw(25) << initialStateVectors[i][27] << setw(25)
                                  << initialStateVectors[i][28] << setw(25) << initialStateVectors[i][29] << setw(25)
                                  << initialStateVectors[i][30] << setw(25) << initialStateVectors[i][31] << setw(25)
                                  << initialStateVectors[i][32] << setw(25) << initialStateVectors[i][33] << setw(25)
                                  << initialStateVectors[i][34] << setw(25) << initialStateVectors[i][35] << setw(25)
                                  << initialStateVectors[i][36] << setw(25) << initialStateVectors[i][37] << setw(25)
                                  << initialStateVectors[i][38] << setw(25) << initialStateVectors[i][39] << setw(25)
                                  << initialStateVectors[i][40] << setw(25) << initialStateVectors[i][41] << setw(25)
                                  << initialStateVectors[i][42] << setw(25) << initialStateVectors[i][43] << endl;
    }
}
