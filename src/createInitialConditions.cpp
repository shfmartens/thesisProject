#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>
#include <Eigen/Dense>

#include "thesisProject/src/applyDifferentialCorrection.h"
#include "thesisProject/src/computeDifferentialCorrectionHalo.h"
#include "thesisProject/src/computeEigenvalues.h"
#include "thesisProject/src/propagateOrbit.h"
#include "thesisProject/src/writePeriodicOrbitToFile.h"


void createInitialConditions( int lagrangePointNr, string orbitType,
                              const double primaryGravitationalParameter = tudat::celestial_body_constants::EARTH_GRAVITATIONAL_PARAMETER,
                              const double secondaryGravitationalParameter = tudat::celestial_body_constants::MOON_GRAVITATIONAL_PARAMETER,
                              double maxPositionDeviationFromPeriodicOrbit = 1.0e-11, double maxVelocityDeviationFromPeriodicOrbit = 1.0e-8,
                              double maxDeviationEigenvalue = 1.0e-2)
{
    cout << "\nCreate initial conditions:\n" << endl;

    // Set output precision and clear screen.
    std::cout.precision( 14 );

    // Initialize state vectors and orbital periods
    Eigen::VectorXd initialStateVector = Eigen::VectorXd::Zero(6);
    Eigen::VectorXd stateVectorInclSTM = Eigen::VectorXd::Zero(42);
    std::vector< std::vector <double> > initialStateVectors;
    std::vector<double> tempStateVector;
    std::vector<double> eigenvalues;
    Eigen::VectorXd outputVector( 43 );
    Eigen::VectorXd differentialCorrectionResult;
    double orbitalPeriod;
    double jacobiEnergy;
    bool continueNumericalContinuation = true;

    // Define massParameter
    massParameter = tudat::gravitation::circular_restricted_three_body_problem::computeMassParameter( primaryGravitationalParameter, secondaryGravitationalParameter );

    // TODO extend horizontal, compute halo and vertical
    // Define first state vector guess
    if (orbitType == "horizontal") {
        if (lagrangePointNr == 1) {
            initialStateVector(0) = 0.836764423217002;
            initialStateVector(4) = 0.00126332372252124;
        } else if (lagrangePointNr == 2) {
            initialStateVector(0) = 1.15566536294209;
            initialStateVector(4) = 9.10547315691362e-5;
        }
    } else if (orbitType == "vertical"){
        if (lagrangePointNr == 1){
//            // Az = 4e-2 refined
//            initialStateVector(0) = 0.83694313536666;
//            initialStateVector(1) = 0.0;
//            initialStateVector(2) = 0.0;
//            initialStateVector(3) = 0.0;
//            initialStateVector(4) = 5.0843231043579e-05;
//            initialStateVector(5) = -0.013190336393571;
//
//            // Az = 5e-2 refined
//            initialStateVector(0) = 0.83695888935352;
//            initialStateVector(1) = 0.0;
//            initialStateVector(2) = 0.0;
//            initialStateVector(3) = 0.0;
//            initialStateVector(4) = 7.9479699864488e-05;
//            initialStateVector(5) = -0.016490086143727;

            // Az = 1e-2 refined
            initialStateVector(0) = 0.83691689636866;
            initialStateVector(1) = 0.0;
            initialStateVector(2) = 0.0;
            initialStateVector(3) = 0.0;
            initialStateVector(4) = 3.1826569396683e-06;
            initialStateVector(5) = -0.0032972814398178;

        } else if (lagrangePointNr == 2){
            initialStateVector(0) = 1.15565000000000;
            initialStateVector(1) = 0.0;
            initialStateVector(2) = 0.0;
            initialStateVector(3) = 0.0;
            initialStateVector(4) = -0.00007220724803;
            initialStateVector(5) = -0.01108868026317;
        }
    } else if (orbitType == "halo") {
        if (lagrangePointNr == 1) {
            // Az = 1e-6
//            initialStateVector(0) = 0.823801999615813;
//            initialStateVector(2) = 1.60845877178863e-7;
//            initialStateVector(4) = 0.127156026577337;
            // Az = 1e-5
//            initialStateVector(0) = 0.823801999616003;
//            initialStateVector(2) = 1.6084587718275e-6;
//            initialStateVector(4) = 0.127156026623678;
            // Az = 1e-5 corrected
//            initialStateVector(0) = 0.823801999616003;
//            initialStateVector(2) = -7.0016686496699e-07;
//            initialStateVector(4) = 0.12206195114883;
            // Az = 1e-3 corrected
//            initialStateVector(0) = 0.823802001538204;
//            initialStateVector(2) = -0.000160845916447091;
//            initialStateVector(4) = 0.127156494661802;
        } else if (lagrangePointNr == 2) {
            // Az = 1e-6
//            initialStateVector(0) = 1.12167448468914;
//            initialStateVector(2) = 1.47007218383909e-7;
//            initialStateVector(4) = 0.174155643349827;

            // Az = 1e-5
//            initialStateVector(0) = 1.12167448467379;
//            initialStateVector(2) = 1.47007218379311e-6;
//            initialStateVector(4) = 0.174155643398055;

            //Az = 1e-5 corrected
            initialStateVector(0) = 1.12167448467379;
            initialStateVector(2) = 0.00000062290509;
            initialStateVector(4) = 0.16992023760362;

            // Az = 1e-4
//            initialStateVector(0) = 1.12167448313796;
//            initialStateVector(2) = -1.47007217919523e-5;
//            initialStateVector(4) = 0.174155648220847;

            // Az = 1e-4 corrected
//            initialStateVector(0) = 1.121674483138;
//            initialStateVector(2) = -6.2178260839001e-07;
//            initialStateVector(4) = 0.16992024490462;

            // Az = 1e-3 corrected
//            initialStateVector(0) = 1.1216743295551;
//            initialStateVector(2) = 6.2102531069154e-07;
//            initialStateVector(4) = 0.16992097503243;

        }
    }

    // Correct state vector guesses
    differentialCorrectionResult = applyDifferentialCorrection( initialStateVector, orbitType, massParameter, maxPositionDeviationFromPeriodicOrbit, maxVelocityDeviationFromPeriodicOrbit);
    initialStateVector = differentialCorrectionResult.segment(0,6);
    orbitalPeriod = differentialCorrectionResult(6);

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
//    for (int i = 0; i <= 11; i++){
//        tempStateVector.push_back(eigenvalues.at(i));
//    }

    initialStateVectors.push_back(tempStateVector);

    // Define second state vector guess
    initialStateVector = Eigen::VectorXd::Zero(6);
    if (orbitType == "horizontal") {
        if (lagrangePointNr == 1) {
            initialStateVector(0) = 0.836900057031702;
            initialStateVector(4) = 0.000126362888667622;
        } else if (lagrangePointNr == 2) {
            initialStateVector(0) = 1.15551415840187;
            initialStateVector(4) = 0.000910813946034306;
        }
    } else if (orbitType == "vertical"){
        if (lagrangePointNr == 1){
            // Az = 5e-2 refined
            initialStateVector(0) = 0.83695888935352;
            initialStateVector(1) = 0.0;
            initialStateVector(2) = 0.0;
            initialStateVector(3) = 0.0;
            initialStateVector(4) = 7.9479699864488e-05;
            initialStateVector(5) = -0.016490086143727;
        } else if (lagrangePointNr == 2){
            initialStateVector(0) = 1.15560000000000;
            initialStateVector(1) = 0.0;
            initialStateVector(2) = 0.0;
            initialStateVector(3) = 0.0;
            initialStateVector(4) = -0.00018466181747;
            initialStateVector(5) = -0.01772707367258;
        }
    } else if (orbitType == "halo") {
        if (lagrangePointNr == 1) {
            // Az = 1e-5
//            initialStateVector(0) = 0.823801999616003;
//            initialStateVector(2) = 1.6084587718275e-6;
//            initialStateVector(4) = 0.127156026623678;
            // Az = 1e-4
//            initialStateVector(0) = 0.823801999635034;
//            initialStateVector(2) = 1.60845877571507e-5;
//            initialStateVector(4) = 0.127156031257729;
            // Az = 1e-4 corrected
//            initialStateVector(0) = 0.823801999635034;
//            initialStateVector(2) = 7.0118840620795e-07;
//            initialStateVector(4) = 0.12206195095084;
            // Az = 1e-2
            initialStateVector(0) = 0.823802193066798;
            initialStateVector(2) = -0.00160849803277001;
            initialStateVector(4) = 0.127202824963379;

        } else if (lagrangePointNr == 2) {
            // Az = 1e-5
//            initialStateVector(0) = 1.12167448467379;
//            initialStateVector(2) = 1.47007218379311e-6;
//            initialStateVector(4) = 0.174155643398055;
            //Az = 1e-5 corrected
//            initialStateVector(0) = 1.12167448467379;
//            initialStateVector(2) = -0.00000062290509;
//            initialStateVector(4) = 0.16992023760362;
            // Az = 5e-4
//            initialStateVector(0) = 1.12167444590574;
//            initialStateVector(2) = -7.35036033865708e-5;
//            initialStateVector(4) = 0.174155765136978;
            // Az = 1e-4
//            initialStateVector(0) = 1.12167448313796;
//            initialStateVector(2) = -1.47007217919523e-5;
//            initialStateVector(4) = 0.174155648220847;
            // Az = 1e-4 corrected
            initialStateVector(0) = 1.121674483138;
            initialStateVector(2) = -6.2178260839001e-07;
            initialStateVector(4) = 0.16992024490462;
            // 10th orbit after continuation 1e-5 vs 1e-4
//            initialStateVector(0) = 1.12167446931589;
//            initialStateVector(2) = 0.00000061168030;
//            initialStateVector(4) = 0.16992031061364;
            // Az = 1e-3
//            initialStateVector(0) = 1.12167432955512;
//            initialStateVector(2) = 0.000147007171940746;
//            initialStateVector(4) = 0.174156130499591;
            // Az = 1e-4 corrected
//            initialStateVector(0) = 1.121674483138;
//            initialStateVector(2) = 6.2178260839001e-07;
//            initialStateVector(4) = 0.16992024490462;
            // Az = 1e-3 corrected
//            initialStateVector(0) = 1.1216743295551;
//            initialStateVector(2) = 6.2102531069154e-07;
//            initialStateVector(4) = 0.16992097503243;
        }
    }

    // Correct state vector guesses
    differentialCorrectionResult = applyDifferentialCorrection( initialStateVector, orbitType, massParameter, maxPositionDeviationFromPeriodicOrbit, maxVelocityDeviationFromPeriodicOrbit);
    initialStateVector = differentialCorrectionResult.segment(0,6);
    orbitalPeriod = differentialCorrectionResult(6);

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
//    for (int i = 0; i <= 11; i++){
//        tempStateVector.push_back(eigenvalues.at(i));
//    }

    initialStateVectors.push_back(tempStateVector);


    // Set exit parameters of continuation procedure
    int numberOfInitialConditions = 2;
    int maximumNumberOfInitialConditions = 1000;

    while (numberOfInitialConditions <= maximumNumberOfInitialConditions and continueNumericalContinuation){

        continueNumericalContinuation = false;

        // Apply numerical continuation
        initialStateVector = Eigen::VectorXd::Zero(6);
        initialStateVector(0) = initialStateVectors[initialStateVectors.size() - 1][0+2] * 2 - initialStateVectors[initialStateVectors.size() - 2][0+2];
        initialStateVector(1) = initialStateVectors[initialStateVectors.size() - 1][1+2] * 2 - initialStateVectors[initialStateVectors.size() - 2][1+2];
        initialStateVector(2) = initialStateVectors[initialStateVectors.size() - 1][2+2] * 2 - initialStateVectors[initialStateVectors.size() - 2][2+2];
        initialStateVector(3) = initialStateVectors[initialStateVectors.size() - 1][3+2] * 2 - initialStateVectors[initialStateVectors.size() - 2][3+2];
        initialStateVector(4) = initialStateVectors[initialStateVectors.size() - 1][4+2] * 2 - initialStateVectors[initialStateVectors.size() - 2][4+2];
        initialStateVector(5) = initialStateVectors[initialStateVectors.size() - 1][5+2] * 2 - initialStateVectors[initialStateVectors.size() - 2][5+2];

        // Correct state vector guesses
        differentialCorrectionResult = applyDifferentialCorrection( initialStateVector, orbitType, massParameter, maxPositionDeviationFromPeriodicOrbit, maxVelocityDeviationFromPeriodicOrbit);
        initialStateVector = differentialCorrectionResult.segment(0,6);
        orbitalPeriod = differentialCorrectionResult(6);

        // Propagate the initialStateVector for a full period and write output to file.
        stateVectorInclSTM = writePeriodicOrbitToFile( initialStateVector, lagrangePointNr, orbitType, numberOfInitialConditions, orbitalPeriod, massParameter);

        // Check whether continuation procedure has already crossed the position of the second primary
        if (lagrangePointNr == 1){
            if (initialStateVector(0) < (1.0 - massParameter)){
                continueNumericalContinuation = true;
            }
        } else if (lagrangePointNr == 2){
            if (initialStateVector(0) > (1.0 - massParameter)){
                continueNumericalContinuation = true;
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
//        for (int i = 0; i <= 11; i++){
//            tempStateVector.push_back(eigenvalues.at(i));
//        }

        initialStateVectors.push_back(tempStateVector);

        numberOfInitialConditions += 1;
    }

    // Prepare file for initial conditions
    remove(("../data/raw/" + orbitType + "_L" + to_string(lagrangePointNr) + "_initial_conditions.txt").c_str());
    ofstream textFileInitialConditions(("../data/raw/" + orbitType + "_L" + to_string(lagrangePointNr) + "_initial_conditions.txt").c_str());
    textFileInitialConditions.precision(14);

    // Write initial conditions to file
    for (unsigned int i=0; i<initialStateVectors.size(); i++) {
        textFileInitialConditions << left << fixed << setw(25) << i << setw(25)
                                  << initialStateVectors[i][0] << setw(25) << initialStateVectors[i][1] << setw(25)
                                  << initialStateVectors[i][2] << setw(25) << initialStateVectors[i][3] << setw(25)
                                  << initialStateVectors[i][4] << setw(25) << initialStateVectors[i][5] << setw(25)
                                  << initialStateVectors[i][6] << setw(25) << initialStateVectors[i][7] << setw(25)
                                  << initialStateVectors[i][8] << setw(25) << initialStateVectors[i][9] << setw(25)
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
