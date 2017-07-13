#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>
#include <Eigen/Dense>

#include "thesisProject/src/propagateOrbit.h"
#include "thesisProject/src/computeDifferentialCorrectionHalo.h"
#include "thesisProject/src/applyDifferentialCorrection.h"
#include "thesisProject/src/writePeriodicOrbitToFile.h"


void createInitialConditions( int lagrangePointNr, string orbitType, double amplitudeOne, double amplitudeTwo,
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
    Eigen::VectorXd outputVector( 43 );
    Eigen::VectorXd differentialCorrectionResult;
    double orbitalPeriod;
    bool orbitIsPeriodic = true;

    // Define massParameter
    massParameter = tudat::gravitation::circular_restricted_three_body_problem::computeMassParameter( primaryGravitationalParameter, secondaryGravitationalParameter );

    // Define first state vector guess
    initialStateVector(0) = 1.15566536294209;
    initialStateVector(4) = 9.10547315691362e-5;

    // Correct state vector guesses
    differentialCorrectionResult = applyDifferentialCorrection( initialStateVector, orbitType, massParameter, maxPositionDeviationFromPeriodicOrbit, maxVelocityDeviationFromPeriodicOrbit);
    initialStateVector = differentialCorrectionResult.segment(0,6);
    orbitalPeriod = differentialCorrectionResult(6);

    // Propagate the initialStateVector for a full period and write output to file.
    stateVectorInclSTM = writePeriodicOrbitToFile( initialStateVector, lagrangePointNr, orbitType, 0, orbitalPeriod, massParameter);

    // Save initial condition of periodic solution for continuation procedure
    tempStateVector.clear();
    for (int i = 0; i <= 5; i++){
        tempStateVector.push_back(initialStateVector(i));
    }
    initialStateVectors.push_back(tempStateVector);


    // Define second state vector guess
    initialStateVector = Eigen::VectorXd::Zero(6);
    initialStateVector(0) = 1.15551415840187;
    initialStateVector(4) = 0.000910813946034306;

    // Correct state vector guesses
    differentialCorrectionResult = applyDifferentialCorrection( initialStateVector, orbitType, massParameter, maxPositionDeviationFromPeriodicOrbit, maxVelocityDeviationFromPeriodicOrbit);
    initialStateVector = differentialCorrectionResult.segment(0,6);
    orbitalPeriod = differentialCorrectionResult(6);

    // Propagate the initialStateVector for a full period and write output to file.
    stateVectorInclSTM = writePeriodicOrbitToFile( initialStateVector, lagrangePointNr, orbitType, 1, orbitalPeriod, massParameter);

    // Save initial condition of periodic solution for continuation procedure
    tempStateVector.clear();
    for (int i = 0; i <= 5; i++){
        tempStateVector.push_back(initialStateVector(i));
    }
    initialStateVectors.push_back(tempStateVector);


    // Set exit parameters of continuation procedure
    int numberOfInitialConditions = 2;
    int maximumNumberOfInitialConditions = 100;

    while (numberOfInitialConditions < maximumNumberOfInitialConditions and orbitIsPeriodic){

        orbitIsPeriodic = false;

        // Apply numerical continuation
        initialStateVector = Eigen::VectorXd::Zero(6);
        initialStateVector(0) = initialStateVectors[initialStateVectors.size() - 1][0] * 2 - initialStateVectors[initialStateVectors.size() - 2][0];
        initialStateVector(4) = initialStateVectors[initialStateVectors.size() - 1][4] * 2 - initialStateVectors[initialStateVectors.size() - 2][4];

        // Correct state vector guesses
        differentialCorrectionResult = applyDifferentialCorrection( initialStateVector, orbitType, massParameter, maxPositionDeviationFromPeriodicOrbit, maxVelocityDeviationFromPeriodicOrbit);
        initialStateVector = differentialCorrectionResult.segment(0,6);
        orbitalPeriod = differentialCorrectionResult(6);

        // Propagate the initialStateVector for a full period and write output to file.
        stateVectorInclSTM = writePeriodicOrbitToFile( initialStateVector, lagrangePointNr, orbitType, numberOfInitialConditions - 1, orbitalPeriod, massParameter);

        // Reshape the STM for one period to matrix form and compute the eigenvalues
        Eigen::Map<Eigen::MatrixXd> monodromyMatrix = Eigen::Map<Eigen::MatrixXd>(stateVectorInclSTM.segment(6,36).data(),6,6);
        Eigen::EigenSolver<Eigen::MatrixXd> eig(monodromyMatrix);

        // Check whether at leat one of the eigenvalues has a real part of 1 (eigenvalue condition for periodicicity)
        for (int i = 0; i <= 5; i++){
            if (abs(eig.eigenvalues().real()(i) - 1) < maxDeviationEigenvalue){
                orbitIsPeriodic = true;
            }
        }
        cout << "Eigenvalues:\n" << eig.eigenvalues() << endl;

        // Save initial condition of periodic solution for continuation procedure
        tempStateVector.clear();
        for (int i = 0; i <= 5; i++){
            tempStateVector.push_back(initialStateVector(i));
        }
        initialStateVectors.push_back(tempStateVector);

        numberOfInitialConditions += 1;
    }

    // Prepare file for initial conditions
    remove(("../data/raw/" + orbitType + "_L" + to_string(lagrangePointNr) + "_initial_conditions.txt").c_str());
    ofstream textFileInitialConditions(("../data/raw/" + orbitType + "_L" + to_string(lagrangePointNr) + "_initial_conditions.txt").c_str());
    textFileInitialConditions.precision(14);

    // Write initial conditions to file
    for (unsigned int i=0; i<initialStateVectors.size(); i++) {
        textFileInitialConditions << left << fixed << setw(20) << i << setw(20)
                                  << initialStateVectors[i][0] << setw(20) << initialStateVectors[i][1] << setw(20)
                                  << initialStateVectors[i][2] << setw(20) << initialStateVectors[i][3] << setw(20)
                                  << initialStateVectors[i][4] << setw(20) << initialStateVectors[i][5] << endl;
    }
}
