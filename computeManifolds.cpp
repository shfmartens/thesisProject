#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>
#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <math.h>

#include "Tudat/Astrodynamics/Gravitation/librationPoint.h"
#include "thesisProject/propagateOrbit.h"
#include "thesisProject/computeDifferentialCorrectionHalo.h"
#include "thesisProject/computeDifferentialCorrectionNearVertical.h"

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>


// Declare mass parameter.
Eigen::Vector3d thrustVector;
double massParameter;
namespace crtbp = tudat::gravitation::circular_restricted_three_body_problem;
//double thrustAcceleration = 0.0236087689713322;
double thrustAcceleration = 0.0;

void computeManifolds( string orbit_type, string selected_orbit, Eigen::VectorXd initialStateVector,
                       const double primaryGravitationalParameter = tudat::celestial_body_constants::EARTH_GRAVITATIONAL_PARAMETER,
                       const double secondaryGravitationalParameter = tudat::celestial_body_constants::MOON_GRAVITATIONAL_PARAMETER,
                       double displacementFromOrbit = 1.0e-5, double maxDeviationFromPeriodicOrbit = 1.0e-8,
                       double integrationStopTimeManifoldOrbits = 10.0, int numberOfManifoldOrbits = 100,
                       int saveEveryNthIntegrationStep = 100)
{
    // Set output precision and clear screen.
    std::cout.precision( 14 );

    // Define massParameter, initialStateVector and halfPeriodStateVector.
    massParameter = crtbp::computeMassParameter( primaryGravitationalParameter, secondaryGravitationalParameter );

    Eigen::VectorXd initialStateVectorInclSTM = Eigen::VectorXd::Zero(42);
    initialStateVectorInclSTM.segment(0,6) = initialStateVector;
    Eigen::MatrixXd identityMatrix = Eigen::MatrixXd::Identity(6, 6);
    identityMatrix.resize(36, 1);
    initialStateVectorInclSTM.segment(6,36) = identityMatrix;

    Eigen::VectorXd halfPeriodState = propagateOrbit( initialStateVectorInclSTM, massParameter, 0.5, 1.0, orbit_type);
    Eigen::VectorXd differentialCorrection( 6 );
    Eigen::VectorXd outputVector( 43 );
    double deviationFromPeriodicOrbit = 1.0;
    double orbitalPeriod = 0.0;
    cout << "\nInitial state vector:" << endl << initialStateVectorInclSTM.segment(0,6) << endl << "\nDifferential correction:" << endl;


    /** Differential Correction. */

    if (orbit_type == "halo"){
        // Apply differential correction and propagate to half-period point until converged.
        while (deviationFromPeriodicOrbit > maxDeviationFromPeriodicOrbit ) {

            // Apply differential correction.
            differentialCorrection = computeDifferentialCorrectionHalo( halfPeriodState );
            initialStateVectorInclSTM( 0 ) = initialStateVectorInclSTM( 0 ) + differentialCorrection( 0 )/1.0;
            initialStateVectorInclSTM( 2 ) = initialStateVectorInclSTM( 2 ) + differentialCorrection( 2 )/1.0;
            initialStateVectorInclSTM( 4 ) = initialStateVectorInclSTM( 4 ) + differentialCorrection( 4 )/1.0;

            // Propagate new state forward to half-period point.
            outputVector = propagateOrbit( initialStateVectorInclSTM, massParameter, 0.5, 1.0, orbit_type);
            halfPeriodState = outputVector.segment( 0, 42 );
            orbitalPeriod = 2.0 * outputVector( 42 );

            // Calculate deviation from periodic orbit.
            deviationFromPeriodicOrbit = fabs( halfPeriodState( 3 ) ) + fabs( halfPeriodState( 5  ) );
            cout << deviationFromPeriodicOrbit << endl;
        }
    }

    if(orbit_type == "near_vertical"){
        // Apply differential correction and propagate to half-period point until converged.
        while (deviationFromPeriodicOrbit > maxDeviationFromPeriodicOrbit ) {

            // Apply differential correction.
            differentialCorrection = computeDifferentialCorrectionNearVertical( halfPeriodState );
            initialStateVectorInclSTM( 0 ) = initialStateVectorInclSTM( 0 ) + differentialCorrection( 0 )/1.0;
            initialStateVectorInclSTM( 4 ) = initialStateVectorInclSTM( 4 ) + differentialCorrection( 4 )/1.0;
            initialStateVectorInclSTM( 5 ) = initialStateVectorInclSTM( 5 ) + differentialCorrection( 5 )/1.0;

            // Propagate new state forward to half-period point.
            outputVector = propagateOrbit( initialStateVectorInclSTM, massParameter, 0.5, 1.0, orbit_type);
            halfPeriodState = outputVector.segment( 0, 42 );
            orbitalPeriod = 2.0 * outputVector( 42 );

            // Calculate deviation from periodic orbit.
            deviationFromPeriodicOrbit = fabs( halfPeriodState( 3 ) );
            cout << deviationFromPeriodicOrbit << endl;
        }
    }

    // Write initial state to file
    double jacobiEnergy = tudat::gravitation::circular_restricted_three_body_problem::computeJacobiEnergy(massParameter, initialStateVectorInclSTM.segment(0,6));
    cout << "\nFinal initial state:" << endl << initialStateVectorInclSTM.segment(0,6) << endl << "\nwith C: " << jacobiEnergy << " and period: " << orbitalPeriod << endl;
    remove((selected_orbit + "_final_orbit.txt").c_str());
    ofstream textFile((selected_orbit + "_final_orbit.txt").c_str());
    textFile.precision(14);
    textFile << left << fixed << setw(20) << 0.0 << setw(20) << initialStateVectorInclSTM(0) << setw(20) << initialStateVectorInclSTM(1) << setw(20) << initialStateVectorInclSTM(2) << setw(20) << initialStateVectorInclSTM(3) << setw(20) << initialStateVectorInclSTM(4) << setw(20) << initialStateVectorInclSTM(5) << setw(20) << initialStateVectorInclSTM(6) << setw(20) << initialStateVectorInclSTM(7) << setw(20) << initialStateVectorInclSTM(8) << setw(20) << initialStateVectorInclSTM(9) << setw(20) << initialStateVectorInclSTM(10) << setw(20) << initialStateVectorInclSTM(11) << setw(20) << initialStateVectorInclSTM(12) << setw(20) << initialStateVectorInclSTM(13) << setw(20) << initialStateVectorInclSTM(14) << setw(20) << initialStateVectorInclSTM(15) << setw(20) << initialStateVectorInclSTM(16) << setw(20) << initialStateVectorInclSTM(17) << setw(20) <<  initialStateVectorInclSTM(18) << setw(20) << initialStateVectorInclSTM(19) << setw(20) << initialStateVectorInclSTM(20) << setw(20) << initialStateVectorInclSTM(21) << setw(20) << initialStateVectorInclSTM(22) << setw(20) << initialStateVectorInclSTM(23) << setw(20) << initialStateVectorInclSTM(24) << setw(20) << initialStateVectorInclSTM(25) << setw(20) << initialStateVectorInclSTM(26) << setw(20) << initialStateVectorInclSTM(27) << setw(20) << initialStateVectorInclSTM(28) << setw(20) << initialStateVectorInclSTM(29) << setw(20) << initialStateVectorInclSTM(30) << setw(20) << initialStateVectorInclSTM(31) << setw(20) << initialStateVectorInclSTM(32) << setw(20) << initialStateVectorInclSTM(33) << setw(20) << initialStateVectorInclSTM(34) << setw(20) << initialStateVectorInclSTM(35) << setw(20) << initialStateVectorInclSTM(36) << setw(20) << initialStateVectorInclSTM(37) << setw(20) << initialStateVectorInclSTM(38) << setw(20) << initialStateVectorInclSTM(39) << setw(20) << initialStateVectorInclSTM(40) << setw(20) << initialStateVectorInclSTM(41) << endl;


    // Propagate the initialStateVector for a full period and write output to file.
    outputVector = propagateOrbit( initialStateVectorInclSTM, massParameter, 0.0, 1.0, orbit_type);
    Eigen::VectorXd stateVectorInclSTM = outputVector.segment( 0, 42 );
    double currentTime = outputVector( 42 );
    Eigen::VectorXd instantV(3);
    instantV.setZero();

    while (currentTime <= orbitalPeriod) {

        stateVectorInclSTM = outputVector.segment( 0, 42 );
        currentTime = outputVector( 42 );

        // Write to file.
        textFile << left << fixed << setw(20) << currentTime << setw(20) << stateVectorInclSTM(0) << setw(20) << stateVectorInclSTM(1) << setw(20) << stateVectorInclSTM(2) << setw(20) << stateVectorInclSTM(3) << setw(20) << stateVectorInclSTM(4) << setw(20) << stateVectorInclSTM(5) << setw(20) << stateVectorInclSTM(6) << setw(20) << stateVectorInclSTM(7) << setw(20) << stateVectorInclSTM(8) << setw(20) << stateVectorInclSTM(9) << setw(20) << stateVectorInclSTM(10) << setw(20) << stateVectorInclSTM(11) << setw(20) << stateVectorInclSTM(12) << setw(20) << stateVectorInclSTM(13) << setw(20) << stateVectorInclSTM(14) << setw(20) << stateVectorInclSTM(15) << setw(20) << stateVectorInclSTM(16) << setw(20) << stateVectorInclSTM(17) << setw(20) <<  stateVectorInclSTM(18) << setw(20) << stateVectorInclSTM(19) << setw(20) << stateVectorInclSTM(20) << setw(20) << stateVectorInclSTM(21) << setw(20) << stateVectorInclSTM(22) << setw(20) << stateVectorInclSTM(23) << setw(20) << stateVectorInclSTM(24) << setw(20) << stateVectorInclSTM(25) << setw(20) << stateVectorInclSTM(26) << setw(20) << stateVectorInclSTM(27) << setw(20) << stateVectorInclSTM(28) << setw(20) << stateVectorInclSTM(29) << setw(20) << stateVectorInclSTM(30) << setw(20) << stateVectorInclSTM(31) << setw(20) << stateVectorInclSTM(32) << setw(20) << stateVectorInclSTM(33) << setw(20) << stateVectorInclSTM(34) << setw(20) << stateVectorInclSTM(35) << setw(20) << stateVectorInclSTM(36) << setw(20) << stateVectorInclSTM(37) << setw(20) << stateVectorInclSTM(38) << setw(20) << stateVectorInclSTM(39) << setw(20) << stateVectorInclSTM(40) << setw(20) << stateVectorInclSTM(41) << endl;

        // Propagate to next time step.
        outputVector = propagateOrbit( stateVectorInclSTM, massParameter, currentTime, 1.0, orbit_type);
    }

    /** Computation of Invariant Manifolds */
    // Read in the datapoints of the target orbit.
    std::ifstream fin((selected_orbit + "_final_orbit.txt").c_str());
    const int nCol = 43; // read from file
    std::vector< std::vector <double> > dataFromFile;  // your entire data-set of values

    std::vector<double> line(nCol, -1.0);  // create one line of nCol size and fill with -1
    bool done = false;
    while (!done)
    {
        for (int i = 0; !done && i < nCol; i++)
        {
            done = !(fin >> line[i]);
        }
        dataFromFile.push_back(line);
    }
    Eigen::MatrixXd matrixFromFile(dataFromFile.size(),42);
    Eigen::VectorXd timeVector(dataFromFile.size());
    for (int iRow = 0; iRow < dataFromFile.size(); iRow++)
    {
        timeVector(iRow) = dataFromFile[iRow][0];
        for (int iCol = 1; iCol < 43; iCol++)
        {
            matrixFromFile(iRow,iCol-1) = dataFromFile[iRow][iCol];
        }
    }

    // Select 50 points along the Halo to start the manifolds.
    int numberOfHaloPoints = timeVector.size();

    // Reshape the STM to matrix form.
    Eigen::MatrixXd STMEndOfPeriod(6,6);
    STMEndOfPeriod << matrixFromFile(numberOfHaloPoints-2,6), matrixFromFile(numberOfHaloPoints-2,12), matrixFromFile(numberOfHaloPoints-2,18), matrixFromFile(numberOfHaloPoints-2,24), matrixFromFile(numberOfHaloPoints-2,30), matrixFromFile(numberOfHaloPoints-2,36),
            matrixFromFile(numberOfHaloPoints-2,7), matrixFromFile(numberOfHaloPoints-2,13), matrixFromFile(numberOfHaloPoints-2,19), matrixFromFile(numberOfHaloPoints-2,25), matrixFromFile(numberOfHaloPoints-2,31), matrixFromFile(numberOfHaloPoints-2,37),
            matrixFromFile(numberOfHaloPoints-2,8), matrixFromFile(numberOfHaloPoints-2,14), matrixFromFile(numberOfHaloPoints-2,20), matrixFromFile(numberOfHaloPoints-2,26), matrixFromFile(numberOfHaloPoints-2,32), matrixFromFile(numberOfHaloPoints-2,38),
            matrixFromFile(numberOfHaloPoints-2,9), matrixFromFile(numberOfHaloPoints-2,15), matrixFromFile(numberOfHaloPoints-2,21), matrixFromFile(numberOfHaloPoints-2,27), matrixFromFile(numberOfHaloPoints-2,33), matrixFromFile(numberOfHaloPoints-2,39),
            matrixFromFile(numberOfHaloPoints-2,10), matrixFromFile(numberOfHaloPoints-2,16), matrixFromFile(numberOfHaloPoints-2,22), matrixFromFile(numberOfHaloPoints-2,28), matrixFromFile(numberOfHaloPoints-2,34), matrixFromFile(numberOfHaloPoints-2,40),
            matrixFromFile(numberOfHaloPoints-2,11), matrixFromFile(numberOfHaloPoints-2,17),  matrixFromFile(numberOfHaloPoints-2,23), matrixFromFile(numberOfHaloPoints-2,29), matrixFromFile(numberOfHaloPoints-2,35), matrixFromFile(numberOfHaloPoints-2,41);
    cout << "\nSTM:\n" << STMEndOfPeriod << endl << "\n" << endl;
    STMEndOfPeriod.transposeInPlace();
    cout << STMEndOfPeriod << endl;

    //Calculate the eigen vectors.
    Eigen::EigenSolver<Eigen::MatrixXd> eig(STMEndOfPeriod);
    Eigen::VectorXd eigenVector1 = eig.eigenvectors().real().col(0);
    Eigen::VectorXd eigenVector2 = eig.eigenvectors().real().col(1);
    cout << "\nEigenvector:\n" << eigenVector1 << endl << "\n" << endl;
    cout << "\nEigenvector:\n" << eigenVector2 << endl << "\n" << endl;

    // Apply displacement epsilon from the halo at <numberOfOrbits> locations on the halo.
    Eigen::VectorXd manifoldStartingState(42);
    manifoldStartingState.setZero();

    vector<double> offsetSigns = {1.0, -1.0, 1.0, -1.0};
    vector<Eigen::VectorXd> eigenVectors = {eigenVector2, eigenVector2, eigenVector1, eigenVector1};
    vector<double> integrationDirections = {1.0, 1.0, -1.0, -1.0};
    vector<string> fileNames = {selected_orbit + "_W_S_plus.txt", selected_orbit + "_W_S_min.txt",
                               selected_orbit + "_W_U_plus.txt", selected_orbit + "_W_U_min.txt"};

    double offsetSign;
    Eigen::VectorXd eigenVector;
    double integrationDirection;
    string fileName;

    for (int manifoldNumber = 0; manifoldNumber < 4; manifoldNumber++){

        offsetSign = offsetSigns.at(manifoldNumber);
        eigenVector = eigenVectors.at(manifoldNumber);
        integrationDirection = integrationDirections.at(manifoldNumber);
        fileName = fileNames.at(manifoldNumber);

        ofstream textFile2;
        remove(fileName.c_str());
        textFile2.open(fileName.c_str());
        textFile2.precision(14);

        cout << "\n\nManifold: " << fileName << "\n" << endl;

        for (int ii = 0; ii <numberOfManifoldOrbits; ii++) {
            manifoldStartingState.segment(0, 6) =
                    matrixFromFile.block(floor(ii * numberOfHaloPoints / numberOfManifoldOrbits), 0, 1, 6).transpose() +
                            offsetSign * displacementFromOrbit * eigenVector;
            textFile2 << left << fixed << setw(20) << 0.0 << setw(20) << manifoldStartingState(0) << setw(20)
                      << manifoldStartingState(1) << setw(20) << manifoldStartingState(2) << setw(20)
                      << manifoldStartingState(3) << setw(20) << manifoldStartingState(4) << setw(20)
                      << manifoldStartingState(5) << endl;
            outputVector = propagateOrbit(manifoldStartingState, massParameter, 0.0, integrationDirection, orbit_type);
            stateVectorInclSTM = outputVector.segment(0, 42);
            currentTime = outputVector(42);
            cout << "Orbit No.: " << ii + 1 << endl;
            int count = 1;

            while ( fabs( currentTime ) <= integrationStopTimeManifoldOrbits) {

                stateVectorInclSTM = outputVector.segment(0, 42);
                currentTime = outputVector(42);

                if (count % saveEveryNthIntegrationStep == 0) {
                    // Write to file.
                    textFile2 << left << fixed << setw(20) << currentTime << setw(20) << stateVectorInclSTM(0) << setw(20)
                              << stateVectorInclSTM(1) << setw(20) << stateVectorInclSTM(2) << setw(20) << stateVectorInclSTM(3) << setw(20)
                              << stateVectorInclSTM(4) << setw(20) << stateVectorInclSTM(5) << endl;

                }

                // Propagate to next time step.
                outputVector = propagateOrbit(stateVectorInclSTM, massParameter, currentTime, integrationDirection, orbit_type);
                count += 1;
            }
        }

        textFile2.close();
        textFile2.clear();

    }

    cout << "Mass parameter: " << massParameter
         << ", C_0: " << jacobiEnergy
         << ", C_1: " << tudat::gravitation::circular_restricted_three_body_problem::computeJacobiEnergy(massParameter, stateVectorInclSTM.segment(0,6))
         << ", and T: " << orbitalPeriod << endl;
    return;

}
