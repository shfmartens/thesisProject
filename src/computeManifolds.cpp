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
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include "Tudat/Astrodynamics/Gravitation/librationPoint.h"
#include "thesisProject/src/propagateOrbit.h"
#include "thesisProject/src/computeDifferentialCorrection.h"
#include "thesisProject/src/computeDifferentialCorrectionHalo.h"
#include "thesisProject/src/computeDifferentialCorrectionNearVertical.h"


// Declare mass parameter.
//Eigen::Vector3d thrustVector;
//double thrustAcceleration = 0.0;
//double massParameter;

void computeManifolds( string orbit_type, string selected_orbit, Eigen::VectorXd initialStateVector,
                       const double primaryGravitationalParameter = tudat::celestial_body_constants::EARTH_GRAVITATIONAL_PARAMETER,
                       const double secondaryGravitationalParameter = tudat::celestial_body_constants::MOON_GRAVITATIONAL_PARAMETER,
//                       const double primaryGravitationalParameter = tudat::celestial_body_constants::SUN_GRAVITATIONAL_PARAMETER,
//                       const double secondaryGravitationalParameter = tudat::celestial_body_constants::EARTH_GRAVITATIONAL_PARAMETER,
                       double maxPositionDeviationFromPeriodicOrbit = 1.0e-11, double maxVelocityDeviationFromPeriodicOrbit = 1.0e-8,
                       double displacementFromOrbit = 1.0e-6, int numberOfManifoldOrbits = 100, int saveEveryNthIntegrationStep = 100,
                       double maximumIntegrationTimeManifoldOrbits = 50.0)
{
    // Set output precision and clear screen.
    std::cout.precision( 14 );

    // Define massParameter, initialStateVector and halfPeriodStateVector.
    double massParameter = tudat::gravitation::circular_restricted_three_body_problem::computeMassParameter( primaryGravitationalParameter, secondaryGravitationalParameter );

    Eigen::VectorXd initialStateVectorInclSTM = Eigen::VectorXd::Zero(42);
    initialStateVectorInclSTM.segment(0,6) = initialStateVector;
    Eigen::MatrixXd identityMatrix = Eigen::MatrixXd::Identity(6, 6);
    identityMatrix.resize(36, 1);
    initialStateVectorInclSTM.segment(6,36) = identityMatrix;

    Eigen::VectorXd halfPeriodState = propagateOrbit( initialStateVectorInclSTM, massParameter, 0.5, 1.0, orbit_type);
//    Eigen::VectorXd differentialCorrection( 7 );
    Eigen::VectorXd differentialCorrection( 6 );
    Eigen::VectorXd outputVector( 43 );

    // TODO Propagate the initialStateVector until T/2
//    outputVector = propagateOrbit( initialStateVectorInclSTM, massParameter, 0.0, 1.0, orbit_type);
//    Eigen::VectorXd stateVectorInclSTM = outputVector.segment( 0, 42 );
//    double currentTime = outputVector( 42 );
//    double orbitalPeriod = 3.41047588320389;
//
//    while (currentTime <= orbitalPeriod/2) {
//        stateVectorInclSTM = outputVector.segment( 0, 42 );
//        currentTime = outputVector( 42 );
//        // Propagate to next time step.
//        outputVector = propagateOrbit(stateVectorInclSTM, massParameter, currentTime, 1.0, orbit_type);
//    }
//    Eigen::VectorXd halfPeriodState = outputVector.segment( 0, 42 );

    double positionDeviationFromPeriodicOrbit = halfPeriodState(1);
    double velocityDeviationFromPeriodicOrbit = sqrt(pow(halfPeriodState(3),2) + pow(halfPeriodState(5),2));
    double orbitalPeriod = 2.0 * halfPeriodState( 42 );
    cout << "\nInitial state vector:" << endl << initialStateVectorInclSTM.segment(0,6) << endl
         << "\nPosition deviation from periodic orbit: " << positionDeviationFromPeriodicOrbit << endl
         << "\nVelocity deviation from periodic orbit: " << velocityDeviationFromPeriodicOrbit << endl
         << "\nDifferential correction:" << endl;

//    cout << "half state: \n" << halfPeriodState.segment(0,6) << endl;
    //! Differential Correction
    if (orbit_type == "halo"){
        // Apply differential correction and propagate to half-period point until converged.
        while (positionDeviationFromPeriodicOrbit > maxPositionDeviationFromPeriodicOrbit or
               velocityDeviationFromPeriodicOrbit > maxVelocityDeviationFromPeriodicOrbit) {

            // Apply differential correction.
            differentialCorrection = computeDifferentialCorrectionHalo( halfPeriodState );
//            differentialCorrection = computeDifferentialCorrection( halfPeriodState );
            initialStateVectorInclSTM( 0 ) = initialStateVectorInclSTM( 0 ) + differentialCorrection( 0 )/1.0;
            initialStateVectorInclSTM( 2 ) = initialStateVectorInclSTM( 2 ) + differentialCorrection( 2 )/1.0;
            initialStateVectorInclSTM( 4 ) = initialStateVectorInclSTM( 4 ) + differentialCorrection( 4 )/1.0;
//            orbitalPeriod = orbitalPeriod + differentialCorrection( 6 )/1.0;
//            cout<<differentialCorrection<<endl;
            // Propagate new state forward to half-period point.
            outputVector = propagateOrbit( initialStateVectorInclSTM, massParameter, 0.5, 1.0, orbit_type);
//            outputVector = propagateOrbit( initialStateVectorInclSTM, massParameter, 0.0, 1.0, orbit_type);
//            currentTime = outputVector( 42 );
//            while (currentTime <= orbitalPeriod/2) {
//                stateVectorInclSTM = outputVector.segment( 0, 42 );
//                currentTime = outputVector( 42 );
//                outputVector = propagateOrbit(stateVectorInclSTM, massParameter, currentTime, 1.0, orbit_type);
//            }

            halfPeriodState = outputVector.segment( 0, 42 );
//            orbitalPeriod = 2.0 * outputVector( 42 );

//            cout << "\ninitial state:\n" << initialStateVectorInclSTM.segment(0,6) << endl;
//            cout << "diff: \n" << differentialCorrection << endl;
//            cout << "half state: \n" << halfPeriodState.segment(0,6) << endl;
//            cout << "\n" << endl;

            // Calculate deviation from periodic orbit.
//            deviationFromPeriodicOrbit = fabs( halfPeriodState( 3 ) ) + fabs( halfPeriodState( 5  ) );
            velocityDeviationFromPeriodicOrbit = sqrt(pow(halfPeriodState(3),2) + pow(halfPeriodState(5),2));
            positionDeviationFromPeriodicOrbit = halfPeriodState(1);
            cout << velocityDeviationFromPeriodicOrbit << endl;
//            cout << positionDeviationFromPeriodicOrbit << endl;
            cout << "\n" << endl;
        }
    }

    if(orbit_type == "near_vertical"){
        // Apply differential correction and propagate to half-period point until converged.
        while (velocityDeviationFromPeriodicOrbit > maxVelocityDeviationFromPeriodicOrbit ) {

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
            velocityDeviationFromPeriodicOrbit = sqrt(pow(halfPeriodState(3),2) + pow(halfPeriodState(5),2));
            cout << velocityDeviationFromPeriodicOrbit << endl;
        }
    }

    double jacobiEnergy = tudat::gravitation::circular_restricted_three_body_problem::computeJacobiEnergy(massParameter, initialStateVectorInclSTM.segment(0,6));
    cout << "\nFinal initial state:" << endl << initialStateVectorInclSTM.segment(0,6) << endl
         << "\nwith C: " << jacobiEnergy << ", T: " << orbitalPeriod << ", T/2: " << orbitalPeriod/2.0 << endl;;

    // Write initial state to file
    remove(("../data/raw/" + selected_orbit + "_final_orbit.txt").c_str());
    ofstream textFile(("../data/raw/" + selected_orbit + "_final_orbit.txt").c_str());
    textFile.precision(14);
    textFile << left << fixed << setw(20) << 0.0 << setw(20)
             << initialStateVectorInclSTM(0) << setw(20) << initialStateVectorInclSTM(1) << setw(20)
             << initialStateVectorInclSTM(2) << setw(20) << initialStateVectorInclSTM(3) << setw(20)
             << initialStateVectorInclSTM(4) << setw(20) << initialStateVectorInclSTM(5)  << endl;

    std::vector< std::vector <double> > orbitStateVectors;
    std::vector<double> tempStateVector;
    int numberOfPointsOnPeriodicOrbit = 1;  // Initial state

    for (int i = 0; i <= 41; i++){
        tempStateVector.push_back(initialStateVectorInclSTM(i));
    }
    orbitStateVectors.push_back(tempStateVector);

    // Propagate the initialStateVector for a full period and write output to file.
    outputVector = propagateOrbit( initialStateVectorInclSTM, massParameter, 0.0, 1.0, orbit_type);
    Eigen::VectorXd stateVectorInclSTM = outputVector.segment( 0, 42 );
//    stateVectorInclSTM = outputVector.segment( 0, 42 );
    double currentTime = outputVector( 42 );
//    currentTime = outputVector( 42 );

    while (currentTime <= orbitalPeriod) {

        stateVectorInclSTM = outputVector.segment( 0, 42 );
        currentTime = outputVector( 42 );
        numberOfPointsOnPeriodicOrbit += 1;

        // Write to file.
        textFile << left << fixed << setw(20) << currentTime << setw(20)
                 << stateVectorInclSTM(0) << setw(20) << stateVectorInclSTM(1) << setw(20)
                 << stateVectorInclSTM(2) << setw(20) << stateVectorInclSTM(3) << setw(20)
                 << stateVectorInclSTM(4) << setw(20) << stateVectorInclSTM(5)  << endl;

        tempStateVector.clear();
        for (int i = 0; i <= 41; i++){
            tempStateVector.push_back(stateVectorInclSTM(i));
        }
        orbitStateVectors.push_back(tempStateVector);

        // Propagate to next time step.
        outputVector = propagateOrbit(stateVectorInclSTM, massParameter, currentTime, 1.0, orbit_type);
    }
    cout << "number of points on orbit: " << numberOfPointsOnPeriodicOrbit << endl;

    Eigen::MatrixXd orbitStateVectorsMatrix(orbitStateVectors.size(), 42);
    for (unsigned int iRow = 0; iRow < orbitStateVectors.size(); iRow++)
    {
        for (int iCol = 0; iCol <= 41; iCol++)
        {
            orbitStateVectorsMatrix(iRow,iCol) = orbitStateVectors[iRow][iCol];
        }
    }

    //! Computation of Invariant Manifolds
    // Reshape the STM for one period to matrix form.
    Eigen::Map<Eigen::MatrixXd> monodromyMatrix = Eigen::Map<Eigen::MatrixXd>(stateVectorInclSTM.segment(6,36).data(),6,6);
    cout << "\nMonodromy matrix:\n" << monodromyMatrix << "\n" << endl;

    // Compute eigenvectors of the monodromy matrix (find minimum eigenvalue, corresponding to stable, and large for unstable)
    Eigen::EigenSolver<Eigen::MatrixXd> eig(monodromyMatrix);
    cout << "Eigenvectors:\n" << eig.eigenvectors() << "\n\n" << "Eigenvalues:\n" << eig.eigenvalues() << "\n" << endl;

    int indexMaximumEigenvalue;
    double maximumEigenvalue = 0.0;
    int indexMinimumEigenvalue;
    double minimumEigenvalue = 1000.0;

    for (int i = 0; i <= 5; i++){
        if (eig.eigenvalues().real()(i) > maximumEigenvalue and abs(eig.eigenvalues().imag()(i)) < 1e-8){
            maximumEigenvalue = eig.eigenvalues().real()(i);
            indexMaximumEigenvalue = i;
        }
        if (abs(eig.eigenvalues().real()(i)) < minimumEigenvalue and abs(eig.eigenvalues().imag()(i)) < 1e-8){
            minimumEigenvalue = abs(eig.eigenvalues().real()(i));
            indexMinimumEigenvalue = i;
        }
    }

    Eigen::VectorXd eigenVector1 = eig.eigenvectors().real().col(indexMaximumEigenvalue);
    Eigen::VectorXd eigenVector2 = eig.eigenvectors().real().col(indexMinimumEigenvalue);
    cout << "Maximum real eigenvalue of " << maximumEigenvalue << " at " << indexMaximumEigenvalue
         << ", corresponding to eigenvector (unstable manifold): \n" << eigenVector1 << "\n\n"
         << "Minimum absolute real eigenvalue: " << minimumEigenvalue << " at " << indexMinimumEigenvalue
         << ", corresponding to eigenvector (stable manifold): \n" << eigenVector2 << endl;

    // Check whether the two selected eigenvalues belong to the same reciprocal pair
    if ((1.0 / minimumEigenvalue - maximumEigenvalue) > 1e-4){
        cout << "\n\n\nERROR - EIGENVALUES MIGHT NOT BELONG TO SAME RECIPROCAL PAIR" << endl;
        std::ofstream errorFile;
        errorFile.open("../data/raw/error_file.txt", std::ios_base::app);
        errorFile << selected_orbit << "\n\n"
                  << "Eigenvectors:\n" << eig.eigenvectors() << "\n"
                  << "Eigenvalues:\n" << eig.eigenvalues() << "\n"
                  << "Maximum real eigenvalue of " << maximumEigenvalue << " at " << indexMaximumEigenvalue
                  << ", corresponding to eigenvector (unstable manifold): \n" << eigenVector1 << "\n\n"
                  << "Minimum absolute real eigenvalue: " << minimumEigenvalue << " at " << indexMinimumEigenvalue
                  << ", corresponding to eigenvector (stable manifold): \n" << eigenVector2 << "\n\n" << endl;
        errorFile.close();
    }

    Eigen::VectorXd manifoldStartingState(42);
    manifoldStartingState.setZero();

    vector<double> offsetSigns = {1.0, -1.0, 1.0, -1.0};
    vector<Eigen::VectorXd> eigenVectors = {eigenVector2, eigenVector2, eigenVector1, eigenVector1};
    vector<double> integrationDirections = {-1.0, -1.0, 1.0, 1.0};
    vector<string> fileNames = {selected_orbit + "_W_S_plus.txt", selected_orbit + "_W_S_min.txt",
                                selected_orbit + "_W_U_plus.txt", selected_orbit + "_W_U_min.txt"};

//    boost::property_tree::ptree jsontree;
//    boost::property_tree::read_json("../config/config.json", jsontree);
//
//    jsontree.put( orbit_type + "." + selected_orbit + ".x",      initialStateVectorInclSTM( 0 ) );
//    jsontree.put( orbit_type + "." + selected_orbit + ".y",      initialStateVectorInclSTM( 1 ) );
//    jsontree.put( orbit_type + "." + selected_orbit + ".z",      initialStateVectorInclSTM( 2 ) );
//    jsontree.put( orbit_type + "." + selected_orbit + ".x_dot",  initialStateVectorInclSTM( 3 ) );
//    jsontree.put( orbit_type + "." + selected_orbit + ".y_dot",  initialStateVectorInclSTM( 4 ) );
//    jsontree.put( orbit_type + "." + selected_orbit + ".z_dot",  initialStateVectorInclSTM( 5 ) );
//    jsontree.put( orbit_type + "." + selected_orbit + ".C",      jacobiEnergy);
//    jsontree.put( orbit_type + "." + selected_orbit + ".T",      orbitalPeriod);
//    jsontree.put( orbit_type + "." + selected_orbit + ".l_1_re", eig.eigenvalues( ).row( 0 ).real( ) );
//    jsontree.put( orbit_type + "." + selected_orbit + ".l_1_im", eig.eigenvalues( ).row( 0 ).imag( ) );
//    jsontree.put( orbit_type + "." + selected_orbit + ".l_2_re", eig.eigenvalues( ).row( 1 ).real( ) );
//    jsontree.put( orbit_type + "." + selected_orbit + ".l_2_im", eig.eigenvalues( ).row( 1 ).imag( ) );
//    jsontree.put( orbit_type + "." + selected_orbit + ".l_3_re", eig.eigenvalues( ).row( 2 ).real( ) );
//    jsontree.put( orbit_type + "." + selected_orbit + ".l_3_im", eig.eigenvalues( ).row( 2 ).imag( ) );
//    jsontree.put( orbit_type + "." + selected_orbit + ".l_4_re", eig.eigenvalues( ).row( 3 ).real( ) );
//    jsontree.put( orbit_type + "." + selected_orbit + ".l_4_im", eig.eigenvalues( ).row( 3 ).imag( ) );
//    jsontree.put( orbit_type + "." + selected_orbit + ".l_5_re", eig.eigenvalues( ).row( 4 ).real( ) );
//    jsontree.put( orbit_type + "." + selected_orbit + ".l_5_im", eig.eigenvalues( ).row( 4 ).imag( ) );
//    jsontree.put( orbit_type + "." + selected_orbit + ".l_6_re", eig.eigenvalues( ).row( 5 ).real( ) );
//    jsontree.put( orbit_type + "." + selected_orbit + ".l_6_im", eig.eigenvalues( ).row( 5 ).imag( ) );
//
//    write_json("../config/config.json", jsontree);

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
        remove(("../data/raw/" + fileName).c_str());
        textFile2.open(("../data/raw/" + fileName).c_str());
        textFile2.precision(14);

        bool fullManifoldComputed = false;
        bool ySignSet = false;
        double ySign = 0.0;

        cout << "\n\nManifold: " << fileName << "\n" << endl;
        // Determine the total number of points along the periodic orbit to start the manifolds.
        for (int ii = 0; ii <numberOfManifoldOrbits; ii++) {

            int row_index = floor(ii * numberOfPointsOnPeriodicOrbit / numberOfManifoldOrbits);
            // Reshape the STM from vector to a matrix
            Eigen::VectorXd STMvector = orbitStateVectorsMatrix.row(row_index).segment(6,36);
            Eigen::Map<Eigen::MatrixXd> STM = Eigen::Map<Eigen::MatrixXd>(STMvector.data(),6,6);
//            Eigen::MatrixXd STM (6, 6);
//            STM <<  orbitStateVectorsMatrix(row_index, 6),  orbitStateVectorsMatrix(row_index, 12), orbitStateVectorsMatrix(row_index, 18), orbitStateVectorsMatrix(row_index, 24), orbitStateVectorsMatrix(row_index, 30), orbitStateVectorsMatrix(row_index, 36),
//                    orbitStateVectorsMatrix(row_index, 7),  orbitStateVectorsMatrix(row_index, 13), orbitStateVectorsMatrix(row_index, 19), orbitStateVectorsMatrix(row_index, 25), orbitStateVectorsMatrix(row_index, 31), orbitStateVectorsMatrix(row_index, 37),
//                    orbitStateVectorsMatrix(row_index, 8),  orbitStateVectorsMatrix(row_index, 14), orbitStateVectorsMatrix(row_index, 20), orbitStateVectorsMatrix(row_index, 26), orbitStateVectorsMatrix(row_index, 32), orbitStateVectorsMatrix(row_index, 38),
//                    orbitStateVectorsMatrix(row_index, 9),  orbitStateVectorsMatrix(row_index, 15), orbitStateVectorsMatrix(row_index, 21), orbitStateVectorsMatrix(row_index, 27), orbitStateVectorsMatrix(row_index, 33), orbitStateVectorsMatrix(row_index, 39),
//                    orbitStateVectorsMatrix(row_index, 10), orbitStateVectorsMatrix(row_index, 16), orbitStateVectorsMatrix(row_index, 22), orbitStateVectorsMatrix(row_index, 28), orbitStateVectorsMatrix(row_index, 34), orbitStateVectorsMatrix(row_index, 40),
//                    orbitStateVectorsMatrix(row_index, 11), orbitStateVectorsMatrix(row_index, 17), orbitStateVectorsMatrix(row_index, 23), orbitStateVectorsMatrix(row_index, 29), orbitStateVectorsMatrix(row_index, 35), orbitStateVectorsMatrix(row_index, 41);
//            cout << "\nSTM:\n" << STM << "\n" << endl;
//            cout << "\neigenvector:\n" << eigenVector << "\n" << endl;
//            cout << "\nSTM*eigenvector:\n" << STM*eigenVector << "\n" << endl;
//            cout << "\nnorm(STM*eigenvector):\n" << (STM*eigenVector).normalized() << "\n" << endl;
//            textFile2 << left << fixed << setw(20)
//                      << (STM*eigenVector).normalized()(0) << setw(20) << (STM*eigenVector).normalized()(1) << setw(20)
//                      << (STM*eigenVector).normalized()(2) << setw(20) << (STM*eigenVector).normalized()(3) << setw(20)
//                      << (STM*eigenVector).normalized()(4) << setw(20) << (STM*eigenVector).normalized()(5) << endl;

            // Apply displacement epsilon from the halo at <numberOfManifoldOrbits> locations on the final orbit.
            manifoldStartingState.segment(0, 6) = orbitStateVectorsMatrix.block(
                    floor(ii * numberOfPointsOnPeriodicOrbit / numberOfManifoldOrbits), 0, 1, 6).transpose() +
                            offsetSign * displacementFromOrbit * (STM*eigenVector).normalized();

            textFile2 << left << fixed << setw(20) << 0.0 << setw(20)
                      << manifoldStartingState(0) << setw(20) << manifoldStartingState(1) << setw(20)
                      << manifoldStartingState(2) << setw(20) << manifoldStartingState(3) << setw(20)
                      << manifoldStartingState(4) << setw(20) << manifoldStartingState(5) << endl;

            outputVector = propagateOrbit(manifoldStartingState, massParameter, 0.0, integrationDirection, orbit_type);
            stateVectorInclSTM = outputVector.segment(0, 42);
            currentTime = outputVector(42);
//            cout << "Orbit No.: " << ii + 1 << endl;
            int count = 1;

            while ( (fabs( currentTime ) <= maximumIntegrationTimeManifoldOrbits) and !fullManifoldComputed ) {

                // Determine sign of Y near x = 0
                if ( (stateVectorInclSTM(0) < 0) and !ySignSet ){
                    if ( stateVectorInclSTM(1) < 0 ){
                        ySign = -1.0;
                    }
                    if ( stateVectorInclSTM(1) > 0 ){
                        ySign = 1.0;
                    }
                    ySignSet = true;
                }

                // Determine when the manifold crosses the x-axis again
                if ( (stateVectorInclSTM(1) * ySign < 0) and ySignSet ){
                    fullManifoldComputed = true;
                }

                stateVectorInclSTM = outputVector.segment(0, 42);
                currentTime = outputVector(42);

                // Write every nth integration step to file.
                if (count % saveEveryNthIntegrationStep == 0 or fullManifoldComputed) {
                    textFile2 << left << fixed << setw(20) << currentTime << setw(20)
                              << stateVectorInclSTM(0) << setw(20) << stateVectorInclSTM(1) << setw(20)
                              << stateVectorInclSTM(2) << setw(20) << stateVectorInclSTM(3) << setw(20)
                              << stateVectorInclSTM(4) << setw(20) << stateVectorInclSTM(5) << endl;
                }

                // Propagate to next time step.
                outputVector = propagateOrbit(stateVectorInclSTM, massParameter, currentTime, integrationDirection, orbit_type);
                count += 1;
            }

            ySignSet = false;
            fullManifoldComputed = false;
        }
        textFile2.close();
        textFile2.clear();

    }

    std::cout << std::endl
              << "=================================================================="           << std::endl
              << "                          "   << selected_orbit << "                        " << std::endl
              << "Mass parameter: "             << massParameter                                << std::endl
              << "C at initial conditions: "    << jacobiEnergy                                 << std::endl
              << "C at end of manifold orbit: " << tudat::gravitation::circular_restricted_three_body_problem::computeJacobiEnergy(massParameter, stateVectorInclSTM.segment(0,6)) << std::endl
              << "T: " << orbitalPeriod                                                         << std::endl
              << "=================================================================="           << std::endl;
    return;

}
