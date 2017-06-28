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
#include "thesisProject/src/computeDifferentialCorrectionHalo.h"
#include "thesisProject/src/computeDifferentialCorrectionNearVertical.h"
//#include "thesisProject/src/writeToDatabase.h"


// Declare mass parameter.
Eigen::Vector3d thrustVector;
double massParameter;
namespace crtbp = tudat::gravitation::circular_restricted_three_body_problem;
double thrustAcceleration = 0.0;

void computeManifolds( string orbit_type, string selected_orbit, Eigen::VectorXd initialStateVector,
                       const double primaryGravitationalParameter = tudat::celestial_body_constants::EARTH_GRAVITATIONAL_PARAMETER,
                       const double secondaryGravitationalParameter = tudat::celestial_body_constants::MOON_GRAVITATIONAL_PARAMETER,
                       double displacementFromOrbit = 1.0e-6, double maxDeviationFromPeriodicOrbit = 1.0e-8,
                       int numberOfManifoldOrbits = 100, int saveEveryNthIntegrationStep = 100,
                       double maximumIntegrationTimeManifoldOrbits = 50.0)
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
    double deviationFromPeriodicOrbit = fabs( halfPeriodState( 3 ) );
    double orbitalPeriod = 2.0 * halfPeriodState( 42 );
    cout << "\nInitial state vector:" << endl << initialStateVectorInclSTM.segment(0,6) << endl
         << "\nDeviation from periodic orbit: " << deviationFromPeriodicOrbit << endl
         << "\nDifferential correction:" << endl;

    //! Differential Correction
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

    double jacobiEnergy = tudat::gravitation::circular_restricted_three_body_problem::computeJacobiEnergy(massParameter, initialStateVectorInclSTM.segment(0,6));
    cout << "\nFinal initial state:" << endl << initialStateVectorInclSTM.segment(0,6) << endl
         << "\nwith C: " << jacobiEnergy << ", T: " << orbitalPeriod << ", T/2: " << orbitalPeriod/2.0 << endl;;

    // Write initial state to file
    remove(("../data/raw/" + selected_orbit + "_final_orbit.txt").c_str());
    ofstream textFile(("../data/raw/" + selected_orbit + "_final_orbit.txt").c_str());
//    remove(("../src/verification/" + selected_orbit + "_l1.txt").c_str());
//    ofstream textFile(("../src/verification/" + selected_orbit + "_l1.txt").c_str());
    textFile.precision(14);
    textFile << left << fixed << setw(20) << 0.0 << setw(20)
             << initialStateVectorInclSTM(0) << setw(20) << initialStateVectorInclSTM(1) << setw(20)
             << initialStateVectorInclSTM(2) << setw(20) << initialStateVectorInclSTM(3) << setw(20)
             << initialStateVectorInclSTM(4) << setw(20) << initialStateVectorInclSTM(5)  << endl;

    std::vector< std::vector <double> > orbitStateVectors;
    std::vector<double> tempStateVector;
    int numberOfPointsOnPeriodicOrbit = 1;  // Initial state

    for (int i = 0; i <= 5; i++){
        tempStateVector.push_back(initialStateVectorInclSTM(i));
    }
    orbitStateVectors.push_back(tempStateVector);

    // Propagate the initialStateVector for a full period and write output to file.
    outputVector = propagateOrbit( initialStateVectorInclSTM, massParameter, 0.0, 1.0, orbit_type);
    Eigen::VectorXd stateVectorInclSTM = outputVector.segment( 0, 42 );
    double currentTime = outputVector( 42 );

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
        for (int i = 0; i <= 5; i++){
            tempStateVector.push_back(stateVectorInclSTM(i));
        }
        orbitStateVectors.push_back(tempStateVector);

        // Propagate to next time step.
        outputVector = propagateOrbit(stateVectorInclSTM, massParameter, currentTime, 1.0, orbit_type);
    }

    Eigen::MatrixXd orbitStateVectorsMatrix(orbitStateVectors.size(),6);
    for (unsigned int iRow = 0; iRow < orbitStateVectors.size(); iRow++)
    {
        for (int iCol = 0; iCol <= 5; iCol++)
        {
            orbitStateVectorsMatrix(iRow,iCol) = orbitStateVectors[iRow][iCol];
        }
    }

    //! Computation of Invariant Manifolds
    // Reshape the STM for one period to matrix form.
    Eigen::MatrixXd monodromyMatrix (6,6);
    monodromyMatrix <<  stateVectorInclSTM(6), stateVectorInclSTM(12), stateVectorInclSTM(18), stateVectorInclSTM(24), stateVectorInclSTM(30), stateVectorInclSTM(36),
                        stateVectorInclSTM(7), stateVectorInclSTM(13), stateVectorInclSTM(19), stateVectorInclSTM(25), stateVectorInclSTM(31), stateVectorInclSTM(37),
                        stateVectorInclSTM(8), stateVectorInclSTM(14), stateVectorInclSTM(20), stateVectorInclSTM(26), stateVectorInclSTM(32), stateVectorInclSTM(38),
                        stateVectorInclSTM(9), stateVectorInclSTM(15), stateVectorInclSTM(21), stateVectorInclSTM(27), stateVectorInclSTM(33), stateVectorInclSTM(39),
                        stateVectorInclSTM(10), stateVectorInclSTM(16), stateVectorInclSTM(22), stateVectorInclSTM(28), stateVectorInclSTM(34), stateVectorInclSTM(40),
                        stateVectorInclSTM(11), stateVectorInclSTM(17),  stateVectorInclSTM(23), stateVectorInclSTM(29), stateVectorInclSTM(35), stateVectorInclSTM(41);
    cout << "\nMonodromy matrix:\n" << monodromyMatrix << "\n" << endl;
    monodromyMatrix.transposeInPlace();

    // Compute eigenvectors of the monodromy matrix
    Eigen::EigenSolver<Eigen::MatrixXd> eig(monodromyMatrix);
    cout << "Eigenvectors:\n" << endl;
    cout << eig.eigenvectors() << "\n" << endl;

    cout << "Eigenvalues:\n" << endl;
    cout << eig.eigenvalues() << "\n" << endl;

    Eigen::VectorXd eigenVector1 = eig.eigenvectors().real().col(0);
    Eigen::VectorXd eigenVector2 = eig.eigenvectors().real().col(1);
    cout << "Eigenvectors:\n" << eigenVector1 << endl << "\n" << eigenVector2 << "\n" << endl;


    Eigen::VectorXd manifoldStartingState(42);
    manifoldStartingState.setZero();

    vector<double> offsetSigns = {1.0, -1.0, 1.0, -1.0};
    vector<Eigen::VectorXd> eigenVectors = {eigenVector2, eigenVector2, eigenVector1, eigenVector1};
    vector<double> integrationDirections = {-1.0, -1.0, 1.0, 1.0};
    vector<string> fileNames = {selected_orbit + "_W_S_plus.txt", selected_orbit + "_W_S_min.txt",
                                selected_orbit + "_W_U_plus.txt", selected_orbit + "_W_U_min.txt"};


//    // Write initial state and eigenvalues to file
//    ofstream initialConditionsFile(("../data/raw/initial_conditions.txt").c_str());
//    initialConditionsFile.precision(14);
//
//    initialConditionsFile << left << fixed << setw( 20 ) << 0.0 << setw( 20 )
//                          << initialStateVectorInclSTM( 0 ) << setw( 20 ) << initialStateVectorInclSTM( 1 ) << setw( 20 )
//                          << initialStateVectorInclSTM( 2 ) << setw( 20 ) << initialStateVectorInclSTM( 3 ) << setw( 20 )
//                          << initialStateVectorInclSTM( 4 ) << setw( 20 ) << initialStateVectorInclSTM( 5 )
//                          << eig.eigenvalues( ).row( 0 ).col( 0 ) << eig.eigenvalues( ).row( 0 ).col( 1 )
//                          << eig.eigenvalues( ).row( 1 ).col( 0 ) << eig.eigenvalues( ).row( 1 ).col( 1 )
//                          << eig.eigenvalues( ).row( 2 ).col( 0 ) << eig.eigenvalues( ).row( 2 ).col( 1 )
//                          << eig.eigenvalues( ).row( 3 ).col( 0 ) << eig.eigenvalues( ).row( 3 ).col( 1 )
//                          << eig.eigenvalues( ).row( 4 ).col( 0 ) << eig.eigenvalues( ).row( 4 ).col( 1 )
//                          << eig.eigenvalues( ).row( 5 ).col( 0 ) << eig.eigenvalues( ).row( 5 ).col( 1 ) << endl;
//
//    initialConditionsFile.close();

//    // Open database in read/write mode.
//    SQLite::Database database( input.databasePath.c_str( ),
//                               SQLITE_OPEN_READWRITE|SQLITE_OPEN_CREATE );
//
//    // Create table for Lambert scanner results in SQLite database.
//    std::cout << "Creating SQLite database table if needed ... " << std::endl;
//    createInitialConditionsTable( database );
//    std::cout << "SQLite database set up successfully!" << std::endl;
//
//    // Start SQL transaction.
//    SQLite::Transaction transaction( database );
//
//    // Setup insert query.
//    std::ostringstream createInitialConditionsTableInsert;
//    createInitialConditionsTableInsert
//            << "INSERT INTO initial_conditions VALUES ("
//            << ":orbit_type,"
//            << "NULL,"
//            << ":departure_object_id,"
//            << ":x,"
//            << ":y,"
//            << ":z,"
//            << ":x_dot,"
//            << ":y_dot,"
//            << ":z_dot,"
//            << ":lambda_1_r,"
//            << ":lambda_1_i,"
//            << ":lambda_2_r,"
//            << ":lambda_2_i,"
//            << ":lambda_3_r,"
//            << ":lambda_3_i,"
//            << ":lambda_4_r,"
//            << ":lambda_4_i,"
//            << ":lambda_5_r,"
//            << ":lambda_5_i,"
//            << ":lambda_6_r,"
//            << ":lambda_6_i,"
//            << ");";
//
//    SQLite::Statement query( database, createInitialConditionsTableInsert.str( ) );
//
//    {
//        query.bind( ":orbit_type",           orbit_type );
//        query.bind( ":x",                    initialStateVectorInclSTM( 0 ) );
//        query.bind( ":y",                    initialStateVectorInclSTM( 1 ) );
//        query.bind( ":z",                    initialStateVectorInclSTM( 2 ) );
//        query.bind( ":x_dot",                initialStateVectorInclSTM( 3 ) );
//        query.bind( ":y_dot",                initialStateVectorInclSTM( 4 ) );
//        query.bind( ":z_dot",                initialStateVectorInclSTM( 5 ) );
//        query.bind( ":lambda_1_r",           eig.eigenvalues( ).row( 0 ).col( 0 ) );
//        query.bind( ":lambda_1_i",           eig.eigenvalues( ).row( 0 ).col( 1 ) );
//        query.bind( ":lambda_2_r",           eig.eigenvalues( ).row( 0 ).col( 0 ) );
//        query.bind( ":lambda_2_i",           eig.eigenvalues( ).row( 0 ).col( 1 ) );
//        query.bind( ":lambda_3_r",           eig.eigenvalues( ).row( 0 ).col( 0 ) );
//        query.bind( ":lambda_3_i",           eig.eigenvalues( ).row( 0 ).col( 1 ) );
//        query.bind( ":lambda_4_r",           eig.eigenvalues( ).row( 0 ).col( 0 ) );
//        query.bind( ":lambda_4_i",           eig.eigenvalues( ).row( 0 ).col( 1 ) );
//        query.bind( ":lambda_5_r",           eig.eigenvalues( ).row( 0 ).col( 0 ) );
//        query.bind( ":lambda_5_i",           eig.eigenvalues( ).row( 0 ).col( 1 ) );
//        query.bind( ":lambda_6_r",           eig.eigenvalues( ).row( 0 ).col( 0 ) );
//        query.bind( ":lambda_6_i",           eig.eigenvalues( ).row( 0 ).col( 1 ) );
//
//        // Execute insert query.
//        query.executeStep( );
//
//        // Reset SQL insert query.
//        query.reset( );
//    }

//    boost::property_tree::ptree jsontree;
////    boost::property_tree::read_json("../src/verification/halo_verification_l1.json", jsontree);
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
////    write_json("../src/verification/halo_verification_l1.json", jsontree);
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

        cout << "\nManifold: " << fileName << "\n" << endl;
        // Determine the total number of points along the periodic orbit to start the manifolds.
        for (int ii = 0; ii <numberOfManifoldOrbits; ii++) {

            // Apply displacement epsilon from the halo at <numberOfManifoldOrbits> locations on the final orbit.
            manifoldStartingState.segment(0, 6) = orbitStateVectorsMatrix.block(
                    floor(ii * numberOfPointsOnPeriodicOrbit / numberOfManifoldOrbits), 0, 1, 6).transpose() +
                            offsetSign * displacementFromOrbit * eigenVector;

            textFile2 << left << fixed << setw(20) << 0.0 << setw(20)
                      << manifoldStartingState(0) << setw(20) << manifoldStartingState(1) << setw(20)
                      << manifoldStartingState(2) << setw(20) << manifoldStartingState(3) << setw(20)
                      << manifoldStartingState(4) << setw(20) << manifoldStartingState(5) << endl;

            outputVector = propagateOrbit(manifoldStartingState, massParameter, 0.0, integrationDirection, orbit_type);
            stateVectorInclSTM = outputVector.segment(0, 42);
            currentTime = outputVector(42);
            cout << "Orbit No.: " << ii + 1 << endl;
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

    cout << "\nMass parameter: " << massParameter << endl
         << "C at initial conditions: " << jacobiEnergy << endl
         << "C at end of manifold orbit: " << tudat::gravitation::circular_restricted_three_body_problem::computeJacobiEnergy(massParameter, stateVectorInclSTM.segment(0,6)) << endl
         << "T: " << orbitalPeriod << endl;
    return;

}
