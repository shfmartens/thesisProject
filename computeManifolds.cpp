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
#include "thesisProject/propagateHalo.h"
#include "thesisProject/createStateVector.h"
#include "thesisProject/computeDifferentialCorrection.h"

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>


// Declare mass parameter.
Eigen::Vector3d thrustVector;
double massParameter;
double thrustAcceleration = 0.0236087689713322;

namespace crtbp = tudat::gravitation::circular_restricted_three_body_problem;

void computeManifolds( string selected_orbit )
{

    // Set output precision and clear screen.
    std::cout.precision( 14 );

    // Load configuration parameters
    boost::property_tree::ptree jsontree;
    boost::property_tree::read_json("config.json", jsontree);

    double x_0 = jsontree.get<double>("initial_states.halo." + selected_orbit + ".x");
    double y_0 = jsontree.get<double>("initial_states.halo." + selected_orbit + ".y");
    double z_0 = jsontree.get<double>("initial_states.halo." + selected_orbit + ".z");
    double x_dot_0 = jsontree.get<double>("initial_states.halo." + selected_orbit + ".x_dot");
    double y_dot_0 = jsontree.get<double>("initial_states.halo." + selected_orbit + ".y_dot");
    double z_dot_0 = jsontree.get<double>("initial_states.halo." + selected_orbit + ".z_dot");
    double epsilon = jsontree.get<double>("manifold_parameters.epsilon");
    int numberOfOrbits = jsontree.get<int>("manifold_parameters.numberOfOrbits");

    // Define massParameter.
    const double earthGravitationalParameter = tudat::celestial_body_constants::EARTH_GRAVITATIONAL_PARAMETER;
    const double moonGravitationalParameter = tudat::celestial_body_constants::MOON_GRAVITATIONAL_PARAMETER;
    massParameter = crtbp::computeMassParameter( earthGravitationalParameter, moonGravitationalParameter ); // NB: also change in state derivative!;

    //Set-up initialStateVector and halfPeriodStateVector.
    Eigen::VectorXd state_vector_0 = Eigen::VectorXd::Zero(6);
    state_vector_0(0) = x_0;
    state_vector_0(1) = y_0;
    state_vector_0(2) = z_0;
    state_vector_0(3) = x_dot_0;
    state_vector_0(4) = y_dot_0;
    state_vector_0(5) = z_dot_0;
    double jacobiEnergy_0 = tudat::gravitation::circular_restricted_three_body_problem::computeJacobiEnergy(massParameter, state_vector_0);

    Eigen::VectorXd initialStateVector = createStateVector(x_0, z_0, massParameter, jacobiEnergy_0);
    Eigen::VectorXd halfPeriodState = propagateHalo( initialStateVector, massParameter, 0.5, 1.0 );
    Eigen::VectorXd differentialCorrection( 6 );
    Eigen::VectorXd outputVector( 43 );
    double deviationFromPerfectHalo = 1.0;
    double haloPeriod = 10.0;
    cout << "\nInitial state vector:" << endl << initialStateVector.segment(0,6) << endl << "\nDifferential correction:" << endl;

                // Apply differential correction and propagate to half-period point until converged.
                while (deviationFromPerfectHalo > 1.0e-8 ) {

                    // Apply differential correction.
                    differentialCorrection = computeDifferentialCorrection( halfPeriodState );
                    initialStateVector( 0 ) = initialStateVector( 0 ) + differentialCorrection( 0 )/1.0;
                    initialStateVector( 2 ) = initialStateVector( 2 ) + differentialCorrection( 2 )/1.0;
                    initialStateVector( 4 ) = initialStateVector( 4 ) + differentialCorrection( 4 )/1.0;

                    // Propagate new state forward to half-period point
                    outputVector = propagateHalo( initialStateVector, massParameter, 0.5, 1.0);
                    halfPeriodState = outputVector.segment( 0, 42 );
                    haloPeriod = 2.0 * outputVector( 42 );

                    // Calculate deviation from perfect scenario.
                    deviationFromPerfectHalo = fabs( halfPeriodState( 3 ) ) + fabs( halfPeriodState( 5  ) );
                    cout << deviationFromPerfectHalo << endl;
                }

    // Write initial state to file
    jacobiEnergy_0 = tudat::gravitation::circular_restricted_three_body_problem::computeJacobiEnergy(massParameter, initialStateVector.segment(0,6));
    jsontree.put("initial_states.halo." + selected_orbit + ".C", jacobiEnergy_0);
    jsontree.put("initial_states.halo." + selected_orbit + ".T", haloPeriod);
    write_json("config.json", jsontree);

    cout << "\nFinal initial state:" << endl << initialStateVector.segment(0,6) << endl << "\nwith C: " << jacobiEnergy_0 << " and period: " << haloPeriod << endl;
    remove((selected_orbit + "_final_orbit.txt").c_str());
    ofstream textFile((selected_orbit + "_final_orbit.txt").c_str());
    textFile.precision(14);
    textFile << left << fixed << setw(20) << 0.0 << setw(20) << initialStateVector(0) << setw(20) << initialStateVector(1) << setw(20) << initialStateVector(2) << setw(20) << initialStateVector(3) << setw(20) << initialStateVector(4) << setw(20) << initialStateVector(5) << setw(20) << initialStateVector(6) << setw(20) << initialStateVector(7) << setw(20) << initialStateVector(8) << setw(20) << initialStateVector(9) << setw(20) << initialStateVector(10) << setw(20) << initialStateVector(11) << setw(20) << initialStateVector(12) << setw(20) << initialStateVector(13) << setw(20) << initialStateVector(14) << setw(20) << initialStateVector(15) << setw(20) << initialStateVector(16) << setw(20) << initialStateVector(17) << setw(20) <<  initialStateVector(18) << setw(20) << initialStateVector(19) << setw(20) << initialStateVector(20) << setw(20) << initialStateVector(21) << setw(20) << initialStateVector(22) << setw(20) << initialStateVector(23) << setw(20) << initialStateVector(24) << setw(20) << initialStateVector(25) << setw(20) << initialStateVector(26) << setw(20) << initialStateVector(27) << setw(20) << initialStateVector(28) << setw(20) << initialStateVector(29) << setw(20) << initialStateVector(30) << setw(20) << initialStateVector(31) << setw(20) << initialStateVector(32) << setw(20) << initialStateVector(33) << setw(20) << initialStateVector(34) << setw(20) << initialStateVector(35) << setw(20) << initialStateVector(36) << setw(20) << initialStateVector(37) << setw(20) << initialStateVector(38) << setw(20) << initialStateVector(39) << setw(20) << initialStateVector(40) << setw(20) << initialStateVector(41) << endl;


    // Propagate the initialStateVector for a full period and write output to file.
    outputVector = propagateHalo( initialStateVector, massParameter, 0.0, 1.0);
    Eigen::VectorXd haloState = outputVector.segment( 0, 42 );
    double currentTime = outputVector( 42 );
//    double angle;
//    double VNorm;
    Eigen::VectorXd instantV(3);
    instantV.setZero();
//    double scalingFactor = 1e3*1.0;
//    double checkFlag = 0.0;

    while (currentTime <= haloPeriod) {

        haloState = outputVector.segment( 0, 42 );
        currentTime = outputVector( 42 );

        // Write to file.
        textFile << left << fixed << setw(20) << currentTime << setw(20) << haloState(0) << setw(20) << haloState(1) << setw(20) << haloState(2) << setw(20) << haloState(3) << setw(20) << haloState(4) << setw(20) << haloState(5) << setw(20) << haloState(6) << setw(20) << haloState(7) << setw(20) << haloState(8) << setw(20) << haloState(9) << setw(20) << haloState(10) << setw(20) << haloState(11) << setw(20) << haloState(12) << setw(20) << haloState(13) << setw(20) << haloState(14) << setw(20) << haloState(15) << setw(20) << haloState(16) << setw(20) << haloState(17) << setw(20) <<  haloState(18) << setw(20) << haloState(19) << setw(20) << haloState(20) << setw(20) << haloState(21) << setw(20) << haloState(22) << setw(20) << haloState(23) << setw(20) << haloState(24) << setw(20) << haloState(25) << setw(20) << haloState(26) << setw(20) << haloState(27) << setw(20) << haloState(28) << setw(20) << haloState(29) << setw(20) << haloState(30) << setw(20) << haloState(31) << setw(20) << haloState(32) << setw(20) << haloState(33) << setw(20) << haloState(34) << setw(20) << haloState(35) << setw(20) << haloState(36) << setw(20) << haloState(37) << setw(20) << haloState(38) << setw(20) << haloState(39) << setw(20) << haloState(40) << setw(20) << haloState(41) << endl;

        // Propagate to next time step.
        outputVector = propagateHalo( haloState, massParameter, currentTime, 1.0);
    }

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

                //Calculate the eigenvectors.
                Eigen::EigenSolver<Eigen::MatrixXd> eig(STMEndOfPeriod);
                Eigen::VectorXd eigenVector = eig.eigenvectors().real().col(1);
                cout << "\nEigenvector:\n" << eigenVector << endl << "\n" << endl;

                // Apply displacement epsilon from the halo at 50 location on the halo.
                Eigen::VectorXd manifoldStartingState(42);
                manifoldStartingState.setZero();
                remove((selected_orbit + "_manifold.txt").c_str());
                ofstream textFile2((selected_orbit + "_manifold.txt").c_str());
                textFile2.precision(14);
                Eigen::VectorXd tempTemp(42);
            //    Eigen::VectorXd outputVector( 43 );//LETOP!
            //    double currentTime = 0.0;  //LETOP!
            //    Eigen::VectorXd haloState = outputVector.segment( 0, 42 );//LETOP!

                for (int ii = 0; ii <numberOfOrbits; ii++) {
                    manifoldStartingState.segment(0,6) = matrixFromFile.block(floor(ii*numberOfHaloPoints/numberOfOrbits),0,1,6).transpose() - epsilon*eigenVector;
                    textFile2 << left << fixed << setw(20) << 0.0 << setw(20) << manifoldStartingState(0) << setw(20) << manifoldStartingState(1) << setw(20) << manifoldStartingState(2) << setw(20) << manifoldStartingState(3) << setw(20) << manifoldStartingState(4) << setw(20) << manifoldStartingState(5) << endl;
                    outputVector = propagateHalo( manifoldStartingState, massParameter, 0.0, -1.0); // LETOP!
                    haloState = outputVector.segment( 0, 42 );
                    currentTime = outputVector(42);
                    cout << "Manifold No.: " << ii+1 << endl;
                    while (currentTime >= -5) {//LETOP!

                        haloState = outputVector.segment( 0, 42 );
                        currentTime = outputVector( 42 );

                        // Write to file.
                        textFile2 << left << fixed << setw(20) << currentTime << setw(20) << haloState(0) << setw(20) << haloState(1) << setw(20) << haloState(2) << setw(20) << haloState(3) << setw(20) << haloState(4) << setw(20) << haloState(5) << endl;

                        // Propagate to next time step.
                        outputVector = propagateHalo( haloState, massParameter, currentTime, -1.0); //LETOP!
                    }

                }
    cout << "Mass parameter: " << massParameter <<  " and C: " << tudat::gravitation::circular_restricted_three_body_problem::computeJacobiEnergy(massParameter, haloState.segment(0,6)) << endl;
    return;

}
