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



void computeManifolds( Eigen::VectorXd initialStateVector, double orbitalPeriod, int librationPointNr,
                       std::string orbitType, int orbitId,
                       const double primaryGravitationalParameter = tudat::celestial_body_constants::EARTH_GRAVITATIONAL_PARAMETER,
                       const double secondaryGravitationalParameter = tudat::celestial_body_constants::MOON_GRAVITATIONAL_PARAMETER,
                       double displacementFromOrbit = 1.0e-6, int numberOfManifoldOrbits = 100, int saveEveryNthIntegrationStep = 1000,
                       double maximumIntegrationTimeManifoldOrbits = 50.0, double maxEigenvalueDeviation = 1.0e-3,
                       double normalizedRadiusSecondPrimary = 4.5187304890738815e-3) //TODO replace hardcoded with normalized moon radius
{
    // Set output maximum precision
    std::cout.precision(std::numeric_limits<double>::digits10);

    // Define massParameter, initialStateVector and halfPeriodStateVector.
    massParameter = tudat::gravitation::circular_restricted_three_body_problem::computeMassParameter( primaryGravitationalParameter, secondaryGravitationalParameter );
    double jacobiEnergy = tudat::gravitation::circular_restricted_three_body_problem::computeJacobiEnergy(massParameter, initialStateVector);

    Eigen::VectorXd initialStateVectorInclSTM = Eigen::VectorXd::Zero(42);
    initialStateVectorInclSTM.segment(0,6) = initialStateVector;
    Eigen::MatrixXd identityMatrix = Eigen::MatrixXd::Identity(6, 6);
    identityMatrix.resize(36, 1);
    initialStateVectorInclSTM.segment(6,36) = identityMatrix;

    std::cout << "\nInitial state vector:" << std::endl << initialStateVectorInclSTM.segment(0,6) << std::endl
              << "\nwith C: " << jacobiEnergy << ", T: " << orbitalPeriod << std::endl;;

    std::vector< std::vector <double> > orbitStateVectors;
    std::vector<double> tempStateVector;
    int numberOfPointsOnPeriodicOrbit = 1;  // Initial state

    for (int i = 0; i <= 41; i++){
        tempStateVector.push_back(initialStateVectorInclSTM(i));
    }
    orbitStateVectors.push_back(tempStateVector);

    // Perform first integration step
    Eigen::VectorXd outputVector               = propagateOrbit( initialStateVectorInclSTM, massParameter, 0.0, 1.0 );
    Eigen::VectorXd stateVectorInclSTM         = outputVector.segment(0,42);
    Eigen::VectorXd previousOutputVector       = outputVector;
    double currentTime                         = outputVector(42);

    // Perform integration steps until end of orbital period
    for (int i = 5; i <= 12; i++) {

        double initialStepSize = pow(10,(static_cast<float>(-i)));
        double maximumStepSize = pow(10,(static_cast<float>(-i) + 1.0));

        while (currentTime <= orbitalPeriod) {
            stateVectorInclSTM      = outputVector.segment(0, 42);
            currentTime             = outputVector(42);
            previousOutputVector    = outputVector;

            // Save the STM at every point along the orbit, for the same stepsize
            if (i == 5){
                numberOfPointsOnPeriodicOrbit += 1;
                tempStateVector.clear();

                for (int j = 0; j <= 41; j++){
                    tempStateVector.push_back(stateVectorInclSTM(j));
                }
                orbitStateVectors.push_back(tempStateVector);
            }

            // Propagate to next time step
            outputVector         = propagateOrbit(stateVectorInclSTM, massParameter, currentTime, 1.0, initialStepSize, maximumStepSize);

            if (outputVector(42) > orbitalPeriod) {
                outputVector = previousOutputVector;
                break;
            }
        }
    }
    std::cout << "number of points on orbit: " << numberOfPointsOnPeriodicOrbit << std::endl;

    // Reshape vector of vectors to matrix
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
    std::cout << "\nMonodromy matrix:\n" << monodromyMatrix << "\n" << std::endl;

    // Compute eigenvectors of the monodromy matrix (find minimum eigenvalue, corresponding to stable, and large for unstable)
    Eigen::EigenSolver<Eigen::MatrixXd> eig(monodromyMatrix);
    std::cout << "Eigenvectors:\n" << eig.eigenvectors() << "\n\n" << "Eigenvalues:\n" << eig.eigenvalues() << "\n" << std::endl;

    int indexMaximumEigenvalue;
    double maximumEigenvalue = 0.0;
    int indexMinimumEigenvalue;
    double minimumEigenvalue = 1000.0;

    for (int i = 0; i <= 5; i++){
        if (eig.eigenvalues().real()(i) > maximumEigenvalue and std::abs(eig.eigenvalues().imag()(i)) < maxEigenvalueDeviation){
            maximumEigenvalue = eig.eigenvalues().real()(i);
            indexMaximumEigenvalue = i;
        }
        if (std::abs(eig.eigenvalues().real()(i)) < minimumEigenvalue and std::abs(eig.eigenvalues().imag()(i)) < maxEigenvalueDeviation){
            minimumEigenvalue = std::abs(eig.eigenvalues().real()(i));
            indexMinimumEigenvalue = i;
        }
    }

    Eigen::VectorXd eigenVector1 = eig.eigenvectors().real().col(indexMaximumEigenvalue);
    Eigen::VectorXd eigenVector2 = eig.eigenvectors().real().col(indexMinimumEigenvalue);
    std::cout << "Maximum real eigenvalue of " << maximumEigenvalue << " at " << indexMaximumEigenvalue
         << ", corresponding to eigenvector (unstable manifold): \n" << eigenVector1 << "\n\n"
         << "Minimum absolute real eigenvalue: " << minimumEigenvalue << " at " << indexMinimumEigenvalue
         << ", corresponding to eigenvector (stable manifold): \n" << eigenVector2 << std::endl;

    // Check whether the two selected eigenvalues belong to the same reciprocal pair
    if ((1.0 / minimumEigenvalue - maximumEigenvalue) > maxEigenvalueDeviation){
        std::cout << "\n\n\nERROR - EIGENVALUES MIGHT NOT BELONG TO SAME RECIPROCAL PAIR" << std::endl;
        std::ofstream textFileEigenvalueError;
        textFileEigenvalueError.open("../data/raw/manifolds/error_file.txt", std::ios_base::app);
        textFileEigenvalueError << orbitId << "\n\n"
                  << "Eigenvectors:\n" << eig.eigenvectors() << "\n"
                  << "Eigenvalues:\n" << eig.eigenvalues() << "\n"
                  << "Maximum real eigenvalue of " << maximumEigenvalue << " at " << indexMaximumEigenvalue
                  << ", corresponding to eigenvector (unstable manifold): \n" << eigenVector1 << "\n\n"
                  << "Minimum absolute real eigenvalue: " << minimumEigenvalue << " at " << indexMinimumEigenvalue
                  << ", corresponding to eigenvector (stable manifold): \n" << eigenVector2 << "\n\n" << std::endl;
        textFileEigenvalueError.close();
    }

    Eigen::VectorXd manifoldStartingState      = Eigen::VectorXd::Zero(42);
    Eigen::VectorXd localStateVector           = Eigen::VectorXd::Zero(6);
    Eigen::VectorXd localNormalizedEigenvector = Eigen::VectorXd::Zero(6);

    double signEigenvector1;
    double signEigenvector2;

    if (eigenVector1(0) > 0.0){
        signEigenvector1 = 1.0;
    } else {
        signEigenvector1 = -1.0;
    }
    if (eigenVector2(0) > 0.0){
        signEigenvector2 = 1.0;
    } else {
        signEigenvector2 = -1.0;
    }

//    std::vector<double> offsetSigns           = {1.0, -1.0, 1.0, -1.0};
    std::vector<double> offsetSigns                        = {1.0*signEigenvector2, -1.0*signEigenvector2, 1.0*signEigenvector1, -1.0*signEigenvector1};
    std::vector<Eigen::VectorXd> eigenVectors              = {eigenVector2, eigenVector2, eigenVector1, eigenVector1};
    std::vector<double> integrationDirections              = {-1.0, -1.0, 1.0, 1.0};
    std::vector<std::string> fileNamesStateVectors         = {"L" + std::to_string(librationPointNr) + "_" + orbitType + "_" + std::to_string(orbitId) + "_W_S_plus.txt",
                                                              "L" + std::to_string(librationPointNr) + "_" + orbitType + "_" + std::to_string(orbitId) + "_W_S_min.txt",
                                                              "L" + std::to_string(librationPointNr) + "_" + orbitType + "_" + std::to_string(orbitId) + "_W_U_plus.txt",
                                                              "L" + std::to_string(librationPointNr) + "_" + orbitType + "_" + std::to_string(orbitId) + "_W_U_min.txt"};
    std::vector<std::string> fileNamesEigenvectors         = {"L" + std::to_string(librationPointNr) + "_" + orbitType + "_" + std::to_string(orbitId) + "_W_S_plus_eigenvector.txt",
                                                              "L" + std::to_string(librationPointNr) + "_" + orbitType + "_" + std::to_string(orbitId) + "_W_S_min_eigenvector.txt",
                                                              "L" + std::to_string(librationPointNr) + "_" + orbitType + "_" + std::to_string(orbitId) + "_W_U_plus_eigenvector.txt",
                                                              "L" + std::to_string(librationPointNr) + "_" + orbitType + "_" + std::to_string(orbitId) + "_W_U_min_eigenvector.txt"};
    std::vector<std::string> fileNamesEigenvectorLocations = {"L" + std::to_string(librationPointNr) + "_" + orbitType + "_" + std::to_string(orbitId) + "_W_S_plus_eigenvector_location.txt",
                                                              "L" + std::to_string(librationPointNr) + "_" + orbitType + "_" + std::to_string(orbitId) + "_W_S_min_eigenvector_location.txt",
                                                              "L" + std::to_string(librationPointNr) + "_" + orbitType + "_" + std::to_string(orbitId) + "_W_U_plus_eigenvector_location.txt",
                                                              "L" + std::to_string(librationPointNr) + "_" + orbitType + "_" + std::to_string(orbitId) + "_W_U_min_eigenvector_location.txt"};
    double offsetSign;
    Eigen::VectorXd eigenVector;
    double integrationDirection;
    std::string fileNameStateVector;
    std::string fileNameEigenvector;
    std::string fileNameEigenvectorLocation;

    for (int manifoldNumber = 0; manifoldNumber < 4; manifoldNumber++){
        offsetSign                  = offsetSigns.at(manifoldNumber);
        eigenVector                 = eigenVectors.at(manifoldNumber);
        integrationDirection        = integrationDirections.at(manifoldNumber);
        fileNameStateVector         = fileNamesStateVectors.at(manifoldNumber);
        fileNameEigenvector         = fileNamesEigenvectors.at(manifoldNumber);
        fileNameEigenvectorLocation = fileNamesEigenvectorLocations.at(manifoldNumber);

        std::ofstream textFileStateVectors;
        remove(("../data/raw/manifolds/" + fileNameStateVector).c_str());
        textFileStateVectors.open(("../data/raw/manifolds/" + fileNameStateVector).c_str());
        textFileStateVectors.precision(14);

        std::ofstream textFileEigenvectors;
        remove(("../data/raw/manifolds/" + fileNameEigenvector).c_str());
        textFileEigenvectors.open(("../data/raw/manifolds/" + fileNameEigenvector).c_str());
        textFileEigenvectors.precision(14);

        std::ofstream textFileEigenvectorLocations;
        remove(("../data/raw/manifolds/" + fileNameEigenvectorLocation).c_str());
        textFileEigenvectorLocations.open(("../data/raw/manifolds/" + fileNameEigenvectorLocation).c_str());
        textFileEigenvectorLocations.precision(14);

        bool fullManifoldComputed = false;
        bool xDiffSignSet = false;
        double xDiffSign = 0.0;
        bool ySignSet = false;
        double ySign = 0.0;

        std::cout << "\n\nManifold: " << fileNameStateVector << "\n" << std::endl;

        // Determine the total number of points along the periodic orbit to start the manifolds.
        for (int ii = 0; ii <numberOfManifoldOrbits; ii++) {
            int row_index = floor(ii * numberOfPointsOnPeriodicOrbit / numberOfManifoldOrbits);

            // Reshape the STM from vector to a matrix
            Eigen::VectorXd STMvector       = orbitStateVectorsMatrix.row(row_index).segment(6,36);
            Eigen::Map<Eigen::MatrixXd> STM = Eigen::Map<Eigen::MatrixXd>(STMvector.data(),6,6);
//            Eigen::MatrixXd STM (6, 6);
//            STM <<  orbitStateVectorsMatrix(row_index, 6),  orbitStateVectorsMatrix(row_index, 12), orbitStateVectorsMatrix(row_index, 18), orbitStateVectorsMatrix(row_index, 24), orbitStateVectorsMatrix(row_index, 30), orbitStateVectorsMatrix(row_index, 36),
//                    orbitStateVectorsMatrix(row_index, 7),  orbitStateVectorsMatrix(row_index, 13), orbitStateVectorsMatrix(row_index, 19), orbitStateVectorsMatrix(row_index, 25), orbitStateVectorsMatrix(row_index, 31), orbitStateVectorsMatrix(row_index, 37),
//                    orbitStateVectorsMatrix(row_index, 8),  orbitStateVectorsMatrix(row_index, 14), orbitStateVectorsMatrix(row_index, 20), orbitStateVectorsMatrix(row_index, 26), orbitStateVectorsMatrix(row_index, 32), orbitStateVectorsMatrix(row_index, 38),
//                    orbitStateVectorsMatrix(row_index, 9),  orbitStateVectorsMatrix(row_index, 15), orbitStateVectorsMatrix(row_index, 21), orbitStateVectorsMatrix(row_index, 27), orbitStateVectorsMatrix(row_index, 33), orbitStateVectorsMatrix(row_index, 39),
//                    orbitStateVectorsMatrix(row_index, 10), orbitStateVectorsMatrix(row_index, 16), orbitStateVectorsMatrix(row_index, 22), orbitStateVectorsMatrix(row_index, 28), orbitStateVectorsMatrix(row_index, 34), orbitStateVectorsMatrix(row_index, 40),
//                    orbitStateVectorsMatrix(row_index, 11), orbitStateVectorsMatrix(row_index, 17), orbitStateVectorsMatrix(row_index, 23), orbitStateVectorsMatrix(row_index, 29), orbitStateVectorsMatrix(row_index, 35), orbitStateVectorsMatrix(row_index, 41);
//            std::cout << "\nSTM:\n" << STM << "\n" << std::endl;
//            std::cout << "\neigenvector:\n" << eigenVector << "\n" << std::endl;
//            std::cout << "\nSTM*eigenvector:\n" << STM*eigenVector << "\n" << std::endl;
//            std::cout << "\nnorm(STM*eigenvector):\n" << (STM*eigenVector).normalized() << "\n" << std::endl;
//            textFile1 << left << fixed << setw(20)
//                      << (STM*eigenVector).normalized()(0) << setw(20) << (STM*eigenVector).normalized()(1) << setw(20)
//                      << (STM*eigenVector).normalized()(2) << setw(20) << (STM*eigenVector).normalized()(3) << setw(20)
//                      << (STM*eigenVector).normalized()(4) << setw(20) << (STM*eigenVector).normalized()(5) << endl;

            // Apply displacement epsilon from the periodic orbit at <numberOfManifoldOrbits> locations on the final orbit.
            localStateVector                    = orbitStateVectorsMatrix.block(floor(ii * numberOfPointsOnPeriodicOrbit / numberOfManifoldOrbits), 0, 1, 6).transpose();
            localNormalizedEigenvector          = (STM*eigenVector).normalized();
            manifoldStartingState.segment(0, 6) = localStateVector + offsetSign * displacementFromOrbit * localNormalizedEigenvector;
            manifoldStartingState.segment(6,36) = identityMatrix;

            textFileEigenvectors << std::left << std::scientific << std::setw(25)
                                 << localNormalizedEigenvector(0) << std::setw(25) << localNormalizedEigenvector(1) << std::setw(25)
                                 << localNormalizedEigenvector(2) << std::setw(25) << localNormalizedEigenvector(3) << std::setw(25)
                                 << localNormalizedEigenvector(4) << std::setw(25) << localNormalizedEigenvector(5) << std::endl;

            textFileEigenvectorLocations << std::left << std::scientific << std::setw(25)
                                         << localStateVector(0) << std::setw(25) << localStateVector(1) << std::setw(25)
                                         << localStateVector(2) << std::setw(25) << localStateVector(3) << std::setw(25)
                                         << localStateVector(4) << std::setw(25) << localStateVector(5) << std::endl;

            textFileStateVectors << std::left << std::scientific << std::setw(25) << 0.0         << std::setw(25)
                      << manifoldStartingState(0)  << std::setw(25) << manifoldStartingState(1)  << std::setw(25)
                      << manifoldStartingState(2)  << std::setw(25) << manifoldStartingState(3)  << std::setw(25)
                      << manifoldStartingState(4)  << std::setw(25) << manifoldStartingState(5)  << std::setw(25)
                      << manifoldStartingState(6)  << std::setw(25) << manifoldStartingState(7)  << std::setw(25)
                      << manifoldStartingState(8)  << std::setw(25) << manifoldStartingState(9)  << std::setw(25)
                      << manifoldStartingState(10) << std::setw(25) << manifoldStartingState(11) << std::setw(25)
                      << manifoldStartingState(12) << std::setw(25) << manifoldStartingState(13) << std::setw(25)
                      << manifoldStartingState(14) << std::setw(25) << manifoldStartingState(15) << std::setw(25)
                      << manifoldStartingState(16) << std::setw(25) << manifoldStartingState(17) << std::setw(25)
                      << manifoldStartingState(18) << std::setw(25) << manifoldStartingState(19) << std::setw(25)
                      << manifoldStartingState(20) << std::setw(25) << manifoldStartingState(21) << std::setw(25)
                      << manifoldStartingState(22) << std::setw(25) << manifoldStartingState(23) << std::setw(25)
                      << manifoldStartingState(24) << std::setw(25) << manifoldStartingState(25) << std::setw(25)
                      << manifoldStartingState(26) << std::setw(25) << manifoldStartingState(27) << std::setw(25)
                      << manifoldStartingState(28) << std::setw(25) << manifoldStartingState(29) << std::setw(25)
                      << manifoldStartingState(30) << std::setw(25) << manifoldStartingState(31) << std::setw(25)
                      << manifoldStartingState(32) << std::setw(25) << manifoldStartingState(33) << std::setw(25)
                      << manifoldStartingState(34) << std::setw(25) << manifoldStartingState(35) << std::setw(25)
                      << manifoldStartingState(36) << std::setw(25) << manifoldStartingState(37) << std::setw(25)
                      << manifoldStartingState(38) << std::setw(25) << manifoldStartingState(39) << std::setw(25)
                      << manifoldStartingState(40) << std::setw(25) << manifoldStartingState(41) << std::endl;

            outputVector       = propagateOrbit(manifoldStartingState, massParameter, 0.0, integrationDirection );
            stateVectorInclSTM = outputVector.segment(0, 42);
            currentTime        = outputVector(42);
            std::cout << "Orbit No.: " << ii << std::endl;

            int count = 1;

            while ( (std::abs( currentTime ) <= maximumIntegrationTimeManifoldOrbits) and !fullManifoldComputed ) {
                // Check whether trajectory is not intersecting with the surface of the second primary.
                // This can increase velocities to unreasonably high numbers, causing the integrator to throw a MinimumStepSizeExceedsError
                if ( pow( pow(outputVector(0)-(1.0-massParameter), 2) + pow(outputVector(1), 2)
                          + pow(outputVector(2), 2), 0.5) < normalizedRadiusSecondPrimary ){
                    std::cout << "Integration stopped as trajectory position intersects surface of the second primary" << std::endl;
                    outputVector = previousOutputVector;
                    fullManifoldComputed = true;
                }

                // Determine sign of Y near x = 0  (U1, U4)
                if ( (outputVector(0) < 0) and !ySignSet ){
                    if ( outputVector(1) < 0 ){
                        ySign = -1.0;
                    }
                    if ( outputVector(1) > 0 ){
                        ySign = 1.0;
                    }
                    ySignSet = true;
                }

                // Determine when the manifold crosses the x-axis again (U1, U4)
                if ( (outputVector(1) * ySign < 0) and ySignSet ){

                    outputVector = previousOutputVector;
                    std::cout << "||y|| = " << outputVector(1) << ", at start of iterative procedure" << std::endl;

                    for (int i = 5; i <= 12; i++) {

                        double initialStepSize = pow(10,(static_cast<float>(-i)));
                        double maximumStepSize = pow(10,(static_cast<float>(-i) + 1.0));

                        while (outputVector(1) * ySign > 0) {
                            stateVectorInclSTM      = outputVector.segment(0, 42);
                            currentTime             = outputVector(42);
                            previousOutputVector    = outputVector;
                            outputVector            = propagateOrbit(stateVectorInclSTM, massParameter, currentTime, integrationDirection, initialStepSize, maximumStepSize);

                            if (outputVector(1) * ySign < 0) {
                                outputVector = previousOutputVector;
                                break;
                            }
                        }
                    }
                    std::cout << "||y|| = " << outputVector(1) << ", at end of iterative procedure" << std::endl;
                    fullManifoldComputed = true;
                }

                // Determine sign of Xdot within 5% of second primary, 1-mu  (U2, U3)
                if ( std::abs((outputVector(0) - (1.0 - massParameter)) / (1.0 - massParameter)) < 0.05 and !xDiffSignSet ){
                    if ( (outputVector(0) - (1.0 - massParameter)) < 0 ){
                        xDiffSign = -1.0;
                    }
                    if ( (outputVector(0) - (1.0 - massParameter)) > 0 ){
                        xDiffSign = 1.0;
                    }
                    xDiffSignSet = true;
                }

                // Determine when the manifold crosses the second primary (U1, U4)
                if ( ((outputVector(0) - (1.0 - massParameter)) * xDiffSign < 0)
                     and (std::abs(outputVector(1)) < 1.0) and xDiffSignSet and !ySignSet ){

                    outputVector = previousOutputVector;
                    std::cout << "||x - (1-mu)|| = " << (outputVector(0) - (1.0 - massParameter)) << ", at start of iterative procedure" << std::endl;

                    for (int i = 5; i <= 12; i++) {

                        double initialStepSize = pow(10,(static_cast<float>(-i)));
                        double maximumStepSize = pow(10,(static_cast<float>(-i) + 1.0));

                        while ((outputVector(0) - (1.0 - massParameter)) * xDiffSign > 0) {
                            stateVectorInclSTM      = outputVector.segment(0, 42);
                            currentTime             = outputVector(42);
                            previousOutputVector    = outputVector;
                            outputVector            = propagateOrbit(stateVectorInclSTM, massParameter, currentTime, integrationDirection, initialStepSize, maximumStepSize);

                            if ((outputVector(0) - 1.0 + massParameter) * xDiffSign < 0) {
                                outputVector = previousOutputVector;
                                break;
                            }
                        }
                    }

                    std::cout << "||x - (1-mu)|| = " << (outputVector(0) - 1.0 + massParameter) << ", at end of iterative procedure" << std::endl;
                    fullManifoldComputed = true;
                }

                stateVectorInclSTM = outputVector.segment(0, 42);
                currentTime = outputVector(42);

                // Write every nth integration step to file.
                if (count % saveEveryNthIntegrationStep == 0 or fullManifoldComputed) {
                    textFileStateVectors << std::left << std::scientific << std::setw(25) << currentTime << std::setw(25)
                                         << stateVectorInclSTM(0)  << std::setw(25) << stateVectorInclSTM(1)  << std::setw(25)
                                         << stateVectorInclSTM(2)  << std::setw(25) << stateVectorInclSTM(3)  << std::setw(25)
                                         << stateVectorInclSTM(4)  << std::setw(25) << stateVectorInclSTM(5)  << std::setw(25)
                                         << stateVectorInclSTM(6)  << std::setw(25) << stateVectorInclSTM(7)  << std::setw(25)
                                         << stateVectorInclSTM(8)  << std::setw(25) << stateVectorInclSTM(9)  << std::setw(25)
                                         << stateVectorInclSTM(10) << std::setw(25) << stateVectorInclSTM(11) << std::setw(25)
                                         << stateVectorInclSTM(12) << std::setw(25) << stateVectorInclSTM(13) << std::setw(25)
                                         << stateVectorInclSTM(14) << std::setw(25) << stateVectorInclSTM(15) << std::setw(25)
                                         << stateVectorInclSTM(16) << std::setw(25) << stateVectorInclSTM(17) << std::setw(25)
                                         << stateVectorInclSTM(18) << std::setw(25) << stateVectorInclSTM(19) << std::setw(25)
                                         << stateVectorInclSTM(20) << std::setw(25) << stateVectorInclSTM(21) << std::setw(25)
                                         << stateVectorInclSTM(22) << std::setw(25) << stateVectorInclSTM(23) << std::setw(25)
                                         << stateVectorInclSTM(24) << std::setw(25) << stateVectorInclSTM(25) << std::setw(25)
                                         << stateVectorInclSTM(26) << std::setw(25) << stateVectorInclSTM(27) << std::setw(25)
                                         << stateVectorInclSTM(28) << std::setw(25) << stateVectorInclSTM(29) << std::setw(25)
                                         << stateVectorInclSTM(30) << std::setw(25) << stateVectorInclSTM(31) << std::setw(25)
                                         << stateVectorInclSTM(32) << std::setw(25) << stateVectorInclSTM(33) << std::setw(25)
                                         << stateVectorInclSTM(34) << std::setw(25) << stateVectorInclSTM(35) << std::setw(25)
                                         << stateVectorInclSTM(36) << std::setw(25) << stateVectorInclSTM(37) << std::setw(25)
                                         << stateVectorInclSTM(38) << std::setw(25) << stateVectorInclSTM(39) << std::setw(25)
                                         << stateVectorInclSTM(40) << std::setw(25) << stateVectorInclSTM(41) << std::endl;
                }
                // Propagate to next time step.
                previousOutputVector = outputVector;
                outputVector         = propagateOrbit(stateVectorInclSTM, massParameter, currentTime, integrationDirection );
                count += 1;
            }
            ySignSet = false;
            xDiffSignSet = false;
            fullManifoldComputed = false;
        }
        textFileStateVectors.close();
        textFileStateVectors.clear();
        textFileEigenvectors.close();
        textFileEigenvectors.clear();
        textFileEigenvectorLocations.close();
        textFileEigenvectorLocations.clear();
    }


    std::cout << std::endl
              << "=================================================================="           << std::endl
              << "                          "   << orbitId        << "                        " << std::endl
              << "Mass parameter: "             << massParameter                                << std::endl
              << "C at initial conditions: "    << jacobiEnergy                                 << std::endl
              << "C at end of manifold orbit: " << tudat::gravitation::circular_restricted_three_body_problem::computeJacobiEnergy(massParameter, stateVectorInclSTM.segment(0,6)) << std::endl
              << "T: " << orbitalPeriod                                                         << std::endl
              << "=================================================================="           << std::endl;
    return;

}
