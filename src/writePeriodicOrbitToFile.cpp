#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>
#include <Eigen/Dense>

#include "thesisProject/src/propagateOrbit.h"


Eigen::VectorXd writePeriodicOrbitToFile( Eigen::VectorXd initialStateVector, int librationPointNr, string orbitType,
                                          int orbitId, double orbitalPeriod, const double massParameter,
                                          int saveEveryNthIntegrationStep = 100)
{
    // Initialize output vector and STM
    Eigen::VectorXd outputVector( 43 );
    Eigen::VectorXd initialStateVectorInclSTM = Eigen::VectorXd::Zero(42);
    Eigen::MatrixXd identityMatrix = Eigen::MatrixXd::Identity(6, 6);
    identityMatrix.resize(36, 1);
    initialStateVectorInclSTM.segment(0,6) = initialStateVector;
    initialStateVectorInclSTM.segment(6,36) = identityMatrix;

    // Prepare output file
    remove(("../data/raw/" + orbitType + "_L" + to_string(librationPointNr) + "_" + to_string(orbitId) + ".txt").c_str());
    ofstream textFileOrbit(("../data/raw/" + orbitType + "_L" + to_string(librationPointNr) + "_" + to_string(orbitId) + ".txt").c_str());
    textFileOrbit.precision(std::numeric_limits<double>::digits10);

    // Write initial state to file
    textFileOrbit << left << scientific           << setw(25) << 0.0                          << setw(25)
                  << initialStateVectorInclSTM(0) << setw(25) << initialStateVectorInclSTM(1) << setw(25)
                  << initialStateVectorInclSTM(2) << setw(25) << initialStateVectorInclSTM(3) << setw(25)
                  << initialStateVectorInclSTM(4) << setw(25) << initialStateVectorInclSTM(5) << endl;

    // Perform first integration step
    outputVector = propagateOrbit( initialStateVectorInclSTM, massParameter, 0.0, 1.0 );
    Eigen::VectorXd stateVectorInclSTM = outputVector.segment(0,42);
    double currentTime = outputVector(42);

    int count = 1;

    // Perform integration steps until end of orbital period
    while (currentTime <= orbitalPeriod) {

        stateVectorInclSTM = outputVector.segment(0,42);
        currentTime = outputVector(42);

        // Write every nth integration step to file.
        if (count % saveEveryNthIntegrationStep == 0) {
            textFileOrbit << left << scientific    << setw(25) << currentTime           << setw(25)
                          << stateVectorInclSTM(0) << setw(25) << stateVectorInclSTM(1) << setw(25)
                          << stateVectorInclSTM(2) << setw(25) << stateVectorInclSTM(3) << setw(25)
                          << stateVectorInclSTM(4) << setw(25) << stateVectorInclSTM(5) << endl;
        }

        // Propagate to next time step
        outputVector = propagateOrbit(stateVectorInclSTM, massParameter, currentTime, 1.0 );
        count += 1;
    }

    // Save last integration step
    textFileOrbit << left << fixed         << setw(25) << currentTime           << setw(25)
                  << stateVectorInclSTM(0) << setw(25) << stateVectorInclSTM(1) << setw(25)
                  << stateVectorInclSTM(2) << setw(25) << stateVectorInclSTM(3) << setw(25)
                  << stateVectorInclSTM(4) << setw(25) << stateVectorInclSTM(5) << endl;

    // Close orbit file and clear the variable for the next orbit
    textFileOrbit.close();
    textFileOrbit.clear();

    return stateVectorInclSTM;
}
