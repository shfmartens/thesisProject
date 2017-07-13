#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>
#include <Eigen/Dense>

#include "thesisProject/src/propagateOrbit.h"


Eigen::VectorXd writePeriodicOrbitToFile( Eigen::VectorXd initialStateVector, int lagrangePointNr, string orbitType,
                                          int orbitId, double orbitalPeriod, const double massParameter)
{
    // Initialize output vector and STM
    Eigen::VectorXd outputVector( 43 );
    Eigen::VectorXd initialStateVectorInclSTM = Eigen::VectorXd::Zero(42);
    Eigen::MatrixXd identityMatrix = Eigen::MatrixXd::Identity(6, 6);
    identityMatrix.resize(36, 1);
    initialStateVectorInclSTM.segment(0,6) = initialStateVector;
    initialStateVectorInclSTM.segment(6,36) = identityMatrix;

    // Perform first integration step
    outputVector = propagateOrbit( initialStateVectorInclSTM, massParameter, 0.0, 1.0, orbitType);
    Eigen::VectorXd stateVectorInclSTM = outputVector.segment( 0, 42 );
    double currentTime = outputVector( 42 );

    // Prepare output file
    remove(("../data/raw/" + orbitType + "_L" + to_string(lagrangePointNr) + "_" + to_string(orbitId) + ".txt").c_str());
    ofstream textFileOrbit(("../data/raw/" + orbitType + "_L" + to_string(lagrangePointNr) + "_" + to_string(orbitId) + ".txt").c_str());
    textFileOrbit.precision(14);

    // Perform integration steps until end of orbital period
    while (currentTime <= orbitalPeriod) {

        stateVectorInclSTM = outputVector.segment( 0, 42 );
        currentTime = outputVector( 42 );

        // Write to file.
        textFileOrbit << left << fixed << setw(20) << currentTime << setw(20)
                      << stateVectorInclSTM(0) << setw(20) << stateVectorInclSTM(1) << setw(20)
                      << stateVectorInclSTM(2) << setw(20) << stateVectorInclSTM(3) << setw(20)
                      << stateVectorInclSTM(4) << setw(20) << stateVectorInclSTM(5)  << endl;

        // Propagate to next time step.
        outputVector = propagateOrbit(stateVectorInclSTM, massParameter, currentTime, 1.0, orbitType);
    }

    // Close orbit file and clear the variable for the next orbit
    textFileOrbit.close();
    textFileOrbit.clear();

    return stateVectorInclSTM;
}
