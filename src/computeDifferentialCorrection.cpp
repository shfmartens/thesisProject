#include <Eigen/Core>
#include <Eigen/LU>
#include <iostream>

#include "computeDifferentialCorrection.h"
#include "stateDerivativeModel.h"



Eigen::VectorXd computeDifferentialCorrection( const int librationPointNr, const std::string& orbitType,
                                               const Eigen::MatrixXd& cartesianStateWithStm, const bool xPositionFixed )
{
    // Initiate vectors, matrices etc.
    Eigen::VectorXd cartesianState = cartesianStateWithStm.block( 0, 0, 6, 1 );
    Eigen::MatrixXd stmPartOfStateVectorInMatrixForm = cartesianStateWithStm.block( 0, 1, 6, 6 );

    Eigen::VectorXd differentialCorrection(7);
    Eigen::VectorXd corrections(3);
    Eigen::MatrixXd updateMatrix(3,3);
    Eigen::MatrixXd multiplicationMatrix(3,1);

    //std::cout << "stmPartOfStateVectorInMatrixForm: \n"<<stmPartOfStateVectorInMatrixForm << std::endl;


    // Compute the accelerations and velocities (in X- and Z-direction) on the spacecraft and put them in a 2x1 vector.
    Eigen::VectorXd cartesianAccelerations = computeStateDerivative(0.0, cartesianStateWithStm).block(0,0,6,1);

    // If type is axial, the desired state vector has the form [x, 0, 0, 0, ydot, zdot] and requires a differential correction for {x, ydot, T/2}
    if (orbitType == "axial")
    {
        std::cout << "z-position: " << cartesianState(2) << std::endl;
        std::cout << "x-velocity: " << cartesianState(3) << std::endl;
        // Check which deviation is larger: x-velocity or z-position.
        if ( std::abs(cartesianState(2)) < std::abs(cartesianState(3)) and !xPositionFixed )
        {
            // Correction on {x, ydot, T/2} for constant {zdot}

            // Set the correct multiplication matrix (state at T/2)
            multiplicationMatrix << cartesianState(1), cartesianState(2), cartesianState(3);

            // Compute the update matrix.
            updateMatrix << stmPartOfStateVectorInMatrixForm(1, 0), stmPartOfStateVectorInMatrixForm(1, 4), cartesianState(4),
                            stmPartOfStateVectorInMatrixForm(2, 0), stmPartOfStateVectorInMatrixForm(2, 4), cartesianState(5),
                            stmPartOfStateVectorInMatrixForm(3, 0), stmPartOfStateVectorInMatrixForm(3, 4), cartesianAccelerations(3);

            // Compute the necessary differential correction.
            corrections = updateMatrix.inverse() * multiplicationMatrix;

            // Put corrections in correct format.
            differentialCorrection.setZero();
            differentialCorrection(0) = -corrections(0);
            differentialCorrection(4) = -corrections(1);
            differentialCorrection(6) = -corrections(2);
        }
        else
        {
            // Correction on {ydot, zdot T/2} for constant {x}

            // Set the correct multiplication matrix (state at T/2)
            multiplicationMatrix << cartesianState(1), cartesianState(2), cartesianState(3);

            // Compute the update matrix.
            updateMatrix << stmPartOfStateVectorInMatrixForm(1, 4), stmPartOfStateVectorInMatrixForm(1, 5), cartesianState(4),
                            stmPartOfStateVectorInMatrixForm(2, 4), stmPartOfStateVectorInMatrixForm(2, 5), cartesianState(5),
                            stmPartOfStateVectorInMatrixForm(3, 4), stmPartOfStateVectorInMatrixForm(3, 5), cartesianAccelerations(3);

            // Compute the necessary differential correction.
            corrections = updateMatrix.inverse() * multiplicationMatrix;

            // Put corrections in correct format.
            differentialCorrection.setZero();
            differentialCorrection(4) = -corrections(0);
            differentialCorrection(5) = -corrections(1);
            differentialCorrection(6) = -corrections(2);
        }
    }

    // If type is not axial, the desired state vector has the form [x, 0, z, 0, ydot, 0] and requires a differential correction for either {z, ydot, T/2} or {x, ydot, T/2}
    else
    {

        // Check which deviation is larger: x-velocity or z-velocity.
        if ( std::abs(cartesianState(3)) < std::abs(cartesianState(5)) or orbitType == "horizontal" or
             (orbitType == "halo" and librationPointNr == 2) )
        {
            // Correction on {z, ydot, T/2} for constant {x}

            // Set the correct multiplication matrix (state at T/2)
            multiplicationMatrix << cartesianState(1), cartesianState(3), cartesianState(5);

            // Compute the update matrix.
            updateMatrix         << stmPartOfStateVectorInMatrixForm(1, 2), stmPartOfStateVectorInMatrixForm(1, 4), cartesianState(4),
                                    stmPartOfStateVectorInMatrixForm(3, 2), stmPartOfStateVectorInMatrixForm(3, 4), cartesianAccelerations(3),
                                    stmPartOfStateVectorInMatrixForm(5, 2), stmPartOfStateVectorInMatrixForm(5, 4), cartesianAccelerations(5);

            // Compute the necessary differential correction.
            corrections = updateMatrix.inverse() * multiplicationMatrix;

            // Put corrections in correct format.
            differentialCorrection.setZero();
            differentialCorrection(2) = -corrections(0);
            differentialCorrection(4) = -corrections(1);
            differentialCorrection(6) = -corrections(2);
        }
        else
        {
            // Correction on {x, ydot, T/2} for constant {z}
            // Set the correct multiplication matrix (state at T/2)
            multiplicationMatrix << cartesianState(1), cartesianState(3), cartesianState(5);

            // Compute the update matrix.
            updateMatrix << stmPartOfStateVectorInMatrixForm(1, 0), stmPartOfStateVectorInMatrixForm(1, 4), cartesianState(4),
                            stmPartOfStateVectorInMatrixForm(3, 0), stmPartOfStateVectorInMatrixForm(3, 4), cartesianAccelerations(3),
                            stmPartOfStateVectorInMatrixForm(5, 0), stmPartOfStateVectorInMatrixForm(5, 4), cartesianAccelerations(5);

            // Compute the necessary differential correction.
            corrections = updateMatrix.inverse() * multiplicationMatrix;

            // Put corrections in correct format.
            differentialCorrection.setZero();
            differentialCorrection(0) = -corrections(0);
            differentialCorrection(4) = -corrections(1);
            differentialCorrection(6) = -corrections(2);
        }
    }

    // Return differential correction.
    return differentialCorrection;

}
