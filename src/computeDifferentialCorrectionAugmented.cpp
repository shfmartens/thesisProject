#include <Eigen/Core>
#include <Eigen/LU>
#include <iostream>

#include "computeDifferentialCorrectionAugmented.h"
#include "stateDerivativeModelAugmented.h"



Eigen::VectorXd computeDifferentialCorrectionAugmented( const Eigen::MatrixXd& cartesianStateWithStm, const Eigen::VectorXd& deviationVector, const bool symmetryDependence, const int stateIndex )
{

    // Initiate vectors, matrices etc.
    Eigen::VectorXd cartesianState = cartesianStateWithStm.block( 0, 0, 6, 1 );
    Eigen::MatrixXd stmPartOfStateVectorInMatrixForm = cartesianStateWithStm.block( 0, 1, 6, 6 );

    Eigen::VectorXd differentialCorrection(11);
    Eigen::VectorXd corrections(3);
    Eigen::VectorXd accelerations(2);
    Eigen::MatrixXd relevantPartStm(1,3);
    Eigen::MatrixXd rewrittenOrbitalPeriod(2,3);
    Eigen::MatrixXd updateMatrixIntermediate(2,3);
    Eigen::MatrixXd updateMatrix(2,3);
    Eigen::VectorXd multiplicationMatrix(2);

    Eigen::VectorXd correctionsSYMM(3);
    Eigen::VectorXd accelerationsSYMM(3);
    Eigen::MatrixXd relevantPartStmSYMM(1,3);
    Eigen::MatrixXd rewrittenOrbitalPeriodSYMM(3,3);
    Eigen::MatrixXd updateMatrixIntermediateSYMM(3,3);
    Eigen::MatrixXd updateMatrixSYMM(3,3);
    Eigen::VectorXd multiplicationMatrixSYMM(3);

    // Compute the accelerations and velocities (in X- and Z-direction) on the spacecraft and put them in a 2x1 vector.
    Eigen::VectorXd cartesianAccelerations = computeStateDerivativeAugmented(0.0, cartesianStateWithStm).block(0,0,6,1);

    if (symmetryDependence == true)
    {
        // Set vectors and matrices to zero
        differentialCorrection.setZero();
        corrections.setZero();
        accelerations.setZero();
        relevantPartStm.setZero();
        rewrittenOrbitalPeriod.setZero();
        updateMatrixIntermediate.setZero();
        updateMatrix.setZero();
        multiplicationMatrix.setZero();

        // Calculate deviations at T
        multiplicationMatrix <<  deviationVector(3), deviationVector(5);

        // Compute the update matrix.
        updateMatrixIntermediate << stmPartOfStateVectorInMatrixForm(3, 0), stmPartOfStateVectorInMatrixForm(3, 2), stmPartOfStateVectorInMatrixForm(3, 4),
                                    stmPartOfStateVectorInMatrixForm(5, 0), stmPartOfStateVectorInMatrixForm(5, 2), stmPartOfStateVectorInMatrixForm(5, 4);

        accelerations << cartesianAccelerations(3), cartesianAccelerations(5);


        relevantPartStm << stmPartOfStateVectorInMatrixForm(1,0), stmPartOfStateVectorInMatrixForm(1,2), stmPartOfStateVectorInMatrixForm(1,4);
        rewrittenOrbitalPeriod = (-1.0 / cartesianAccelerations(1) ) * accelerations * relevantPartStm;


        updateMatrix = updateMatrixIntermediate + rewrittenOrbitalPeriod;

        // Compute the necessary differential correction.

        corrections =  updateMatrix.transpose() * (updateMatrix * updateMatrix.transpose()).inverse() * multiplicationMatrix;

        // Put corrections in correct format.
        differentialCorrection.setZero();
        differentialCorrection(0) = corrections(0);
        differentialCorrection(2) = corrections(1);
        differentialCorrection(4) = corrections(2);
        differentialCorrection(10) = deviationVector(10);
    }
    else {

        // Set vectors and matrices to zero
        differentialCorrection.setZero();
        correctionsSYMM.setZero();
        accelerationsSYMM.setZero();
        relevantPartStmSYMM.setZero();
        rewrittenOrbitalPeriodSYMM.setZero();
        updateMatrixIntermediateSYMM.setZero();
        updateMatrixSYMM.setZero();
        multiplicationMatrixSYMM.setZero();

        if (stateIndex == 1 )  {

            //Set the correct multiplication matrix (state deviation at T )
                multiplicationMatrixSYMM << deviationVector(0), deviationVector(3), deviationVector(4);

            // Compute the update matrix.
                updateMatrixIntermediateSYMM << stmPartOfStateVectorInMatrixForm(0, 0), stmPartOfStateVectorInMatrixForm(0, 3), stmPartOfStateVectorInMatrixForm(0, 4),
                                                stmPartOfStateVectorInMatrixForm(3, 0), stmPartOfStateVectorInMatrixForm(3, 3), stmPartOfStateVectorInMatrixForm(3, 4),
                                                stmPartOfStateVectorInMatrixForm(4, 0), stmPartOfStateVectorInMatrixForm(4, 3), stmPartOfStateVectorInMatrixForm(4, 4);

                accelerationsSYMM << cartesianAccelerations(0), cartesianAccelerations(3), cartesianAccelerations(4);

                relevantPartStmSYMM << stmPartOfStateVectorInMatrixForm(1,0), stmPartOfStateVectorInMatrixForm(1,3), stmPartOfStateVectorInMatrixForm(1,4);

                rewrittenOrbitalPeriodSYMM = (-1.0 / cartesianAccelerations(1) ) * accelerationsSYMM * relevantPartStmSYMM;

                updateMatrixSYMM = updateMatrixIntermediateSYMM + rewrittenOrbitalPeriodSYMM;

           // Compute the necessary differential correction.
                correctionsSYMM =  updateMatrixSYMM.inverse() * multiplicationMatrixSYMM;

                differentialCorrection(0) = correctionsSYMM(0);
                differentialCorrection(3) = correctionsSYMM(1);
                differentialCorrection(4) = correctionsSYMM(2);
                differentialCorrection(10) = deviationVector(10);

        } else {

            //Set the correct multiplication matrix (state deviation at T )
                multiplicationMatrixSYMM << deviationVector(1), deviationVector(3), deviationVector(4);

            // Compute the update matrix.
                updateMatrixIntermediateSYMM << stmPartOfStateVectorInMatrixForm(1, 1), stmPartOfStateVectorInMatrixForm(1, 3), stmPartOfStateVectorInMatrixForm(1, 4),
                                                stmPartOfStateVectorInMatrixForm(3, 1), stmPartOfStateVectorInMatrixForm(3, 3), stmPartOfStateVectorInMatrixForm(3, 4),
                                                stmPartOfStateVectorInMatrixForm(4, 1), stmPartOfStateVectorInMatrixForm(4, 3), stmPartOfStateVectorInMatrixForm(4, 4);

                accelerationsSYMM << cartesianAccelerations(0), cartesianAccelerations(3), cartesianAccelerations(4);

                relevantPartStmSYMM << stmPartOfStateVectorInMatrixForm(0,1), stmPartOfStateVectorInMatrixForm(0,3), stmPartOfStateVectorInMatrixForm(0,4);

                rewrittenOrbitalPeriodSYMM = (-1.0 / cartesianAccelerations(0) ) * accelerationsSYMM * relevantPartStmSYMM;

                updateMatrixSYMM = updateMatrixIntermediateSYMM + rewrittenOrbitalPeriodSYMM;

           // Compute the necessary differential correction.
                correctionsSYMM =  updateMatrixSYMM.inverse() * multiplicationMatrixSYMM;

                differentialCorrection(1) = correctionsSYMM(0);
                differentialCorrection(3) = correctionsSYMM(1);
                differentialCorrection(4) = correctionsSYMM(2);
                differentialCorrection(10) = deviationVector(10);

        }
    }

//        std::cout << "============= Compute Differential Correction Matrices ========= " << std::endl
//                  << "multiplicationMatrix: \n" << multiplicationMatrix << std::endl
//                  << "updateMatrix: \n" << updateMatrix << std::endl
//                 << "updateMatrix.inverse(): \n" << updateMatrix.transpose() * ( updateMatrix * updateMatrix.transpose()).inverse() << std::endl
//                 << "corrections: \n"  << corrections << std::endl
//                  << "differentialCorrection: \n" << differentialCorrection << std::endl;
//        std::cout << "================================================================ " << std::endl;

//        std::cout << "============= Compute Differential Correction Matrices ========= " << std::endl
//                  << "multiplicationMatrix: \n" << multiplicationMatrixSYMM << std::endl
//                  << "updateMatrix: \n" << updateMatrixSYMM << std::endl
//                 << "updateMatrix.inverse(): \n" << updateMatrixSYMM.transpose() * ( updateMatrixSYMM * updateMatrixSYMM.transpose()).inverse() << std::endl
//                 << "corrections: \n"  << correctionsSYMM << std::endl
//                  << "differentialCorrection: \n" << differentialCorrection << std::endl;
//        std::cout << "================================================================ " << std::endl;

    return differentialCorrection;
}
