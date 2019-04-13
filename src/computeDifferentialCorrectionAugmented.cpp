#include <Eigen/Core>
#include <Eigen/LU>
#include <iostream>

#include "computeDifferentialCorrectionAugmented.h"
#include "stateDerivativeModelAugmented.h"



Eigen::VectorXd computeDifferentialCorrectionAugmented( const Eigen::MatrixXd& cartesianStateWithStm, const Eigen::VectorXd& deviationVector )
{
    // Specifiy run conditions
    bool minimumNorm = false;
    bool symmetryDependence = true;

    // Initiate vectors, matrices etc.
    Eigen::VectorXd cartesianState = cartesianStateWithStm.block( 0, 0, 6, 1 );
    Eigen::MatrixXd stmPartOfStateVectorInMatrixForm = cartesianStateWithStm.block( 0, 1, 6, 6 );
    Eigen::VectorXd differentialCorrection(11);

    // Compute the accelerations and velocities (in X- and Z-direction) on the spacecraft and put them in a 2x1 vector.
    Eigen::VectorXd cartesianAccelerations = computeStateDerivativeAugmented(0.0, cartesianStateWithStm).block(0,0,6,1);

    if( minimumNorm == true)
    {

        // Initiate vectors, matrices etc.
        Eigen::VectorXd correctionsLSQ(3);
        Eigen::VectorXd accelerationsLSQ(2);
        Eigen::MatrixXd relevantPartStmLSQ(1,3);
        Eigen::MatrixXd rewrittenOrbitalPeriodLSQ(2,3);
        Eigen::MatrixXd updateMatrixIntermediateLSQ(2,3);
        Eigen::MatrixXd updateMatrixLSQ(2,3);
        Eigen::VectorXd multiplicationMatrixLSQ(2);

        // Set vectors and matrices to zero
        differentialCorrection.setZero();
        correctionsLSQ.setZero();
        accelerationsLSQ.setZero();
        relevantPartStmLSQ.setZero();
        rewrittenOrbitalPeriodLSQ.setZero();
        updateMatrixIntermediateLSQ.setZero();
        updateMatrixLSQ.setZero();
        multiplicationMatrixLSQ.setZero();

        // Calculate deviations at T
        multiplicationMatrixLSQ <<  deviationVector(3), deviationVector(5);

        // Compute the update matrix.
        updateMatrixIntermediateLSQ << stmPartOfStateVectorInMatrixForm(3, 0), stmPartOfStateVectorInMatrixForm(3, 2), stmPartOfStateVectorInMatrixForm(3, 4),
                                    stmPartOfStateVectorInMatrixForm(5, 0), stmPartOfStateVectorInMatrixForm(5, 2), stmPartOfStateVectorInMatrixForm(5, 4);

        accelerationsLSQ << cartesianAccelerations(3), cartesianAccelerations(5);


        relevantPartStmLSQ << stmPartOfStateVectorInMatrixForm(1,0), stmPartOfStateVectorInMatrixForm(1,2), stmPartOfStateVectorInMatrixForm(1,4);
        rewrittenOrbitalPeriodLSQ = (-1.0 / cartesianAccelerations(1) ) * accelerationsLSQ * relevantPartStmLSQ;


        updateMatrixLSQ = updateMatrixIntermediateLSQ + rewrittenOrbitalPeriodLSQ;

        // Compute the necessary differential correction.

        correctionsLSQ =  updateMatrixLSQ.transpose() * (updateMatrixLSQ * updateMatrixLSQ.transpose()).inverse() * multiplicationMatrixLSQ;

        // Put corrections in correct format.
        differentialCorrection.setZero();
        differentialCorrection(0) = correctionsLSQ(0);
        differentialCorrection(2) = correctionsLSQ(1);
        differentialCorrection(4) = correctionsLSQ(2);
        differentialCorrection(10) = deviationVector(10);

    }
    else
    {
        if (symmetryDependence == true){

            // Initiate vectors, matrices etc.
            Eigen::VectorXd corrections(3);
            Eigen::VectorXd accelerations(3);
            Eigen::MatrixXd updateMatrix(3,3);
            Eigen::VectorXd multiplicationMatrix(3);

            // Set vectors and matrices to zero
            differentialCorrection.setZero();
            corrections.setZero();
            accelerations.setZero();
            updateMatrix.setZero();
            multiplicationMatrix.setZero();


             //Set the correct multiplication matrix (state deviation at T )
             multiplicationMatrix << deviationVector(1), deviationVector(3), deviationVector(5);

             // Compute the update matrix.
             updateMatrix << stmPartOfStateVectorInMatrixForm(1, 0), stmPartOfStateVectorInMatrixForm(1, 2), stmPartOfStateVectorInMatrixForm(1, 4),
                                             stmPartOfStateVectorInMatrixForm(3, 0), stmPartOfStateVectorInMatrixForm(3, 2), stmPartOfStateVectorInMatrixForm(3, 4),
                                             stmPartOfStateVectorInMatrixForm(5, 0), stmPartOfStateVectorInMatrixForm(5, 2), stmPartOfStateVectorInMatrixForm(5, 4);


             // Compute the necessary differential correction.
                        corrections =  updateMatrix.inverse() * multiplicationMatrix;

                        differentialCorrection(0) = corrections(0);
                        differentialCorrection(2) = corrections(1);
                        differentialCorrection(4) = corrections(2);
                        //differentialCorrection(10) = deviationVector(10);

        } else {

            // Initiate vectors, matrices etc.
            Eigen::VectorXd corrections(6);
            Eigen::VectorXd accelerations(6);
            Eigen::MatrixXd updateMatrix(6,6);
            Eigen::VectorXd multiplicationMatrix(6);

            // Set vectors and matrices to zero
            differentialCorrection.setZero();
            corrections.setZero();
            accelerations.setZero();
            updateMatrix.setZero();
            multiplicationMatrix.setZero();

            multiplicationMatrix = deviationVector.segment(0,6);
            updateMatrix = stmPartOfStateVectorInMatrixForm;
            corrections = updateMatrix.inverse() * multiplicationMatrix;

            differentialCorrection(0) = corrections(0);
            differentialCorrection(1) = corrections(1);
            differentialCorrection(2) = corrections(2);
            differentialCorrection(3) = corrections(3);
            differentialCorrection(4) = corrections(4);
            differentialCorrection(5) = corrections(5);
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


    return differentialCorrection;
}
