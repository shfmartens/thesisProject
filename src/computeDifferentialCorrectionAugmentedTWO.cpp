#include <Eigen/Core>
#include <Eigen/LU>
#include <iostream>

#include "computeDifferentialCorrectionAugmented.h"
#include "stateDerivativeModelAugmented.h"



Eigen::VectorXd computeDifferentialCorrectionAugmented(const Eigen::MatrixXd& cartesianStateWithStm, const Eigen::VectorXd& deviationVector)
{
//    // Initiate vectors, matrices etc.
    Eigen::VectorXd cartesianState = cartesianStateWithStm.block( 0, 0, 6, 1 );
    Eigen::MatrixXd stmPart = cartesianStateWithStm.block( 0, 1, 6, 6 );

    Eigen::VectorXd corrections(6);
    Eigen::VectorXd deviationVector2d(4);
    Eigen::VectorXd differentialCorrection(11);
    Eigen::VectorXd correctionVector(6);
    Eigen::MatrixXd jacobianMatrix(6,6);
    Eigen::MatrixXd jacobianMatrix2d(4,4);
    Eigen::VectorXd constraintVector(6);
    Eigen::VectorXd constraintVector2d(4);

    Eigen::MatrixXd::Index minRow, minCol;
    Eigen::MatrixXd::Index minRow2d, minCol2d;

    std::cout << "Print deviation Vector: " << deviationVector << std::endl;
    double periodCorrection = deviationVector(6);

    // Determine which direction the deviation in etiher position or velocity is smallest
    double minimumDeviationValue = deviationVector.segment(0,6).cwiseAbs().minCoeff(&minRow,&minCol);
    double minimumDeviationValue2d = deviationVector2d.segment(0,6).cwiseAbs().minCoeff(&minRow2d,&minCol2d);
    deviationVector2d.segment(0,2) = deviationVector.segment(0,2);
    deviationVector2d.segment(2,2) = deviationVector.segment(3,2);


    // Compute the accelerations and velocities (in X- and Z-direction) on the spacecraft and put them in a 6x1 vector.
    Eigen::VectorXd cartesianAccelerations = computeStateDerivativeAugmented(0.0, cartesianStateWithStm).block(0,0,6,1);
    Eigen::VectorXd cartesianAccelerations2d(4);
    cartesianAccelerations2d.segment(0,2) = cartesianAccelerations.segment(0,2);
    cartesianAccelerations2d.segment(2,2) = cartesianAccelerations.segment(3,2);

    constraintVector << deviationVector.segment(0,6);
    constraintVector2d << deviationVector2d;

    Eigen::VectorXd updateVector(4);
    updateVector = constraintVector2d - cartesianAccelerations2d*deviationVector(6);

    jacobianMatrix2d << stmPart(0,0), stmPart(0,1), stmPart(3,0), stmPart(4,0),
                        stmPart(1,0), stmPart(1,1), stmPart(3,1), stmPart(4,1),
                        stmPart(3,0), stmPart(3,1), stmPart(3,3), stmPart(4,3),
                        stmPart(4,0), stmPart(4,1), stmPart(3,4), stmPart(4,4);

    corrections = jacobianMatrix2d.inverse() * updateVector;

    differentialCorrection.setZero();
    differentialCorrection(0) = -corrections(0);
    differentialCorrection(1) = -corrections(1);
    differentialCorrection(3) = -corrections(2);
    differentialCorrection(4) = -corrections(3);
    differentialCorrection(10) = deviationVector(6);




//    if (minRow2d == 0) {
//        std::cout << "\n Keeping x-position constant \n" << std::endl;
//        std::cout << " Deviation Vector 2d \n" << deviationVector2d << std::endl;

//        jacobianMatrix2d << stmPart(0,1), stmPart(0,3), stmPart(0,4), cartesianAccelerations(0),
//                          stmPart(1,1), stmPart(1,3), stmPart(0,4), cartesianAccelerations(1),
//                          stmPart(3,1), stmPart(3,3), stmPart(0,4), cartesianAccelerations(3),
//                          stmPart(4,1), stmPart(0,3), stmPart(0,4), cartesianAccelerations(4);


//        std::cout << " DET jacobian Matrix2d \n" << jacobianMatrix2d.determinant() << std::endl;
//        std::cout << " jacobian Matrix2d \n" << jacobianMatrix << std::endl;

//        corrections = jacobianMatrix2d.inverse() * constraintVector2d;

//        differentialCorrection.setZero();
//        differentialCorrection(1) = -corrections(0);
//        differentialCorrection(3) = -corrections(1);
//        differentialCorrection(4) = -corrections(2);
//        differentialCorrection(10) = -corrections(3);


//    } else if (minRow2d == 1) {
//        std::cout << "\n Keeping y-position constant \n" << std::endl;
//        std::cout << " Deviation Vector 2d \n" << deviationVector2d << std::endl;

//        jacobianMatrix2d << stmPart(0,0), stmPart(0,3), stmPart(0,4), cartesianAccelerations(0),
//                          stmPart(1,0), stmPart(1,3), stmPart(0,4), cartesianAccelerations(1),
//                          stmPart(3,0), stmPart(3,3), stmPart(0,4), cartesianAccelerations(3),
//                          stmPart(4,0), stmPart(0,3), stmPart(0,4), cartesianAccelerations(4);

//        std::cout << " DET jacobian Matrix2d \n" << jacobianMatrix2d.determinant() << std::endl;
//        std::cout << " jacobian Matrix2d \n" << jacobianMatrix2d.inverse() << std::endl;

//        corrections = jacobianMatrix2d.inverse() * constraintVector2d;

//        differentialCorrection.setZero();
//        differentialCorrection(0) = -corrections(0);
//        differentialCorrection(3) = -corrections(1);
//        differentialCorrection(4) = -corrections(2);
//        differentialCorrection(10) = -corrections(3);

//    } else if (minRow2d == 2) {
//        std::cout << "\n Keeping x-velocity constant \n" << std::endl;
//        std::cout << " Deviation Vector 2d \n" << deviationVector2d << std::endl;

//        jacobianMatrix2d << stmPart(0,0), stmPart(0,1), stmPart(0,4), cartesianAccelerations(0),
//                          stmPart(1,0), stmPart(1,1), stmPart(0,4), cartesianAccelerations(1),
//                          stmPart(3,0), stmPart(3,1), stmPart(0,4), cartesianAccelerations(3),
//                          stmPart(4,0), stmPart(0,1), stmPart(0,4), cartesianAccelerations(4);

//        std::cout << " DET jacobian Matrix2d \n" << jacobianMatrix2d.determinant() << std::endl;
//        std::cout << " jacobian Matrix2d \n" << jacobianMatrix2d.inverse() << std::endl;

//        corrections = jacobianMatrix2d.inverse() * constraintVector2d;

//        differentialCorrection.setZero();
//        differentialCorrection(0) = -corrections(0);
//        differentialCorrection(1) = -corrections(1);
//        differentialCorrection(4) = -corrections(2);
//        differentialCorrection(10) = -corrections(3);

//    } else {
//        std::cout << "\n Keeping y-velocity constant \n" << std::endl;
//        std::cout << " Deviation Vector 2d \n" << deviationVector2d << std::endl;

//        jacobianMatrix2d << stmPart(0,0), stmPart(0,1), stmPart(0,4), cartesianAccelerations(0),
//                          stmPart(1,0), stmPart(1,1), stmPart(0,4), cartesianAccelerations(1),
//                          stmPart(3,0), stmPart(3,1), stmPart(0,4), cartesianAccelerations(3),
//                          stmPart(4,0), stmPart(0,1), stmPart(0,4), cartesianAccelerations(4);

//        std::cout << " DET jacobian Matrix2d \n" << jacobianMatrix2d.determinant() << std::endl;
//        std::cout << " jacobian Matrix2d \n" << jacobianMatrix2d.inverse() << std::endl;

//        corrections = jacobianMatrix2d.inverse() * constraintVector2d;

//        differentialCorrection.setZero();
//        differentialCorrection(0) = -corrections(0);
//        differentialCorrection(1) = -corrections(1);
//        differentialCorrection(3) = -corrections(2);
//        differentialCorrection(10) = -corrections(3);

//    }

//    minRow = 0;

//    if (minRow == 0) {
//        std::cout << "\n Keeping x-position constant \n" << std::endl;
//        jacobianMatrix << stmPart(0,1), stmPart(0,2), stmPart(0,3), stmPart(0,4), stmPart(0,5), cartesianAccelerations(0),
//                          stmPart(1,1), stmPart(1,2), stmPart(1,3), stmPart(1,4), stmPart(1,5), cartesianAccelerations(1),
//                          stmPart(2,1), stmPart(2,2), stmPart(2,3), stmPart(2,4), stmPart(2,5), cartesianAccelerations(2),
//                          stmPart(3,1), stmPart(3,2), stmPart(3,3), stmPart(3,4), stmPart(3,5), cartesianAccelerations(3),
//                          stmPart(4,1), stmPart(4,2), stmPart(4,3), stmPart(4,4), stmPart(4,5), cartesianAccelerations(4),
//                          stmPart(5,1), stmPart(5,2), stmPart(5,3), stmPart(5,4), stmPart(5,5), cartesianAccelerations(5);


//        corrections = jacobianMatrix.inverse() * constraintVector;

//        differentialCorrection.setZero();
//        differentialCorrection(1) = -corrections(0);
//        //differentialCorrection(2) = -corrections(1);
//        differentialCorrection(3) = -corrections(2);
//        differentialCorrection(4) = -corrections(3);
//        //differentialCorrection(5) = -corrections(4);
//        differentialCorrection(10) = -corrections(5);

//    } else if (minRow == 1) {
//        std::cout << "\n Keeping y-position constant \n" << std::endl;
//        jacobianMatrix << stmPart(0,0), stmPart(0,2), stmPart(0,3), stmPart(0,4), stmPart(0,5), cartesianAccelerations(0),
//                          stmPart(1,0), stmPart(1,2), stmPart(1,3), stmPart(1,4), stmPart(1,5), cartesianAccelerations(1),
//                          stmPart(2,0), stmPart(2,2), stmPart(2,3), stmPart(2,4), stmPart(2,5), cartesianAccelerations(2),
//                          stmPart(3,0), stmPart(3,2), stmPart(3,3), stmPart(3,4), stmPart(3,5), cartesianAccelerations(3),
//                          stmPart(4,0), stmPart(4,2), stmPart(4,3), stmPart(4,4), stmPart(4,5), cartesianAccelerations(4),
//                          stmPart(5,0), stmPart(5,2), stmPart(5,3), stmPart(5,4), stmPart(5,5), cartesianAccelerations(5);


//        corrections = jacobianMatrix.inverse() * constraintVector;

//        differentialCorrection.setZero();
//        differentialCorrection(0) = -corrections(0);
//        //differentialCorrection(2) = -corrections(1);
//        differentialCorrection(3) = -corrections(2);
//        differentialCorrection(4) = -corrections(3);
//        //differentialCorrection(5) = -corrections(4);
//        differentialCorrection(10) = -corrections(5);

//    } else if (minRow == 2) {
//        std::cout << "\n Keeping z-position constant \n" << std::endl;
//        jacobianMatrix << stmPart(0,0), stmPart(0,1), stmPart(0,3), stmPart(0,4), stmPart(0,5), cartesianAccelerations(0),
//                          stmPart(1,0), stmPart(1,1), stmPart(1,3), stmPart(1,4), stmPart(1,5), cartesianAccelerations(1),
//                          stmPart(2,0), stmPart(2,1), stmPart(2,3), stmPart(2,4), stmPart(2,5), cartesianAccelerations(2),
//                          stmPart(3,0), stmPart(3,1), stmPart(3,3), stmPart(3,4), stmPart(3,5), cartesianAccelerations(3),
//                          stmPart(4,0), stmPart(4,1), stmPart(4,3), stmPart(4,4), stmPart(4,5), cartesianAccelerations(4),
//                          stmPart(5,0), stmPart(5,1), stmPart(5,3), stmPart(5,4), stmPart(5,5), cartesianAccelerations(5);


//        corrections = jacobianMatrix.inverse() * constraintVector;

//        differentialCorrection.setZero();
//        differentialCorrection(0) = -corrections(0);
//        differentialCorrection(1) = -corrections(1);
//        differentialCorrection(3) = -corrections(2);
//        differentialCorrection(4) = -corrections(3);
//        //differentialCorrection(5) = -corrections(4);
//        differentialCorrection(10) = -corrections(5);

//    } else if (minRow == 3 ) {
//        std::cout << "\n Keeping x-velocity constant \n" << std::endl;
//        jacobianMatrix << stmPart(0,0), stmPart(0,1), stmPart(0,2), stmPart(0,4), stmPart(0,5), cartesianAccelerations(0),
//                          stmPart(1,0), stmPart(1,1), stmPart(1,2), stmPart(1,4), stmPart(1,5), cartesianAccelerations(1),
//                          stmPart(2,0), stmPart(2,1), stmPart(2,2), stmPart(2,4), stmPart(2,5), cartesianAccelerations(2),
//                          stmPart(3,0), stmPart(3,1), stmPart(3,2), stmPart(3,4), stmPart(3,5), cartesianAccelerations(3),
//                          stmPart(4,0), stmPart(4,1), stmPart(4,2), stmPart(4,4), stmPart(4,5), cartesianAccelerations(4),
//                          stmPart(5,0), stmPart(5,1), stmPart(5,2), stmPart(5,4), stmPart(5,5), cartesianAccelerations(5);


//        corrections = jacobianMatrix.inverse() * constraintVector;

//        differentialCorrection.setZero();
//        differentialCorrection(0) = -corrections(0);
//        differentialCorrection(1) = -corrections(1);
//        //differentialCorrection(2) = -corrections(2);
//        differentialCorrection(2) = -corrections(2);
//        differentialCorrection(4) = -corrections(3);
//        differentialCorrection(5) = -corrections(4);
//        //differentialCorrection(5) = -corrections(4);
//        differentialCorrection(10) = -corrections(5);
//    } else if (minRow == 4) {
//        std::cout << "\n Keeping y-velocity constant \n" << std::endl;
//        jacobianMatrix << stmPart(0,0), stmPart(0,1), stmPart(0,2), stmPart(0,3), stmPart(0,5), cartesianAccelerations(0),
//                          stmPart(1,0), stmPart(1,1), stmPart(1,2), stmPart(1,3), stmPart(1,5), cartesianAccelerations(1),
//                          stmPart(2,0), stmPart(2,1), stmPart(2,2), stmPart(2,3), stmPart(2,5), cartesianAccelerations(2),
//                          stmPart(3,0), stmPart(3,1), stmPart(3,2), stmPart(3,3), stmPart(3,5), cartesianAccelerations(3),
//                          stmPart(4,0), stmPart(4,1), stmPart(4,2), stmPart(4,3), stmPart(4,5), cartesianAccelerations(4),
//                          stmPart(5,0), stmPart(5,1), stmPart(5,2), stmPart(5,3), stmPart(5,5), cartesianAccelerations(5);


//        corrections = jacobianMatrix.inverse() * constraintVector;

//        differentialCorrection.setZero();
//        differentialCorrection(0) = -corrections(0);
//        differentialCorrection(1) = -corrections(1);
//        //differentialCorrection(2) = -corrections(2);
//        differentialCorrection(3) = -corrections(3);
//        //differentialCorrection(5) = -corrections(4);
//        differentialCorrection(10) = -corrections(5);

//    } else {
//        std::cout << "\n Keeping z-velocity constant \n" << std::endl;
//        jacobianMatrix << stmPart(0,0), stmPart(0,1), stmPart(0,2), stmPart(0,3), stmPart(0,4), cartesianAccelerations(0),
//                          stmPart(1,0), stmPart(1,1), stmPart(1,2), stmPart(1,3), stmPart(1,4), cartesianAccelerations(1),
//                          stmPart(2,0), stmPart(2,1), stmPart(2,2), stmPart(2,3), stmPart(2,4), cartesianAccelerations(2),
//                          stmPart(3,0), stmPart(3,1), stmPart(3,2), stmPart(3,3), stmPart(3,4), cartesianAccelerations(3),
//                          stmPart(4,0), stmPart(4,1), stmPart(4,2), stmPart(4,3), stmPart(4,4), cartesianAccelerations(4),
//                          stmPart(5,0), stmPart(5,1), stmPart(5,2), stmPart(5,3), stmPart(5,4), cartesianAccelerations(5);


//        corrections = jacobianMatrix.inverse() * constraintVector;

//        differentialCorrection.setZero();
//        differentialCorrection(0) = -corrections(0);
//        differentialCorrection(1) = -corrections(1);
//        //differentialCorrection(2) = -corrections(2);
//        differentialCorrection(3) = -corrections(3);
//        differentialCorrection(4) = -corrections(4);
//        differentialCorrection(10) = -corrections(5);
//    }

    return differentialCorrection;
}
