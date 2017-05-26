#include <Eigen/Core>
#include <Eigen/QR>
#include <iostream>
#include <Eigen/Eigenvalues>
#include <complex>
#include <vector>

using namespace std;
using namespace Eigen;

// Function to compute the differential correction
Eigen::VectorXd generateManifold(std::vector< std::vector <double> > dataFromFile)
{
    // Declare the initial point of the manifold.
    Eigen::VectorXd manifoldInitialPoint(6);
    manifoldInitialPoint.setZero();

    // Reshape the STM-part of the state vector to 6x6 format.
 /*
    Eigen::VectorXd stmPartOfStateVector = currentState.segment(6,36);
    Eigen::Map<Eigen::MatrixXd> stmPartOfStateVectorInMatrixForm = Eigen::Map<Eigen::MatrixXd>(stmPartOfStateVector.data(),6,6);

    // Compute the eigenvalues and vectors of the STM.
    EigenSolver<MatrixXd> es(stmPartOfStateVectorInMatrixForm);
    VectorXcd tempVector = es.eigenvectors().col(1);
    Eigen::VectorXd smallestEigenVector(6);
    smallestEigenVector << real(tempVector(0)),
            real(tempVector(1)),
            real(tempVector(2)),
            real(tempVector(3)),
            real(tempVector(4)),
            real(tempVector(5));

    // Compute the initial point on the manifold.
    //Eigen::VectorXd correctionApplied = 10e-6*smallestEigenVector;
    manifoldInitialPoint = currentState.segment(0,6) + 10e-3*smallestEigenVector;

    //
*/
    // Return differential correction.
    return manifoldInitialPoint;

}

