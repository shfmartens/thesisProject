#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>
#include <Eigen/Dense>



bool checkEigenvalues( Eigen::VectorXd stateVectorInclSTM, double maxEigenvalueDeviation = 1.0e-3 )
{
    // Initialize variables
    bool eigenvalueRealOne = false;

    // Reshape the STM for one period to matrix form and compute the eigenvalues
    Eigen::Map<Eigen::MatrixXd> monodromyMatrix = Eigen::Map<Eigen::MatrixXd>(stateVectorInclSTM.segment(6,36).data(),6,6);
    Eigen::EigenSolver<Eigen::MatrixXd> eig(monodromyMatrix);

    // Add eigenvalues
    for (int i = 0; i <= 5; i++){
        if (std::abs(eig.eigenvalues().imag()(i)) < maxEigenvalueDeviation){
            if (std::abs(eig.eigenvalues().real()(i) - 1.0) < maxEigenvalueDeviation){
                eigenvalueRealOne = true;
            }
        }
    }

    return eigenvalueRealOne;
}
