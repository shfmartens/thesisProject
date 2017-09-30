#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>
#include <Eigen/Dense>



std::vector<double> computeEigenvalues( const Eigen::VectorXd& stateVectorInclSTM )
{
    // Initialize variables
    std::vector<double> eigenvalues;

    // Reshape the STM for one period to matrix form and compute the eigenvalues
    Eigen::Map<Eigen::MatrixXd> monodromyMatrix = Eigen::Map<Eigen::MatrixXd>(stateVectorInclSTM.segment(6,36).data(),6,6);
    Eigen::EigenSolver<Eigen::MatrixXd> eig(monodromyMatrix);

    // Add eigenvalues
    for (int i = 0; i <= 5; i++){
        eigenvalues.push_back(eig.eigenvalues().real()(i));
        eigenvalues.push_back(eig.eigenvalues().imag()(i));
    }

    return eigenvalues;
}
