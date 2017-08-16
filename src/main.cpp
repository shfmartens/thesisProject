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
#include "createInitialConditions.h"
#include "computeManifolds.h"
#include "completeInitialConditionsHaloFamily.h"
#include "createInitialConditionsAxialFamily.h"
#include <omp.h>



double massParameter;


int main (){

    // ================================
    // == Compute initial conditions ==
    // ================================

    #pragma omp parallel num_threads(6)
    {
        #pragma omp for
        for (unsigned int i=1; i<=6; i++) {
            if (i ==1){
                createInitialConditions(1, "horizontal");
            }
            if (i ==2){
                createInitialConditions(2, "horizontal");
            }
            if (i ==3){
                createInitialConditions(1, "vertical");
            }
            if (i ==4){
                createInitialConditions(2, "vertical");
            }
            if (i ==5){
                createInitialConditions(1, "halo");
            }
            if (i ==6){
                createInitialConditions(2, "halo");
            }
        }
    }


    #pragma omp parallel num_threads(2)
    {
        #pragma omp for
        for (unsigned int librationPointNr = 2; librationPointNr <= 2; librationPointNr++) {

            // =========================================
            // == Load precomputed initial conditions ==
            // =========================================

            std::ifstream textFileInitialConditions(
                    "../data/raw/L" + std::to_string(librationPointNr) + "_horizontal_initial_conditions.txt");
            std::vector<std::vector<double>> initialConditions;

            if (textFileInitialConditions) {
                std::string line;

                while (std::getline(textFileInitialConditions, line)) {
                    initialConditions.push_back(std::vector<double>());

                    // Break down the row into column values
                    std::stringstream split(line);
                    double value;

                    while (split >> value)
                        initialConditions.back().push_back(value);
                }
            }

            // ==============================================================================================
            // == Complete initial conditions for the halo family, until connection to horizontal Lyapunov ==
            // ==============================================================================================

            double orbitalPeriod1 = initialConditions[2][1];
            Eigen::VectorXd initialStateVector1 = Eigen::VectorXd::Zero(6);
            initialStateVector1(0) = initialConditions[2][2];
            initialStateVector1(1) = initialConditions[2][3];
            initialStateVector1(2) = initialConditions[2][4];
            initialStateVector1(3) = initialConditions[2][5];
            initialStateVector1(4) = initialConditions[2][6];
            initialStateVector1(5) = initialConditions[2][7];

            double orbitalPeriod2 = initialConditions[1][1];
            Eigen::VectorXd initialStateVector2 = Eigen::VectorXd::Zero(6);
            initialStateVector2(0) = initialConditions[1][2];
            initialStateVector2(1) = initialConditions[1][3];
            initialStateVector2(2) = initialConditions[1][4];
            initialStateVector2(3) = initialConditions[1][5];
            initialStateVector2(4) = initialConditions[1][6];
            initialStateVector2(5) = initialConditions[1][7];

            completeInitialConditionsHaloFamily( initialStateVector1, initialStateVector2, orbitalPeriod1, orbitalPeriod2, librationPointNr);

            // ==================================================================================================================
            // == Create initial conditions for the axial family, based on the bifurcation from the horizontal Lyapunov family ==
            // ==================================================================================================================

            // Create initial conditions axial family
//            int orbitIdForBifurcationToAxial;
//            double signZdot;
//            if (librationPointNr == 1) {
//                orbitIdForBifurcationToAxial = 938;  // For L1
//                signZdot = -1.0;
//            } else {
////                orbitIdForBifurcationToAxial = 353;  // For L2
//                signZdot = 1.0;
//            }
//
//            double orbitalPeriod1 = initialConditions[orbitIdForBifurcationToAxial][1];
//            Eigen::VectorXd initialStateVector1 = Eigen::VectorXd::Zero(6);
//            initialStateVector1(0) = initialConditions[orbitIdForBifurcationToAxial][2];
//            initialStateVector1(1) = initialConditions[orbitIdForBifurcationToAxial][3];
//            initialStateVector1(2) = initialConditions[orbitIdForBifurcationToAxial][4];
//            initialStateVector1(3) = initialConditions[orbitIdForBifurcationToAxial][5];
//            initialStateVector1(4) = initialConditions[orbitIdForBifurcationToAxial][6];
//            initialStateVector1(5) = initialConditions[orbitIdForBifurcationToAxial][7] + signZdot * 0.01;
//            Eigen::VectorXd stateVectorInclSTM;
//            stateVectorInclSTM = writePeriodicOrbitToFile( initialStateVector1, librationPointNr, "axial", 0, orbitalPeriod1, massParameter);
//
//            double orbitalPeriod2 = initialConditions[orbitIdForBifurcationToAxial][1];
//            Eigen::VectorXd initialStateVector2 = Eigen::VectorXd::Zero(6);
//            initialStateVector2(0) = initialConditions[orbitIdForBifurcationToAxial][2];
//            initialStateVector2(1) = initialConditions[orbitIdForBifurcationToAxial][3];
//            initialStateVector2(2) = initialConditions[orbitIdForBifurcationToAxial][4];
//            initialStateVector2(3) = initialConditions[orbitIdForBifurcationToAxial][5];
//            initialStateVector2(4) = initialConditions[orbitIdForBifurcationToAxial][6];
//            initialStateVector2(5) = initialConditions[orbitIdForBifurcationToAxial][7] + signZdot * 0.02;
//
//            stateVectorInclSTM = writePeriodicOrbitToFile( initialStateVector2, librationPointNr, "axial", 1, orbitalPeriod2, massParameter);
//
//            createInitialConditionsAxialFamily(initialStateVector1, initialStateVector2, orbitalPeriod1, orbitalPeriod2, librationPointNr);
//
        }
    }

    // ===============================================================
    // == Compute manifolds based on precomputed initial conditions ==
    // ===============================================================

//    #pragma omp parallel num_threads(14)
//    {
//        #pragma omp for
//        for (unsigned int i = 0; i <= initialConditions.size(); i++) {
//        for (unsigned int i = 501; i <= 501; i++) {
//            double orbitalPeriod = initialConditions[i][1];
//
//            Eigen::VectorXd initialStateVector = Eigen::VectorXd::Zero(6);
//            initialStateVector(0) = initialConditions[i][2];
//            initialStateVector(1) = initialConditions[i][3];
//            initialStateVector(2) = initialConditions[i][4];
//            initialStateVector(3) = initialConditions[i][5];
//            initialStateVector(4) = initialConditions[i][6];
//            initialStateVector(5) = initialConditions[i][7];
//
//            std::string selected_orbit = "L1_horizontal_" + std::to_string(i);
//
//            std::cout                                                                                 << std::endl;
//            std::cout << "=================================================================="         << std::endl;
//            std::cout << "                          " << selected_orbit << "                        " << std::endl;
//            std::cout << "=================================================================="         << std::endl;
//
//            computeManifolds(initialStateVector, orbitalPeriod, 1, "horizontal", i);
//        }
//    }

    return 0;
}
