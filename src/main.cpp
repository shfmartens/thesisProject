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

//    #pragma omp parallel num_threads(6)
//    {
//        #pragma omp for
//        for (unsigned int i=1; i<=6; i++) {
//            if (i ==1){
//                createInitialConditions(1, "horizontal");
//            }
//            if (i ==2){
//                createInitialConditions(2, "horizontal");
//            }
//            if (i ==3){
//                createInitialConditions(1, "halo");
//            }
//            if (i ==4){
//                createInitialConditions(2, "halo");
//            }
//            if (i ==5){
//                createInitialConditions(1, "vertical");
//            }
//            if (i ==6){
//                createInitialConditions(2, "vertical");
//            }
//        }
//    }



    #pragma omp parallel num_threads(6)
    {
        #pragma omp for
        for (unsigned int i=0; i<6; i++) {

            std::string orbitType;
            int librationPointNr;

            if (i == 0){
                orbitType = "horizontal";
                librationPointNr = 1;
            } if (i == 1){
                orbitType = "horizontal";
                librationPointNr = 2;
            } if (i == 2){
                orbitType = "halo";
                librationPointNr = 1;
            } if (i == 3){
                orbitType = "halo";
                librationPointNr = 2;
            } if (i == 4){
                orbitType = "vertical";
                librationPointNr = 1;
            } if (i == 5){
                orbitType = "vertical";
                librationPointNr = 2;
            }
//                    for (unsigned int librationPointNr = 2; librationPointNr <= 2; librationPointNr++) {

            // =========================================
            // == Load precomputed initial conditions ==
            // =========================================

            std::ifstream textFileInitialConditions("../data/raw/orbits/L" + std::to_string(librationPointNr) + "_" + orbitType + "_initial_conditions.txt");
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
            double orbitalPeriod1;
            double orbitalPeriod2;
            Eigen::VectorXd initialStateVector1;
            Eigen::VectorXd initialStateVector2;

            // ==============================================================================================
            // == Complete initial conditions for the halo family, until connection to horizontal Lyapunov ==
            // ==============================================================================================

//            orbitalPeriod1         = initialConditions[2][1];
//            initialStateVector1    = Eigen::VectorXd::Zero(6);
//            initialStateVector1(0) = initialConditions[2][2];
//            initialStateVector1(1) = initialConditions[2][3];
//            initialStateVector1(2) = initialConditions[2][4];
//            initialStateVector1(3) = initialConditions[2][5];
//            initialStateVector1(4) = initialConditions[2][6];
//            initialStateVector1(5) = initialConditions[2][7];
//
//            orbitalPeriod2         = initialConditions[1][1];
//            initialStateVector2    = Eigen::VectorXd::Zero(6);
//            initialStateVector2(0) = initialConditions[1][2];
//            initialStateVector2(1) = initialConditions[1][3];
//            initialStateVector2(2) = initialConditions[1][4];
//            initialStateVector2(3) = initialConditions[1][5];
//            initialStateVector2(4) = initialConditions[1][6];
//            initialStateVector2(5) = initialConditions[1][7];
//
//            completeInitialConditionsHaloFamily( initialStateVector1, initialStateVector2, orbitalPeriod1, orbitalPeriod2, librationPointNr);

            // ==================================================================================================================
            // == Create initial conditions for the axial family, based on the bifurcation from the horizontal Lyapunov family ==
            // ==================================================================================================================

            // Create initial conditions axial family
//            int orbitIdForBifurcationToAxial;
//            double offsetForBifurcationToAxial1;
//            double offsetForBifurcationToAxial2;
//            if (librationPointNr == 1) {
//                orbitIdForBifurcationToAxial = 938;  // Indices for L1 bifurcations: [181, 938, 1233]
//                offsetForBifurcationToAxial1 = -0.01;
//                offsetForBifurcationToAxial2 = -0.02;
//            } else {
//                orbitIdForBifurcationToAxial = 1262;  // Indices for L2 bifurcations: [353, 1262, 1514]
//                offsetForBifurcationToAxial1 = -8.170178408168770e-04;
//                offsetForBifurcationToAxial2 = -1.817017673845933e-03;
//            }
//
//            orbitalPeriod1         = initialConditions[orbitIdForBifurcationToAxial][1];
//            initialStateVector1    = Eigen::VectorXd::Zero(6);
//            initialStateVector1(0) = initialConditions[orbitIdForBifurcationToAxial][2];
//            initialStateVector1(1) = initialConditions[orbitIdForBifurcationToAxial][3];
//            initialStateVector1(2) = initialConditions[orbitIdForBifurcationToAxial][4];
//            initialStateVector1(3) = initialConditions[orbitIdForBifurcationToAxial][5];
//            initialStateVector1(4) = initialConditions[orbitIdForBifurcationToAxial][6];
//            initialStateVector1(5) = initialConditions[orbitIdForBifurcationToAxial][7] + offsetForBifurcationToAxial1;
//            Eigen::VectorXd stateVectorInclSTM;
//            stateVectorInclSTM     = writePeriodicOrbitToFile( initialStateVector1, librationPointNr, "axial", 0, orbitalPeriod1, massParameter);
//
//            orbitalPeriod2         = initialConditions[orbitIdForBifurcationToAxial][1];
//            initialStateVector2    = Eigen::VectorXd::Zero(6);
//            initialStateVector2(0) = initialConditions[orbitIdForBifurcationToAxial][2];
//            initialStateVector2(1) = initialConditions[orbitIdForBifurcationToAxial][3];
//            initialStateVector2(2) = initialConditions[orbitIdForBifurcationToAxial][4];
//            initialStateVector2(3) = initialConditions[orbitIdForBifurcationToAxial][5];
//            initialStateVector2(4) = initialConditions[orbitIdForBifurcationToAxial][6];
//            initialStateVector2(5) = initialConditions[orbitIdForBifurcationToAxial][7] + offsetForBifurcationToAxial2;
//
//            stateVectorInclSTM     = writePeriodicOrbitToFile( initialStateVector2, librationPointNr, "axial", 1, orbitalPeriod2, massParameter);
//
//            createInitialConditionsAxialFamily(initialStateVector1, initialStateVector2, orbitalPeriod1, orbitalPeriod2, librationPointNr);

            // ===============================================================
            // == Compute manifolds based on precomputed initial conditions ==
            // ===============================================================

            int orbitIdForManifold;
            if (orbitType == "horizontal") {
                if (librationPointNr == 1) {
//                    orbitIdForManifold = 808;  // C = 3.05
//                    orbitIdForManifold = 577;  // C = 3.1
                    orbitIdForManifold = 330;  // C = 3.15
                } else {
//                    orbitIdForManifold = 1066;  // C = 3.05
//                    orbitIdForManifold = 760;  // C = 3.1
                    orbitIdForManifold = 373;  // C = 3.15
                }
            } if (orbitType == "halo") {
                if (librationPointNr == 1) {
//                    orbitIdForManifold = 1235;  // C = 3.05
//                    orbitIdForManifold = 836;  // C = 3.1
                    orbitIdForManifold = 358;  // C = 3.15
                } else {
//                    orbitIdForManifold = 1093;  // C = 3.05
//                    orbitIdForManifold = 651;  // C = 3.1
                    orbitIdForManifold = 0;  // C = 3.15
                }
            } if (orbitType == "vertical") {
                if (librationPointNr == 1) {
//                    orbitIdForManifold = 1664;  // C = 3.05
//                    orbitIdForManifold = 1159;  // C = 3.1
                    orbitIdForManifold = 600;  // C = 3.15
                } else {
//                    orbitIdForManifold = 1878;  // C = 3.05
//                    orbitIdForManifold = 1275;  // C = 3.1
                    orbitIdForManifold = 513;  // C = 3.15
                }
            }

            double orbitalPeriod               = initialConditions[orbitIdForManifold][1];
            Eigen::VectorXd initialStateVector = Eigen::VectorXd::Zero(6);
            initialStateVector(0) = initialConditions[orbitIdForManifold][2];
            initialStateVector(1) = initialConditions[orbitIdForManifold][3];
            initialStateVector(2) = initialConditions[orbitIdForManifold][4];
            initialStateVector(3) = initialConditions[orbitIdForManifold][5];
            initialStateVector(4) = initialConditions[orbitIdForManifold][6];
            initialStateVector(5) = initialConditions[orbitIdForManifold][7];

            std::string selected_orbit = "L" + std::to_string(librationPointNr) + "_" + orbitType + "_W_" + std::to_string(orbitIdForManifold);
            std::cout                                                                                 << std::endl;
            std::cout << "=================================================================="         << std::endl;
            std::cout << "                          " << selected_orbit << "                        " << std::endl;
            std::cout << "=================================================================="         << std::endl;

            computeManifolds(initialStateVector, orbitalPeriod, librationPointNr, orbitType, orbitIdForManifold);
        }
    }

    return 0;
}
