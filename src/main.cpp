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
//#include <omp.h>



double massParameter;

int main (){

    // Read initial conditions from file
    std::ifstream textFileInitialConditions("../data/raw/horizontal_L2_initial_conditions.txt");
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

    // Compute manifolds
//    #pragma omp parallel num_threads(30)
    {
//        #pragma omp for
//        for (unsigned int i = 0; i <= initialConditions.size(); i++) {
        for (unsigned int i = 1066; i <= 1066; i++) {
            double orbitalPeriod = initialConditions[i][1];

            Eigen::VectorXd initialStateVector = Eigen::VectorXd::Zero(6);
            initialStateVector(0) = initialConditions[i][2];
            initialStateVector(1) = initialConditions[i][3];
            initialStateVector(2) = initialConditions[i][4];
            initialStateVector(3) = initialConditions[i][5];
            initialStateVector(4) = initialConditions[i][6];
            initialStateVector(5) = initialConditions[i][7];

            std::string selected_orbit = "L1_horizontal_" + std::to_string(i);

            std::cout                                                                                 << std::endl;
            std::cout << "=================================================================="         << std::endl;
            std::cout << "                          " << selected_orbit << "                        " << std::endl;
            std::cout << "=================================================================="         << std::endl;

            computeManifolds(initialStateVector, orbitalPeriod, 1, "horizontal", i);
        }
    }

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
//                createInitialConditions(1, "vertical");
//            }
//            if (i ==4){
//                createInitialConditions(2, "vertical");
//            }
//            if (i ==5){
//                createInitialConditions(1, "halo");
//            }
//            if (i ==6){
//                createInitialConditions(2, "halo");
//            }
//        }
//    }

    return 0;
}
