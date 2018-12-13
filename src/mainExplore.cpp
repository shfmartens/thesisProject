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
#include "Tudat/Astrodynamics/BasicAstrodynamics/celestialBodyConstants.h"
#include "Tudat/Astrodynamics/Gravitation/jacobiEnergy.h"

#include "createInitialConditions.h"
#include "computeManifolds.h"
#include "computeManifoldsAugmented.h"
#include "propagateOrbit.h"
#include "propagateOrbitAugmented.h"
//#include "completeInitialConditionsHaloFamily.h"
//#include "createInitialConditionsAxialFamily.h"
#include "connectManifoldsAtTheta.h"
//#include "omp.h"


double massParameter = tudat::gravitation::circular_restricted_three_body_problem::computeMassParameter( tudat::celestial_body_constants::EARTH_GRAVITATIONAL_PARAMETER, tudat::celestial_body_constants::MOON_GRAVITATIONAL_PARAMETER );
std::string spacecraftName = "DeepSpace";
std::string thrustPointing = "right";

int main (){

    std::cout << "Exploration Started" << std::endl;
//    // ================================
//    // == Compute initial conditions ==
//    // ================================

//    #pragma omp parallel num_threads(6)
//    {
//        #pragma omp for
//        for (unsigned int i=1; i<=6; i++) {
//            if (i ==1)
//            {
//                createInitialConditions(1, "horizontal");
//            }
//            if (i ==2)
//            {
//                createInitialConditions(2, "horizontal");
//            }
//            if (i ==3)
//            {
//                createInitialConditions(1, "halo");
//            }
//            if (i ==4)
//            {
//                createInitialConditions(2, "halo");
//            }
//            if (i ==5)
//            {
//                createInitialConditions(1, "vertical");
//            }
//            if (i ==6)
//            {
//                createInitialConditions(2, "vertical");
//            }
//        }
//    }

    // ================================
    // == Compute manifolds ==
    // ================================
#pragma omp parallel num_threads(6)
{
    #pragma omp for
    for (unsigned int i=0; i<1; i++) {

        std::string orbitType;
        int librationPointNr;
        int orbitIdOne;
        double desiredJacobiEnergy;

        if (i == 0){
            orbitType = "horizontal";
            librationPointNr = 1;
            orbitIdOne = 808;
            desiredJacobiEnergy = 3.05;
        } if (i == 1){
            orbitType = "horizontal";
            librationPointNr = 1;
            orbitIdOne = 577;
            desiredJacobiEnergy = 3.1;
        } if (i == 2){
            orbitType = "horizontal";
            librationPointNr = 1;
            orbitIdOne = 330;
            desiredJacobiEnergy = 3.15;
        } if (i == 3){
            orbitType = "horizontal";
            librationPointNr = 2;
            orbitIdOne = 1066;
            desiredJacobiEnergy = 3.05;
        } if (i == 4){
            orbitType = "horizontal";
            librationPointNr = 2;
            orbitIdOne = 760;
            desiredJacobiEnergy = 3.1;
        } if (i == 5){
            orbitType = "horizontal";
            librationPointNr = 2;
            orbitIdOne = 373;
            desiredJacobiEnergy = 3.15;
        }

            std::cout << "Start refinement Jacobi energy of orbit " << orbitIdOne << std::endl;

            Eigen::VectorXd selectedInitialConditions = readInitialConditionsFromFile(librationPointNr, orbitType, orbitIdOne, orbitIdOne + 1, massParameter);
            Eigen::VectorXd refinedJacobiEnergyResult = refineOrbitJacobiEnergy(librationPointNr, orbitType, desiredJacobiEnergy,
                                                                                selectedInitialConditions.segment(1, 6),
                                                                                selectedInitialConditions(0),
                                                                                selectedInitialConditions.segment(8, 6),
                                                                                selectedInitialConditions(7), massParameter);
            Eigen::VectorXd initialStateVector = refinedJacobiEnergyResult.segment(0, 6);
            double orbitalPeriod               = refinedJacobiEnergyResult(6);

            std::cout << "Jaocbi Energy refined of the following orbit number " << orbitIdOne << std::endl;

            Eigen::MatrixXd fullInitialState = getFullInitialState( initialStateVector );
            std::map< double, Eigen::Vector6d > stateHistory;
            double refinedEnergy = tudat::gravitation::computeJacobiEnergy(massParameter, fullInitialState);
            std::cout << "REFINED JACOBI ENERGY: " << refinedEnergy << std::endl;

            std::cout << "Start propagation to Final condition, following orbit " << orbitIdOne << std::endl;

            std::pair< Eigen::MatrixXd, double > endState = propagateOrbitToFinalCondition( fullInitialState, massParameter, orbitalPeriod, 1, stateHistory, 100, 0.0 );

            std::cout << "Propagation to Final condition completed: orbit number: " << orbitIdOne << std::endl;

            writeStateHistoryToFile( stateHistory, orbitIdOne, orbitType, librationPointNr, 1000, false );

            std::cout << "State history of the orbits written to file in raw/orbits, orbit number: " << orbitIdOne << std::endl;

            // ===============================================================
            // == Compute manifolds based on precomputed initial conditions ==
            // ===============================================================

            std::cout << "start computation of manifolds of orbit number: " << orbitIdOne << std::endl;

            computeManifoldsAugmented(initialStateVector, orbitalPeriod, orbitIdOne, librationPointNr, orbitType, spacecraftName, thrustPointing,  massParameter);

    }

}

    return 0;
}
