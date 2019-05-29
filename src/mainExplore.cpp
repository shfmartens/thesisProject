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

#include "createInitialConditions.h"
#include "createLowThrustInitialConditions.h"
#include "computeManifolds.h"
#include "propagateOrbit.h"
//#include "completeInitialConditionsHaloFamily.h"
//#include "createInitialConditionsAxialFamily.h"
#include "connectManifoldsAtTheta.h"
#include "createEquilibriumLocations.h"
#include "omp.h"


double massParameter = tudat::gravitation::circular_restricted_three_body_problem::computeMassParameter( tudat::celestial_body_constants::EARTH_GRAVITATIONAL_PARAMETER, tudat::celestial_body_constants::MOON_GRAVITATIONAL_PARAMETER );

int main (){
    // ================================
    // == Compute equilibria, comment out when computing low-thrust intial positions ==
    // ================================
        //createEquilibriumLocations(1, thrustAcceleration, massParameter);
        //createEquilibriumLocations(2, thrustAcceleration, massParameter);

    // ================================
    // == Compute initial conditions ==
    // ================================
        //std::string orbitType = "horizontal";
        //int continuationIndex = 1; //1: Continuate for H, 8: acceleration, 9: alpha, 10: beta
        //double accelerationMagnitude = 0.01;
        //double accelerationAngle = 0.0;
        //double accelerationAngle2 = 0.0;
        //double initialMass = 1.0;
        //createLowThrustInitialConditions(1, orbitType, continuationIndex, accelerationMagnitude, accelerationAngle, accelerationAngle2, initialMass, massParameter );


    // ================================
    // == Compute initial conditions ==
    // ================================


    #pragma omp parallel num_threads(14)
    {
        #pragma omp for
        for (unsigned int i=1; i<=14; i++) {
            if (i ==1)
            {

                createLowThrustInitialConditions(1, "horizontal", 1, 0.00000, 0.0, 0.0, 1.0, -1.552, massParameter );

            if (i ==2)
            {

                createLowThrustInitialConditions(1, "horizontal", 1, 0.0001, 0.0, 0.0, 1.0, -1.552, massParameter );
            }
            if (i ==3)
            {

                createLowThrustInitialConditions(1, "horizontal", 1, 0.0005, 0.0, 0.0, 1.0, -1.552, massParameter );
            }
            if (i ==4)
            {

                createLowThrustInitialConditions(1, "horizontal", 1, 0.001, 0.0, 0.0, 1.0, -1.552, massParameter );
            }
            if (i ==5)
            {

                createLowThrustInitialConditions(1, "horizontal", 1, 0.005, 0.0, 0.0, 1.0, -1.552, massParameter );
            }
            if (i ==6)
           {

                createLowThrustInitialConditions(1, "horizontal", 1, 0.01, 0.0, 0.0, 1.0, -1.552, massParameter );
            }

            if (i ==7)
           {

                createLowThrustInitialConditions(1, "horizontal", 1, 0.07, 0.0, 0.0, 1.0, -1.552, massParameter );
            }

            if (i ==8)
            {

                createLowThrustInitialConditions(2, "horizontal", 1, 0.00000, 0.0, 0.0, 1.0, -1.552, massParameter );
            }
            if (i ==9)
            {

                createLowThrustInitialConditions(2, "horizontal", 1, 0.0001, 0.0, 0.0, 1.0, -1.552, massParameter );
            }
            if (i ==10)
            {

                createLowThrustInitialConditions(2, "horizontal", 1, 0.0005, 0.0, 0.0, 1.0, -1.552, massParameter );
            }
            if (i ==11)
            {

                createLowThrustInitialConditions(2, "horizontal", 1, 0.001, 0.0, 0.0, 1.0, -1.552, massParameter );
            }
            if (i ==12)
            {

                createLowThrustInitialConditions(2, "horizontal", 1, 0.005, 0.0, 0.0, 1.0, -1.552, massParameter );
            }
            if (i ==13)
           {

                createLowThrustInitialConditions(2, "horizontal", 1, 0.01, 0.0, 0.0, 1.0, -1.552, massParameter );
            }

            if (i ==14)
           {

                createLowThrustInitialConditions(2, "horizontal", 1, 0.07, 0.0, 0.0, 1.0, -1.552, massParameter );
            }


        }
    }
 }

    // ================================
    // == Compute manifolds ==
    // ================================
//    #pragma omp parallel num_threads(12)
//    {
//        #pragma omp for
//        for (unsigned int i=6; i<18; i++) {

//            std::string orbitType;
//            int librationPointNr;
//            int orbitIdOne;
//            double desiredJacobiEnergy;

//            if (i == 6){
//                orbitType = "halo";
//                librationPointNr = 1;
//                orbitIdOne = 1235;
//                desiredJacobiEnergy = 3.05;
//            } if (i == 7){
//                 orbitType = "halo";
//                 librationPointNr = 1;
//                 orbitIdOne = 836;
//                 desiredJacobiEnergy = 3.1;
//            } if (i == 8){
//                orbitType = "halo";
//                librationPointNr = 1;
//                orbitIdOne = 358;
//                desiredJacobiEnergy = 3.15;
//            } if (i == 9){
//                orbitType = "halo";
//                librationPointNr = 2;
//                orbitIdOne = 1093;
//                desiredJacobiEnergy = 3.05;
//            } if (i == 10){
//                orbitType = "halo";
//                librationPointNr = 2;
//                orbitIdOne = 651;
//                desiredJacobiEnergy = 3.1;
//            } if (i == 11){
//              orbitType = "halo";
//              librationPointNr = 2;
//              orbitIdOne = 0;
//              desiredJacobiEnergy = 3.15;
//            } if (i == 12){
//              orbitType = "vertical";
//              librationPointNr = 1;
//              orbitIdOne = 1664;
//              desiredJacobiEnergy = 3.05;
//             } if (i == 13){
//               orbitType = "vertical";
//               librationPointNr = 1;
//               orbitIdOne = 1159;
//               desiredJacobiEnergy = 3.1;
//             } if (i == 14){
//               orbitType = "vertical";
//               librationPointNr = 1;
//               orbitIdOne = 600;
//               desiredJacobiEnergy = 3.15;
//             } if (i == 15){
//               orbitType = "vertical";
//               librationPointNr = 2;
//               orbitIdOne = 1878;
//               desiredJacobiEnergy = 3.05;
//             } if (i == 16){
//               orbitType = "vertical";
//               librationPointNr = 2;
//               orbitIdOne = 1275;
//               desiredJacobiEnergy = 3.1;
//             } if (i == 17){
//               orbitType = "vertical";
//               librationPointNr = 2;
//               orbitIdOne = 513;
//               desiredJacobiEnergy = 3.15;
//             }

//            std::cout << "Start refinement Jacobi energy of orbit " << orbitIdOne << std::endl;

//            Eigen::VectorXd selectedInitialConditions = readInitialConditionsFromFile(librationPointNr, orbitType, orbitIdOne, orbitIdOne + 1, massParameter);
//            Eigen::VectorXd refinedJacobiEnergyResult = refineOrbitJacobiEnergy(librationPointNr, orbitType, desiredJacobiEnergy,
//                                                                                selectedInitialConditions.segment(1, 6),
//                                                                                selectedInitialConditions(0),
//                                                                                selectedInitialConditions.segment(8, 6),
//                                                                                selectedInitialConditions(7), massParameter);
//            Eigen::VectorXd initialStateVector = refinedJacobiEnergyResult.segment(0, 6);
//            double orbitalPeriod               = refinedJacobiEnergyResult(6);

//            std::cout << "Jaocbi Energy refined of the following orbit number " << orbitIdOne << std::endl;

//            Eigen::MatrixXd fullInitialState = getFullInitialState( initialStateVector );
//            std::map< double, Eigen::Vector6d > stateHistory;

//            std::cout << "Start propagation to Final condition, following orbit " << orbitIdOne << std::endl;

//            std::pair< Eigen::MatrixXd, double > endState = propagateOrbitToFinalCondition( fullInitialState, massParameter, orbitalPeriod, 1, stateHistory, 100, 0.0 );

//            std::cout << "Propagation to Final condition completed: orbit number: " << orbitIdOne << std::endl;

//            writeStateHistoryToFile( stateHistory, orbitIdOne, orbitType, librationPointNr, 1000, false );

//            std::cout << "State history of the orbits written to file in raw/orbits, orbit number: " << orbitIdOne << std::endl;

//            // ===============================================================
//            // == Compute manifolds based on precomputed initial conditions ==
//            // ===============================================================

//            std::cout << "start computation of manifolds of orbit number: " << orbitIdOne << std::endl;

//            computeManifolds(initialStateVector, orbitalPeriod, orbitIdOne, librationPointNr, orbitType);

//            std::cout << "FINISHED MANIFOLDS COMPUTATION: " <<  std::endl;
//        }
//    }

    return 0;
}
