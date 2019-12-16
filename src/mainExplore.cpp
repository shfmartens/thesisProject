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
#include <cmath>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <random>

#include "Tudat/Astrodynamics/Gravitation/librationPoint.h"
#include "Tudat/Astrodynamics/BasicAstrodynamics/celestialBodyConstants.h"
#include "Tudat/Astrodynamics/BasicAstrodynamics/physicalConstants.h"
#include "Tudat/Astrodynamics/BasicAstrodynamics/astrodynamicsFunctions.h"


#include "createInitialConditions.h"
#include "createLowThrustInitialConditions.h"
#include "computeManifolds.h"
#include "propagateOrbit.h"
//#include "completeInitialConditionsHaloFamily.h"
//#include "createInitialConditionsAxialFamily.h"
#include "connectManifoldsAtTheta.h"
#include "createEquilibriumLocations.h"
#include "stateDerivativeModelAugmentedVaryingMass.h"
#include "stateDerivativeModelAugmented.h"
#include "omp.h"
#include "applyCollocation.h"
#include "computeCollocationCorrection.h"
#include "interpolatePolynomials.h"


double massParameter = tudat::gravitation::circular_restricted_three_body_problem::computeMassParameter( tudat::celestial_body_constants::EARTH_GRAVITATIONAL_PARAMETER, tudat::celestial_body_constants::MOON_GRAVITATIONAL_PARAMETER );
double maximumThrust = 0.1;
int main (){


     //================================
     //== Compute equilibria, comment out when computing low-thrust intial positions ==
     //================================
//    Eigen::Vector3d tempVector; tempVector.setZero();
//    tempVector(0) = 0.01;
//        tempVector(1) = 0.04;
//        tempVector(2) = 0.066;

//    for(int k = 0; k < 1; k++)
//    {
//        double alpha = 0.0;
//        double accMag = 0.0107;
//        for (int i = 2; i < 3  ; i++)
//        {
//            double tempAcc = 0.064;
//            double tempAng = 120.0;

//           Eigen::Vector2d equilibriumTest = createEquilibriumLocations(i, tempAcc,  tempAng, "acceleration", 1.0, massParameter);
//           Eigen::VectorXd hamiltonianTest(10); hamiltonianTest.setZero();
//           hamiltonianTest.segment(0,2) = equilibriumTest;
//           hamiltonianTest(6) = tempAcc;
//           hamiltonianTest(7) = tempAng;

//            double testHamiltonianValue = computeHamiltonian(massParameter, hamiltonianTest);
//            std::cout << "\n== Eq result =="<< std::endl
//                      << "librationPointNr: " << i << std::endl
//                      << "alt: " << tempAcc << std::endl
//                      << "alpha: " << tempAng << std::endl
//                      << "equilibriumLocation: \n" << equilibriumTest << std::endl
//                      << "testHamiltonianValue: \n" << testHamiltonianValue << std::endl

//                      << "=================" << std::endl;



//        }


//    double semiMajorAxis = 384400*1000;
//    double EarthGravPar = tudat::celestial_body_constants::EARTH_GRAVITATIONAL_PARAMETER;
//    double MoonGravPar = tudat::celestial_body_constants::MOON_GRAVITATIONAL_PARAMETER;
//    double GravConst = tudat::physical_constants::GRAVITATIONAL_CONSTANT;
//    double moonMass = MoonGravPar/GravConst;

//    std::cout.precision(14);
//    double period = tudat::basic_astrodynamics::computeKeplerOrbitalPeriod(semiMajorAxis, EarthGravPar, moonMass  );
//    double velocity = semiMajorAxis / (period/(2.0*tudat::mathematical_constants::PI));
//    std::cout << "semi major axis [m]: " << semiMajorAxis << std::endl;
//    std::cout << "EARTH GM [[m^3 s^-2]]: " << EarthGravPar << std::endl;
//    std::cout << "Moon GM [[m^3 s^-2]]: " << MoonGravPar << std::endl;
//    std::cout << "GravConst [meter^3 per kilogram per second^2]: " << GravConst << std::endl;
//    std::cout << "moonMass []: " << moonMass << std::endl;
//    std::cout << "period []: " << (period /tudat::physical_constants::SIDEREAL_DAY)<< std::endl;
//    std::cout << "velocity []: " << velocity << std::endl;
//    std::cout << "eps: " << std::numeric_limits<double>::epsilon( ) << std::endl;



    // ================================
    // == Compute initial conditions ==
    // ================================
        

    #pragma omp parallel num_threads(1)
    {
        #pragma omp for
        for (unsigned int i=1; i<=1; i++) {
            if (i ==1)
            {
                std::cout << "Run Thread " << i << std::endl;
                std::string orbitType = "horizontal";
                int continuationIndex = 7; //1: Continuate for H, 6: acceleration, 7: alpha, 8: beta
                double accelerationMagnitude = 0.1;
                double accelerationAngle = 60.0;
                double accelerationAngle2 = 0.0;
                double initialMass = 1.0;
                double ySign = -1.0;
                double familyHamiltonian = -1.55;
                int numberOfFamilyMembers = 31;
                bool startContinuationFromTextFile = false;
                createLowThrustInitialConditions(2, ySign, orbitType, continuationIndex, accelerationMagnitude, accelerationAngle, accelerationAngle2, initialMass, familyHamiltonian, startContinuationFromTextFile, numberOfFamilyMembers );



            }
            if (i ==2)
            {
                std::cout << "Run Thread " << i << std::endl;
                std::string orbitType = "horizontal";
                int continuationIndex = 7; //1: Continuate for H, 6: acceleration, 7: alpha, 8: beta
                double accelerationMagnitude = 0.1;
                double accelerationAngle = 60.0;
                double accelerationAngle2 = 0.0;
                double initialMass = 1.0;
                double ySign = -1.0;
                double familyHamiltonian = -1.55;
                int numberOfFamilyMembers = 1000;
                bool startContinuationFromTextFile = false;
                createLowThrustInitialConditions(2, ySign, orbitType, continuationIndex, accelerationMagnitude, accelerationAngle, accelerationAngle2, initialMass, familyHamiltonian, startContinuationFromTextFile, numberOfFamilyMembers );

            }
            if (i ==3)
            {
                std::cout << "Run Thread " << i << std::endl;
                std::string orbitType = "horizontal";
                int continuationIndex = 7; //1: Continuate for H, 6: acceleration, 7: alpha, 8: beta
                double accelerationMagnitude = 0.1;
                double accelerationAngle = 120.0;
                double accelerationAngle2 = 0.0;
                double initialMass = 1.0;
                double ySign = -1.0;
                double familyHamiltonian = -1.50;
                int numberOfFamilyMembers = 1000;
                bool startContinuationFromTextFile = false;
                createLowThrustInitialConditions(2, ySign, orbitType, continuationIndex, accelerationMagnitude, accelerationAngle, accelerationAngle2, initialMass, familyHamiltonian, startContinuationFromTextFile, numberOfFamilyMembers );

            }
            if (i ==4)
            {
                std::cout << "Run Thread " << i << std::endl;
                std::string orbitType = "horizontal";
                int continuationIndex = 7; //1: Continuate for H, 6: acceleration, 7: alpha, 8: beta
                double accelerationMagnitude = 0.1;
                double accelerationAngle = 120.0;
                double accelerationAngle2 = 0.0;
                double initialMass = 1.0;
                double ySign = -1.0;
                double familyHamiltonian = -1.50;
                int numberOfFamilyMembers = 1000;
                bool startContinuationFromTextFile = false;
                createLowThrustInitialConditions(2, ySign, orbitType, continuationIndex, accelerationMagnitude, accelerationAngle, accelerationAngle2, initialMass, familyHamiltonian, startContinuationFromTextFile, numberOfFamilyMembers );


            }
            if (i ==5)
            {
                std::cout << "Run Thread " << i << std::endl;
                std::string orbitType = "horizontal";
                int continuationIndex = 7; //1: Continuate for H, 6: acceleration, 7: alpha, 8: beta
                double accelerationMagnitude = 0.05;
                double accelerationAngle = 0.0;
                double accelerationAngle2 = 0.0;
                double initialMass = 1.0;
                double ySign = 1.0;
                double familyHamiltonian = -1.50;
                int numberOfFamilyMembers = 4450;
                bool startContinuationFromTextFile = false;
                createLowThrustInitialConditions(1, ySign, orbitType, continuationIndex, accelerationMagnitude, accelerationAngle, accelerationAngle2, initialMass, familyHamiltonian, startContinuationFromTextFile, numberOfFamilyMembers );

            }
                if (i ==6)
                {
                    std::cout << "Run Thread " << i << std::endl;
                    std::string orbitType = "horizontal";
                    int continuationIndex = 6; //1: Continuate for H, 6: acceleration, 7: alpha, 8: beta
                    double accelerationMagnitude = 0.0;
                    double accelerationAngle = 0.0;
                    double accelerationAngle2 = 0.0;
                    double initialMass = 1.0;
                    double ySign = 1.0;
                    double familyHamiltonian = -1.50;
                    int numberOfFamilyMembers = 5000;
                    bool startContinuationFromTextFile = false;
                    createLowThrustInitialConditions(2, ySign, orbitType, continuationIndex, accelerationMagnitude, accelerationAngle, accelerationAngle2, initialMass, familyHamiltonian, startContinuationFromTextFile, numberOfFamilyMembers );

                }
                if (i ==7)
                {
                    std::cout << "Run Thread " << i << std::endl;
                    std::string orbitType = "horizontal";
                    int continuationIndex = 6; //1: Continuate for H, 6: acceleration, 7: alpha, 8: beta
                    double accelerationMagnitude = 0.0;
                    double accelerationAngle = 240.0;
                    double accelerationAngle2 = 0.0;
                    double initialMass = 1.0;
                    double ySign = 1.0;
                    double familyHamiltonian = -1.55;
                    int numberOfFamilyMembers = 5000;
                    bool startContinuationFromTextFile = false;
                    createLowThrustInitialConditions(2, ySign, orbitType, continuationIndex, accelerationMagnitude, accelerationAngle, accelerationAngle2, initialMass, familyHamiltonian, startContinuationFromTextFile, numberOfFamilyMembers );

                }

                if (i ==8)
                {
                    std::cout << "Run Thread " << i << std::endl;
                    std::string orbitType = "horizontal";
                    int continuationIndex = 6; //1: Continuate for H, 6: acceleration, 7: alpha, 8: beta
                    double accelerationMagnitude = 0.0;
                    double accelerationAngle = 240.0;
                    double accelerationAngle2 = 0.0;
                    double initialMass = 1.0;
                    double ySign = 1.0;
                    double familyHamiltonian = -1.525;
                    int numberOfFamilyMembers = 5000;
                    bool startContinuationFromTextFile = false;
                    createLowThrustInitialConditions(2, ySign, orbitType, continuationIndex, accelerationMagnitude, accelerationAngle, accelerationAngle2, initialMass, familyHamiltonian, startContinuationFromTextFile, numberOfFamilyMembers );

                }
                if (i ==9)
                {
                    std::cout << "Run Thread " << i << std::endl;
                    std::string orbitType = "horizontal";
                    int continuationIndex = 6; //1: Continuate for H, 6: acceleration, 7: alpha, 8: beta
                    double accelerationMagnitude = 0.0;
                    double accelerationAngle = 240.0;
                    double accelerationAngle2 = 0.0;
                    double initialMass = 1.0;
                    double ySign = 1.0;
                    double familyHamiltonian = -1.50;
                    int numberOfFamilyMembers = 5000;
                    bool startContinuationFromTextFile = false;
                    createLowThrustInitialConditions(2, ySign, orbitType, continuationIndex, accelerationMagnitude, accelerationAngle, accelerationAngle2, initialMass, familyHamiltonian, startContinuationFromTextFile, numberOfFamilyMembers );

                }
                if (i ==10)
                {
                    std::cout << "Run Thread " << i << std::endl;
                    std::string orbitType = "horizontal";
                    int continuationIndex = 6; //1: Continuate for H, 6: acceleration, 7: alpha, 8: beta
                    double accelerationMagnitude = 0.0;
                    double accelerationAngle = 300.0;
                    double accelerationAngle2 = 0.0;
                    double initialMass = 1.0;
                    double ySign = 1.0;
                    double familyHamiltonian = -1.525;
                    int numberOfFamilyMembers = 5000;
                    bool startContinuationFromTextFile = false;
                    createLowThrustInitialConditions(2, ySign, orbitType, continuationIndex, accelerationMagnitude, accelerationAngle, accelerationAngle2, initialMass, familyHamiltonian, startContinuationFromTextFile, numberOfFamilyMembers );

                }
                if (i ==11)
                {
                    std::cout << "Run Thread " << i << std::endl;
                    std::string orbitType = "horizontal";
                    int continuationIndex = 1; //1: Continuate for H, 6: acceleration, 7: alpha, 8: beta
                    double accelerationMagnitude = 0.05;
                    double accelerationAngle = 240.0;
                    double accelerationAngle2 = 0.0;
                    double initialMass = 1.0;
                    double ySign = 1.0;
                    double familyHamiltonian = -1.58;
                    int numberOfFamilyMembers = 5000;
                    bool startContinuationFromTextFile = false;
                    createLowThrustInitialConditions(2, ySign, orbitType, continuationIndex, accelerationMagnitude, accelerationAngle, accelerationAngle2, initialMass, familyHamiltonian, startContinuationFromTextFile, numberOfFamilyMembers );

                  }
                    if (i ==12)
                    {
                        std::cout << "Run Thread " << i << std::endl;
                        std::string orbitType = "horizontal";
                        int continuationIndex = 1; //1: Continuate for H, 6: acceleration, 7: alpha, 8: beta
                        double accelerationMagnitude = 0.05;
                        double accelerationAngle = 300.0;
                        double accelerationAngle2 = 0.0;
                        double initialMass = 1.0;
                        double ySign = 1.0;
                        double familyHamiltonian = -1.58;
                        int numberOfFamilyMembers = 5000;
                        bool startContinuationFromTextFile = false;
                        createLowThrustInitialConditions(2, ySign, orbitType, continuationIndex, accelerationMagnitude, accelerationAngle, accelerationAngle2, initialMass, familyHamiltonian, startContinuationFromTextFile, numberOfFamilyMembers );

                    }
                    if (i ==13)
                    {
                        std::cout << "Run Thread " << i << std::endl;
                        std::string orbitType = "horizontal";
                        int continuationIndex = 1; //1: Continuate for H, 8: acceleration, 9: alpha, 10: beta
                        double accelerationMagnitude = 0.1;
                        double accelerationAngle = 0.0;
                        double accelerationAngle2 = 0.0;
                        double initialMass = 1.0;
                        double ySign = 1.0;
                        double familyHamiltonian = -1.58;
                        int numberOfFamilyMembers = 5000;
                        bool startContinuationFromTextFile = false;
                        createLowThrustInitialConditions(2, ySign, orbitType, continuationIndex, accelerationMagnitude, accelerationAngle, accelerationAngle2, initialMass, familyHamiltonian, startContinuationFromTextFile, numberOfFamilyMembers );

                    }
                    if (i ==14)
                    {
                        std::cout << "Run Thread " << i << std::endl;
                        std::string orbitType = "horizontal";
                        int continuationIndex = 1; //1: Continuate for H, 6: acceleration, 7: alpha, 8: beta
                        double accelerationMagnitude = 0.1;
                        double accelerationAngle = 60.0;
                        double accelerationAngle2 = 0.0;
                        double initialMass = 1.0;
                        double ySign = 1.0;
                        double familyHamiltonian = -1.58;
                        int numberOfFamilyMembers = 5000;
                        bool startContinuationFromTextFile = false;
                        createLowThrustInitialConditions(2, ySign, orbitType, continuationIndex, accelerationMagnitude, accelerationAngle, accelerationAngle2, initialMass, familyHamiltonian, startContinuationFromTextFile, numberOfFamilyMembers );

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


//int testCollocPoints = 5;
//int numberOfSegments = testCollocPoints-1;
//int numberOfOddPoints = 3*numberOfSegments+1;


//Eigen::MatrixXd collocationDesignVector = Eigen::MatrixXd::Zero(83,1);
//Eigen::VectorXd previousDesignVector = Eigen::VectorXd::Zero(11*numberOfOddPoints);

//std::ifstream inFile;
////std::string path = "/Users/Sjors/Desktop/designvector.txt";
//std::string path = "../designvector.txt";

//inFile.open(path);

//if (!inFile) {
//    std::cout << "Unable to open file datafile.txt" << std::endl;
//} else
//{

//    int i = 0;
//    double x;
//    while(inFile >> x)
//    {

//        collocationDesignVector(i) = x;
//        i++;
//    }

//}
