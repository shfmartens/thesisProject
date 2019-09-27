#include <cmath>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Dense>
#include <map>
#include <chrono>
#include <boost/function.hpp>
#include <sstream>


#include "Tudat/Astrodynamics/BasicAstrodynamics/celestialBodyConstants.h"
#include "Tudat/Astrodynamics/Gravitation/librationPoint.h"
#include "Tudat/Astrodynamics/Gravitation/jacobiEnergy.h"
#include "Tudat/InputOutput/basicInputOutput.h"
#include "dirent.h"
#include "stdio.h"
#include "createLowThrustInitialConditions.h"
#include "applyCollocation.h"
#include "applyPredictionCorrection.h"
#include "applyLineSearchAttenuation.h"
#include "stateDerivativeModel.h"
#include "stateDerivativeModelAugmented.h"
#include "propagateOrbitAugmented.h"
#include "computeCollocationCorrection.h"
#include "applyMeshRefinement.h"
#include "interpolatePolynomials.h"

Eigen::VectorXd extractStatesContinuationVector(std::string referenceString, const double familyHamiltonian, int& numberOfCollocationPoints)
{

    // determine the orbit with minimum energy deviation:
    DIR *d;
    struct dirent *dir;

    d = opendir("../data/raw/orbits/augmented/");

    double minimumDifference = 5;
    double closestHamiltonian = 0.0;
    std::string closestFileName;
    std::string closestHamiltonianString;

    if (d)
        {
            while ((dir = readdir(d)) != NULL)
            {

                std::string fileName = dir->d_name;
                if(fileName.size() > referenceString.size())
                {
                    if(fileName.substr(0,referenceString.size()) == referenceString )
                    {

                        double fileHamiltonian = std::stod(fileName.substr(referenceString.size()+1,14));
                        double difference = std::abs(fileHamiltonian-familyHamiltonian);

                        if (difference < minimumDifference and difference > 1.0E-12)
                        {
                            minimumDifference = difference;
                            closestHamiltonian = fileHamiltonian;
                            closestFileName = fileName;
                            closestHamiltonianString = fileName.substr(referenceString.size()+1,14);
                        }

                    }

                }
            }
            closedir(d);
        }

    // search the corresponding statesContinuation Entry and extract the statesContinuation vector
    std::string directory_path = "../data/raw/orbits/augmented/varying_hamiltonian/";
    std::string file_name = referenceString + "_states_continuation.txt";
    std::string file_path = directory_path + file_name;
    std::ifstream file(file_path);
    std::string str;
    std::vector<double> statesContinuationVector;

    std::cout << "closestHamiltonian: " << closestHamiltonian << std::endl;
    int i = 0; // tempoarary variable for testing!
    while (std::getline(file, str)) {

        // Lloop over the textfile line and divide it into substrings: use isstringstream
        std::istringstream iss(str);
        int counter = 0; // set counter to analyze the Hamiltonian value
        bool storeLineInVector = false;

        while(iss)
        {
            // loop over the substrings
            std::string subs;
            double subsDouble;
            iss >> subs;


            // inspect the Hamiltonian value
            if (counter == 1)
               {

                  if (subs.substr(0,14) == closestHamiltonianString )
                  {
                      storeLineInVector = true;
                      closestHamiltonian =  std::stod(subs);

                  }

                }
                counter ++;

             // inspect the Hamiltonian value
             if (counter > 3 and storeLineInVector == true )
             {
                 bool storage = true;
                 try
                 {
                     double value = std::stod(subs);
                     //std::cout << "Converted string to a value of " << value << std::endl;
                 }
                 catch(std::exception& e)
                 {
                     //std::cout << "Could not convert string to double: " << subs << std::endl;
                    storage = false;
                 }

                 if(storage == true)
                 {
                    statesContinuationVector.push_back(std::stod(subs));
                 }

             }


            };

    }

    Eigen::VectorXd outputVector(statesContinuationVector.size()); outputVector.setZero();

    for(int k = 0; k < (statesContinuationVector.size()); k++)
    {
        outputVector(k)  = statesContinuationVector.at(k);
    }

    numberOfCollocationPoints = (outputVector.size()/11 - 1)/3 + 1;

    return outputVector;


}

std::string createReferenceString(const int librationPointNr, const std::string orbitType, const double accelerationMagnitude, const double accelerationAngle, const double accelerationAngle2)
{
    std::ostringstream ssAccelerationMagnitude;
    ssAccelerationMagnitude << std::fixed <<std::setprecision(11) << accelerationMagnitude;
    std::string stringAccelerationMagnitude = ssAccelerationMagnitude.str();

    std::ostringstream ssAccelerationAngle1;
    ssAccelerationAngle1 << std::fixed <<  std::setprecision(11) << accelerationAngle;
    std::string stringAccelerationAngle1 = ssAccelerationAngle1.str();

    std::ostringstream ssAccelerationAngle2;
    ssAccelerationAngle2 << std::fixed << std::setprecision(11) << accelerationAngle2;
    std::string stringAccelerationAngle2 = ssAccelerationAngle2.str();

//    std::ostringstream ssHamiltonian;
//    ssHamiltonian << std::fixed << std::setprecision(11) << familyHamiltonian;
//    std::string stringHamiltonian = ssHamiltonian.str();
    std::string outputString = "L" + std::to_string(librationPointNr) + "_" + orbitType + "_"
            + stringAccelerationMagnitude + "_" + stringAccelerationAngle1 + "_" + stringAccelerationAngle2;
    return outputString;
}

Eigen::MatrixXd refineOrbitHamiltonian (const int librationPointNr, const std::string orbitType,  const double accelerationMagnitude, const double accelerationAngle, const double accelerationAngle2,
                                        const double familyHamiltonian, const double massParameter, const int continuationIndex, int& numberOfCollocationPoints)
{

    // create the reference vector:
    std::string referenceString =  createReferenceString(librationPointNr, orbitType, accelerationMagnitude, accelerationAngle, accelerationAngle2);

    // Determine the orbit with hamiltonian closest to the familyHamiltonian and extract numberOfNodes and statesContinuationVector
    Eigen::VectorXd statesVector = extractStatesContinuationVector(referenceString, familyHamiltonian, numberOfCollocationPoints);

    double orbitHamiltonian = computeHamiltonian(massParameter, statesVector.segment(0,10) );


    return statesVector;



}
