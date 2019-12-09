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

Eigen::VectorXd extractStatesContinuationVector(std::string referenceString, const double familyHamiltonian, const double accelerationMagnitude, const bool startFromAlpha, int& numberOfCollocationPoints)
{

    // determine the orbit with minimum energy deviation:
    DIR *d;
    struct dirent *dir;

    d = opendir("../data/raw/orbits/augmented/");

    double minimumDifference = 5;
    double closestQuantity = 0.0;
    std::string closestFileName;
    std::string closestQuantityString;

    if (d and startFromAlpha == false)
        {
            while ((dir = readdir(d)) != NULL)
            {

                std::string fileName = dir->d_name;
                if(fileName.size() > referenceString.size())
                {
                    if(fileName.substr(0,referenceString.size()) == referenceString )
                    {

                        double fileQuantity;
                            if (startFromAlpha == false)
                            {
                                fileQuantity = std::stod(fileName.substr(referenceString.size()+1,14));
                            } else
                            {
                                fileQuantity = std::stod(fileName.substr(14,13));
                            }

                        double difference;
                        if (startFromAlpha == false)
                        {
                          difference = std::abs(fileQuantity-familyHamiltonian);
                        } else
                        {
                          difference = std::abs(fileQuantity-accelerationMagnitude);
                        }
                        if (difference < minimumDifference and difference > 1.0E-12)
                        {
                            minimumDifference = difference;
                            closestQuantity = fileQuantity;
                            closestFileName = fileName;

                            if (startFromAlpha == false)
                            {
                                closestQuantityString = fileName.substr(referenceString.size()+1,14);
                            } else
                            {
                                closestQuantityString = fileName.substr(14,13);
                            }
                        }

                    }

                }
            }
            closedir(d);
        }

    std::cout << "closestQuantityString: " << closestQuantityString << std::endl;

    std::string directory_path;
    std::string file_name;
    std::string file_path;
    int desiredCounterQuantity;
    int desiredQuantityLength;


    if (startFromAlpha == false)
    {
        directory_path= "../data/raw/orbits/augmented/varying_hamiltonian/";
        file_name = referenceString + "_states_continuation.txt";
        file_path = directory_path + file_name;
        desiredCounterQuantity = 1;
        desiredQuantityLength = 14;


    } else
    {
       directory_path= "../data/raw/orbits/augmented/varying_acceleration/";
       file_name = referenceString + "_states_continuation.txt";
       file_path = directory_path + file_name;
       std::ostringstream ssAccelerationMag1;
       ssAccelerationMag1 << std::fixed <<  std::setprecision(11) << accelerationMagnitude;
       std::string stringAccelerationAngleMag1 = ssAccelerationMag1.str();
        closestQuantityString = stringAccelerationAngleMag1;
        desiredCounterQuantity = 9;
        desiredQuantityLength = 20;

    }

    // search the corresponding statesContinuation Entry and extract the statesContinuation vector
    std::vector<double> statesContinuationVector;
    std::ifstream file(file_path);
    std::string str;
    int orbitCounter = 0;
    int desiredOrbitNumber = 0;
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
            if (counter == desiredCounterQuantity)
               {


                if (startFromAlpha == false)
                {

                    if (subs.substr(0,desiredQuantityLength) == closestQuantityString )
                    {
                        std::cout << "condition REACHED" << std::endl;
                        storeLineInVector = true;
                        closestQuantity =  std::stod(subs);


                    }
                } else {

                        if ( std::abs(std::stod(subs) - accelerationMagnitude) < 1.0e-5 )
                        {
                            storeLineInVector = true;
                            closestQuantity =  std::stod(subs);
                            desiredOrbitNumber = orbitCounter;

                        }
                }


                }
                counter ++;

             // inspect the Hamiltonian value
             if (counter > 3 and storeLineInVector == true and startFromAlpha == false )
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


        orbitCounter++;

    }

    // create vector
    if (startFromAlpha == true)
    {

        std::ifstream file(file_path);
        std::string str;
        int orbitCounter2 = 0;
        while (std::getline(file, str)) {
            // Lloop over the textfile line and divide it into substrings: use isstringstream
            std::istringstream iss(str);
            int counter = 0; // set counter to analyze the Hamiltonian value
            bool storeLineInVector = false;

            std::cout << "orbitCounter2: " << orbitCounter2 << std::endl;
            std::cout << "desiredOrbitNumber: " << desiredOrbitNumber << std::endl;

            while(iss)
            {

                // loop over the substrings
                std::string subs;
                double subsDouble;
                iss >> subs;

                counter ++;
                if (counter > 3 and orbitCounter2 == desiredOrbitNumber )
                {
                    //std::cout << "counter: " << counter << std::endl;
                    //std::cout << "std::stod(subs): " << std::stod(subs) << std::endl;

                    bool storageTwo = true;
                    try
                    {
                        double value = std::stod(subs);
                        //std::cout << "Converted string to a value of " << value << std::endl;
                    }
                    catch(std::exception& e)
                    {
                        //std::cout << "Could not convert string to double: " << subs << std::endl;
                       storageTwo = false;
                    }

                    //std::cout << "storageTwo: " << storageTwo << std::endl;
                    if(storageTwo == true)
                    {
                       //std::cout << "storageTwo condition reached " << std::endl;
                       statesContinuationVector.push_back(std::stod(subs));
                    }


                }

            };

            orbitCounter2++;

        }


    }

    std::cout << "creating output vector " << std::endl;
    Eigen::VectorXd outputVector(statesContinuationVector.size()); outputVector.setZero();
    std::cout << "statesContinuationVector.size(): " << statesContinuationVector.size() << std::endl;

    for(int k = 0; k < (statesContinuationVector.size()); k++)
    {
        outputVector(k)  = statesContinuationVector.at(k);
    }

    numberOfCollocationPoints = (outputVector.size()/11 - 1)/3 + 1;


    return outputVector;


}

std::string createReferenceString(const int librationPointNr, const std::string orbitType, const double accelerationMagnitude, const double accelerationAngle, const double accelerationAngle2, const double familyHamiltonian, const bool startFromAlpha)
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

    std::ostringstream ssFamilyHamiltonian;
    ssFamilyHamiltonian << std::fixed << std::setprecision(11) << familyHamiltonian;
    std::string stringFamilyHamiltonian = ssFamilyHamiltonian.str();



    std::string outputString;
    if (startFromAlpha == false)
    {
        outputString = "L" + std::to_string(librationPointNr) + "_" + orbitType + "_"
                    + stringAccelerationMagnitude + "_" + stringAccelerationAngle1 + "_" + stringAccelerationAngle2;
    } else
    {
        outputString = "L" + std::to_string(librationPointNr) + "_" + orbitType +  "_"
                + stringAccelerationAngle1 + "_" + stringAccelerationAngle2 + "_" + stringFamilyHamiltonian;


    }

    return outputString;
}

Eigen::MatrixXd refineOrbitHamiltonian (const int librationPointNr, const std::string orbitType,  const double accelerationMagnitude, const double accelerationAngle, const double accelerationAngle2,
                                        const double familyHamiltonian, const double massParameter, const int continuationIndex, bool startFromAlpha, int& numberOfCollocationPoints)
{

    // create the reference vector:
    std::string referenceString =  createReferenceString(librationPointNr, orbitType, accelerationMagnitude, accelerationAngle, accelerationAngle2, familyHamiltonian, startFromAlpha);
    std::cout << "referenceString: " << referenceString << std::endl;
    // Determine the orbit with hamiltonian closest to the familyHamiltonian and extract numberOfNodes and statesContinuationVector
    std::cout << "start extract statesContinuationFunction: "  << std::endl;

    Eigen::VectorXd statesVector = extractStatesContinuationVector(referenceString, familyHamiltonian, accelerationMagnitude, startFromAlpha, numberOfCollocationPoints);
    std::cout << " extract statesContinuationFunction Finished: "  << std::endl;
    std::cout << " statesVector.size(): " << statesVector.size()  << std::endl;

    double orbitHamiltonian = computeHamiltonian(massParameter, statesVector.segment(0,10) );

    std::cout << " reached return StateVector Segment: "  << std::endl;

    return statesVector;



}
