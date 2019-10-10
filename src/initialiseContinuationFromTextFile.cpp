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
#include "refineOrbitHamiltonian.h"
#include "createEquilibriumLocations.h"

Eigen::VectorXd extractStatesContinuationVectorFromKnownOrbitNumber(std::string continuation_fileName, const int targetOrbitNumber)
{
    // search the corresponding statesContinuation Entry and extract the statesContinuation vector
    std::string directory_path = "../data/raw/orbits/augmented/varying_hamiltonian/";
    std::string file_path = directory_path + continuation_fileName;
    std::ifstream file(file_path);
    std::string str;
    std::vector<double> statesContinuationVector;

    std::ostringstream ssAccelerationMagnitude;
    ssAccelerationMagnitude << std::fixed <<std::setprecision(13) << targetOrbitNumber;
    std::string targetOrbitNumberString = ssAccelerationMagnitude.str();
    int derivativeMaxLimit = 10;
    int derivativeMaxcounter = 0;

    Eigen::VectorXd outputVector(10); outputVector.setZero();

    //std::cout << "targetHamiltonian: " << targetHamiltonian << std::endl;
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

            int tempOrbitNumber;
            if (counter == 0)
            {




                if (std::stod(subs) == targetOrbitNumber )
                {
                    storeLineInVector = true;

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

                 if(storage == true and derivativeMaxcounter < derivativeMaxLimit)
                 {
                    outputVector(derivativeMaxcounter) = std::stod(subs);
                    derivativeMaxcounter++;
                 }

             }


            };

    }



    return outputVector;
}

Eigen::VectorXd extractStatesContinuationVectorFromKnownHamiltonian(std::string continuation_fileName, const double targetHamiltonian, int& numberOfCollocationPoints, int& orbitNumber)
{
    // search the corresponding statesContinuation Entry and extract the statesContinuation vector
    std::string directory_path = "../data/raw/orbits/augmented/varying_hamiltonian/";
    std::string file_path = directory_path + continuation_fileName;
    std::ifstream file(file_path);
    std::string str;
    std::vector<double> statesContinuationVector;

    std::ostringstream ssAccelerationMagnitude;
    ssAccelerationMagnitude << std::fixed <<std::setprecision(13) << targetHamiltonian;
    std::string targetHamiltonianString = ssAccelerationMagnitude.str();
    double targetHamiltonianReconverted;

    //std::cout << "targetHamiltonian: " << targetHamiltonian << std::endl;
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

            int tempOrbitNumber;
            if (counter == 0)
            {
                tempOrbitNumber = std::stod(subs);
            }

            // inspect the Hamiltonian value
            if (counter == 1)
               {
                  if (subs.substr(0,14) == targetHamiltonianString.substr(0,14) )
                  {
                      storeLineInVector = true;
                      targetHamiltonianReconverted =  std::stod(subs);
                      orbitNumber = static_cast<int>(tempOrbitNumber);

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


    //std::cout << "\n orbitNumber: " << orbitNumber<< std::endl;

    return outputVector;


}

void initialiseContinuationFromTextFile (const int librationPointNr, const std::string orbitType, const double accelerationMagnitude, const double accelerationAngle, const double accelerationAngle2,
                                         const double hamiltonianFirstGuess, const double hamiltonianSecondGuess, const double ySign, const double massParameter,
                                         Eigen::VectorXd& statesContinuationVectorFirstGuess, Eigen::VectorXd& statesContinuationVectorSecondGuess,
                                         int& numberOfCollocationPointsFirstGuess, int& numberOfCollocationPointsSecondGuess, Eigen::VectorXd& adaptedIncrementVector, int& numberOfInitialConditions)
{
      int orbitNumberFirstGuess;
      int orbitNumberSecondGuess;

      // Extract the state continuation vectors of the desired hamiltonians and put them into odd points
      std::string property_string = createReferenceString(librationPointNr, orbitType, accelerationMagnitude, accelerationAngle, accelerationAngle2 );
      std::string continuation_fileName = property_string + "_states_continuation_startup.txt";

      Eigen::VectorXd statesContinuationVectorFirstGuessTemplate = extractStatesContinuationVectorFromKnownHamiltonian(continuation_fileName, hamiltonianFirstGuess, numberOfCollocationPointsFirstGuess, orbitNumberFirstGuess);
      Eigen::VectorXd statesContinuationVectorSecondGuessTemplate = extractStatesContinuationVectorFromKnownHamiltonian(continuation_fileName, hamiltonianSecondGuess, numberOfCollocationPointsSecondGuess, orbitNumberSecondGuess);

      statesContinuationVectorFirstGuess.resize(statesContinuationVectorFirstGuessTemplate.rows()); statesContinuationVectorFirstGuess.setZero();
      statesContinuationVectorSecondGuess.resize(statesContinuationVectorSecondGuessTemplate.rows()); statesContinuationVectorSecondGuess.setZero();

      statesContinuationVectorFirstGuess = statesContinuationVectorFirstGuessTemplate;
      statesContinuationVectorSecondGuess = statesContinuationVectorSecondGuessTemplate;



      // compute adapted increment vector (seek orbit corresponding to  orbit number 1.000000)
      Eigen::VectorXd firstStateOfSecondOrbit = extractStatesContinuationVectorFromKnownOrbitNumber(continuation_fileName, 1);
      std::cout << "firstSTateOfSecondorbit: \n" << firstStateOfSecondOrbit << std::endl;


      Eigen::VectorXd increment = Eigen::VectorXd::Zero(6);
      Eigen::VectorXd fullEquilibriumLocation = Eigen::VectorXd::Zero(6);
      fullEquilibriumLocation.segment(0,2) = createEquilibriumLocations(librationPointNr, accelerationMagnitude, accelerationAngle, "acceleration", ySign, massParameter );
      increment = firstStateOfSecondOrbit.segment(0,6) - fullEquilibriumLocation;
      adaptedIncrementVector = 10.0 *increment / (increment.norm());

      std::cout << "increment: \n" << increment << std::endl;


    // Initialize the procedure by getCollocatedInitialState


    // select the right output conditions
     numberOfInitialConditions = orbitNumberSecondGuess+1;



}
