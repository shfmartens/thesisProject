#include <cmath>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Dense>
#include <map>

#include <chrono>

#include <boost/function.hpp>

#include "Tudat/Astrodynamics/BasicAstrodynamics/celestialBodyConstants.h"
#include "Tudat/Astrodynamics/Gravitation/librationPoint.h"
#include "Tudat/Astrodynamics/Gravitation/jacobiEnergy.h"
#include "Tudat/InputOutput/basicInputOutput.h"


#include "createLowThrustInitialConditions.h"
#include "computeLevel1MassRefinementCorrection.h"
#include "computeLevel2MassRefinementCorrection.h"
#include "computeLevel2Correction.h"
#include "propagateOrbitAugmented.h"
#include "propagateMassVaryingOrbitAugmented.h"
#include "propagateOrbit.h"
#include "stateDerivativeModel.h"
#include "stateDerivativeModelAugmented.h"

void writeMassRefinementDataToFile(const int librationPointNr, const double accelerationMagnitude, const double alpha, const double amplitude, const int numberOfPatchPoints, const double correctionTime,
                              std::map< double, Eigen::VectorXd > stateHistory, const Eigen::VectorXd stateVector, Eigen::VectorXd deviations, const Eigen::MatrixXd propagatedStatesInclSTM,
                              const int cycleNumber, const int correctorLevel, const int numberOfCorrections, const double correctionDuration )
{

    Eigen::VectorXd deviationsAndDuration = Eigen::VectorXd::Zero(7);
    deviationsAndDuration.segment(0,5) = deviations;
    deviationsAndDuration(5) = correctionDuration;
    deviationsAndDuration(6) = numberOfCorrections;

    std::string fileNameStringStateVector;
    std::string fileNameStringStateHistory;
    std::string fileNameStringDeviations;
    std::string fileNameStringPropagatedStates;



    std::string directoryString = "../data/raw/mass_refinement/";

    fileNameStringStateVector = ("L" + std::to_string(librationPointNr) + "_" + std::to_string(accelerationMagnitude)
                                 + "_" + std::to_string(alpha) + "_" + std::to_string(amplitude) + "_" + std::to_string(numberOfPatchPoints) + "_" + std::to_string(correctionTime) + "_" +  std::to_string(cycleNumber) + "_" +  std::to_string(correctorLevel) + "_stateVectors.txt");
    fileNameStringStateHistory = ("L" + std::to_string(librationPointNr) + "_" + std::to_string(accelerationMagnitude)
                                  + "_" + std::to_string(alpha) + "_" + std::to_string(amplitude) + "_" + std::to_string(numberOfPatchPoints) + "_" + std::to_string(correctionTime) + "_" +  std::to_string(cycleNumber) + "_" +  std::to_string(correctorLevel) + "_stateHistory.txt");
    fileNameStringDeviations = ("L" + std::to_string(librationPointNr) + "_" + std::to_string(accelerationMagnitude)
                                + "_" + std::to_string(alpha) + "_" + std::to_string(amplitude) + "_" + std::to_string(numberOfPatchPoints) + "_" + std::to_string(correctionTime) + "_" +  std::to_string(cycleNumber) + "_" +  std::to_string(correctorLevel) + "_deviations.txt");
    fileNameStringPropagatedStates = ("L" + std::to_string(librationPointNr) + "_" + std::to_string(accelerationMagnitude)
                                + "_" + std::to_string(alpha) + "_" + std::to_string(amplitude) + "_" + std::to_string(numberOfPatchPoints) + "_" + std::to_string(correctionTime) + "_" +  std::to_string(cycleNumber) + "_" +  std::to_string(correctorLevel) + "_propagatedStates.txt");

    Eigen::VectorXd propagatedStates = propagatedStatesInclSTM.block(0,0,10*(numberOfPatchPoints-1),1);

    tudat::input_output::writeDataMapToTextFile( stateHistory, fileNameStringStateHistory, directoryString );
    tudat::input_output::writeMatrixToFile( deviationsAndDuration, fileNameStringDeviations, 16, directoryString);
    tudat::input_output::writeMatrixToFile( stateVector, fileNameStringStateVector, 16, directoryString);
    tudat::input_output::writeMatrixToFile( propagatedStates, fileNameStringPropagatedStates, 16, directoryString);

}

void computeMassVaryingDeviations(Eigen::VectorXd inputStateVector, const int numberOfPatchPoints, Eigen::MatrixXd& propagatedStatesInclSTM, Eigen::VectorXd& defectVector, std::map< double, Eigen::VectorXd >& stateHistory, const double massParameter)
{
    // declare full initial state vector for propagation
    Eigen::MatrixXd initialStateAndSTM = Eigen::MatrixXd::Zero(10,11);
    initialStateAndSTM.block(0,0,10,1) = inputStateVector.segment(0,10);
    initialStateAndSTM.block(0,1,10,10).setIdentity();

    // Propagate each patch point to the following time stamp and check deviations
    for (int i = 0; i < (numberOfPatchPoints - 1); i++ )
    {
        double initialTime = inputStateVector(i+(i+1)*10);
        double finalTime = inputStateVector((i+1)+(i+2)*10);

        std::pair< Eigen::MatrixXd, double > endStateAndSTMAndTime = propagateMassVaryingOrbitAugmentedToFinalCondition(
                    initialStateAndSTM, massParameter, finalTime, 1, stateHistory, 1000, initialTime);

        Eigen::MatrixXd endStateAndSTM      = endStateAndSTMAndTime.first;
        double endTime                  = endStateAndSTMAndTime.second;
        Eigen::VectorXd stateVectorOnly = endStateAndSTM.block( 0, 0, 10, 1 );


        // Select the begin state of the next segment
        initialStateAndSTM.setZero();
        initialStateAndSTM.block(0,0,10,1) = inputStateVector.segment(11*(i+1),10);
        initialStateAndSTM.block(0,1,10,10).setIdentity();

        // Compute defects by computing difference between initial state of next segment and end state of current segment
        if(i < numberOfPatchPoints - 2)
        {
            defectVector.segment(i*11,10) = inputStateVector.segment(11*(i+1),10) - stateVectorOnly; // state difference
            defectVector(i*11+10) = finalTime - endTime;

        } else
        {

            // inputStateVector.segment(11*(i+1),10) - stateVectorOnly << std::endl; state difference w.r.t terminal state:D

            // Compare the final state of propagated trajectory to initial patch point to have periodicity!
            defectVector.segment(i*11,10) = inputStateVector.segment(0,10) - stateVectorOnly;
            //defectVector.segment(i*11,10) = inputStateVector.segment(11*(i+1),10) - stateVectorOnly;
            defectVector(i*11+10) = finalTime - endTime;
            defectVector(i*11+9) = inputStateVector(11*(i+1)+9) - stateVectorOnly(9);
        }

        // Store the state and STM at the end of propagation in the propagatedStatesInclSTM matrix
        propagatedStatesInclSTM.block(10*i,0,10,11) = endStateAndSTM;

    }



}

Eigen::VectorXd computeMassVaryingDeviationNorms (const Eigen::VectorXd defectVector, const int numberOfPatchPoints )
{

    Eigen::VectorXd outputVector = Eigen::VectorXd::Zero(6);
    Eigen::VectorXd defectVectorAtPatchPoint = Eigen::VectorXd::Zero(11);
    Eigen::VectorXd positionDeviations(3*(numberOfPatchPoints-1));
    Eigen::VectorXd velocityDeviations(3*(numberOfPatchPoints-1));
    Eigen::VectorXd velocityInteriorDeviations(3*(numberOfPatchPoints-2));
    Eigen::VectorXd velocityExteriorDeviations(3);
    Eigen::VectorXd periodDeviations(numberOfPatchPoints-1);
    Eigen::VectorXd massDeviations(numberOfPatchPoints-1);

    for (int i = 0; i < (numberOfPatchPoints - 1); i++ ){

        defectVectorAtPatchPoint = defectVector.segment(i*11,11);

        positionDeviations.segment(i*3,3) = defectVectorAtPatchPoint.segment(0,3);
        velocityDeviations.segment(i*3,3) = defectVectorAtPatchPoint.segment(3,3);
        periodDeviations(i) =defectVectorAtPatchPoint(10);
        massDeviations(i) =defectVectorAtPatchPoint(9);


        if (i < (numberOfPatchPoints - 2) )
        {
                    velocityInteriorDeviations.segment(i*3,3) = defectVectorAtPatchPoint.segment(3,3);
           } else
        {
                    velocityExteriorDeviations = defectVectorAtPatchPoint.segment(3,3);
                }


    }

    outputVector(0) = positionDeviations.norm();
    outputVector(1) = velocityDeviations.norm();
    outputVector(2) = velocityInteriorDeviations.norm();
    outputVector(3) = velocityExteriorDeviations.norm();
    outputVector(4) = periodDeviations.norm();
    outputVector(5) = massDeviations.norm();

    return outputVector;

}

Eigen::VectorXd applyMassRefinement(const int librationPointNr,
                                            const Eigen::VectorXd& initialStateVector,
                                            const double massParameter, const int numberOfPatchPoints,
                                            double maxPositionDeviationFromPeriodicOrbit,
                                            double maxVelocityDeviationFromPeriodicOrbit, const double maxPeriodDeviationFromPeriodicOrbit,
                                            const int maxNumberOfIterations )
{

    std::cout << "\nAPPLY MASS REFINEMENT\n" << std::endl;
    // Declare variables relevant for sensitivity analysis
    double amplitude = 1.0E-4;
    double correctionTime = 0.05;
    double timeINIT = 0.0;
    double timeLI = 0.0;
    double timeLII = 0.0;


    // Declare relevant variables
    Eigen::VectorXd outputVector = Eigen::VectorXd::Zero(25+11*numberOfPatchPoints);
    Eigen::VectorXd currentTrajectoryGuess = initialStateVector;
    Eigen::MatrixXd propagatedStatesInclSTM = Eigen::MatrixXd::Zero(10*numberOfPatchPoints,11);
    Eigen::VectorXd defectVector =  Eigen::VectorXd::Zero(11* (numberOfPatchPoints - 1) );
    Eigen::VectorXd deviationNorms = Eigen::VectorXd::Zero(6);
    std::map< double, Eigen::VectorXd > stateHistory;
    stateHistory.clear();


    // ========= Compute current defects deviations by propagating the currentTrajectoryGuess in CR3BPLT with varying mass  ======= //
    computeMassVaryingDeviations(currentTrajectoryGuess, numberOfPatchPoints, propagatedStatesInclSTM, defectVector, stateHistory, massParameter);

    deviationNorms = computeMassVaryingDeviationNorms(defectVector, numberOfPatchPoints);

    double positionDeviationNorm = deviationNorms(0);
    double velocityTotalDeviationNorm = deviationNorms(1);
    double velocityInteriorDeviationNorm = deviationNorms(2);
    double velocityExteriorDeviationNorm = deviationNorms(3);
    double timeDeviationNorm = deviationNorms(4);
    double massDeviationNorm = deviationNorms(5);

    std::cout << "\nDeviations at the start of correction procedure" << std::endl
              << "Position deviation: " << positionDeviationNorm << std::endl
              << "Velocity deviation: " << velocityTotalDeviationNorm << std::endl
              << "Velocity int. deviation: " << velocityInteriorDeviationNorm << std::endl
              << "Velocity ext. deviation: " << velocityExteriorDeviationNorm << std::endl
              << "Time deviation: " << timeDeviationNorm << std::endl
              << "Mass deviation: " << massDeviationNorm << std::endl;

    writeMassRefinementDataToFile(librationPointNr, currentTrajectoryGuess(6), currentTrajectoryGuess(7), amplitude, numberOfPatchPoints, correctionTime,
                             stateHistory, currentTrajectoryGuess, deviationNorms, propagatedStatesInclSTM, 0, 0, 0, timeINIT);

    int numberOfCycles = 0;
    // ==== Apply Level 1 Correction ==== //
    while( positionDeviationNorm > maxPositionDeviationFromPeriodicOrbit or
           velocityTotalDeviationNorm > maxVelocityDeviationFromPeriodicOrbit or
           timeDeviationNorm > maxPeriodDeviationFromPeriodicOrbit ){

        int numberOfLevelICorrections = 0;

        auto startLI = std::chrono::high_resolution_clock::now();
        currentTrajectoryGuess = computeLevel1MassRefinementCorrection(defectVector, propagatedStatesInclSTM, currentTrajectoryGuess, numberOfPatchPoints, massParameter, numberOfLevelICorrections );

        stateHistory.clear();
        defectVector.setZero();
        propagatedStatesInclSTM.setZero();

        if (std::abs(currentTrajectoryGuess.norm()) < 1.0E-12 )
        {
            std::cout << " Level 1 did not converge within specified amount of iterations " << std::endl;
            outputVector = Eigen::VectorXd::Zero(25+11*numberOfPatchPoints);
            return outputVector;
        }

        if (numberOfCycles > maxNumberOfIterations )
        {
            std::cout << " Mass Refinement did not converge within specified amount of iterations " << std::endl;
            outputVector = Eigen::VectorXd::Zero(25+11*numberOfPatchPoints);
            return outputVector;
        }

        computeMassVaryingDeviations(currentTrajectoryGuess, numberOfPatchPoints, propagatedStatesInclSTM, defectVector, stateHistory, massParameter);


        deviationNorms = computeMassVaryingDeviationNorms(defectVector, numberOfPatchPoints);

        positionDeviationNorm = deviationNorms(0);
        velocityTotalDeviationNorm = deviationNorms(1);
        velocityInteriorDeviationNorm = deviationNorms(2);
        velocityExteriorDeviationNorm = deviationNorms(3);
        timeDeviationNorm = deviationNorms(4);
        massDeviationNorm = deviationNorms(5);

        std::cout << "\nLevel I Converged, remaining deviations are: " << std::endl
                  << "Position deviation: " << positionDeviationNorm << std::endl
                  << "Velocity deviation: " << velocityTotalDeviationNorm << std::endl
                  << "Velocity int. deviation: " << velocityInteriorDeviationNorm << std::endl
                  << "Velocity ext. deviation: " << velocityExteriorDeviationNorm << std::endl
                  << "Time deviation: " << timeDeviationNorm << std::endl
                  << "Mass deviation: " << massDeviationNorm << std::endl;

        auto stopLI = std::chrono::high_resolution_clock::now();
        auto durationLI = std::chrono::duration_cast<std::chrono::seconds>(stopLI - startLI);
        timeLI = durationLI.count();

        writeMassRefinementDataToFile(librationPointNr, currentTrajectoryGuess(6), currentTrajectoryGuess(7), amplitude, numberOfPatchPoints, correctionTime,
                                 stateHistory, currentTrajectoryGuess, deviationNorms, propagatedStatesInclSTM, numberOfCycles+1, 1, numberOfLevelICorrections, timeLI);

        if(positionDeviationNorm > maxPositionDeviationFromPeriodicOrbit or
                   velocityTotalDeviationNorm > maxVelocityDeviationFromPeriodicOrbit or
                timeDeviationNorm > maxPeriodDeviationFromPeriodicOrbit){

            // ==== Apply Level II Correction ==== //
            auto startLII = std::chrono::high_resolution_clock::now();

            Eigen::VectorXd sampleCorrection = computeLevel2Correction( defectVector, propagatedStatesInclSTM, currentTrajectoryGuess, Eigen::VectorXd::Zero(3), numberOfPatchPoints, massParameter);
            currentTrajectoryGuess = computeLevel2MassRefinementCorrection(defectVector, propagatedStatesInclSTM, currentTrajectoryGuess, numberOfPatchPoints, massParameter);

            stateHistory.clear();
            defectVector.setZero();
            propagatedStatesInclSTM.setZero();

            computeMassVaryingDeviations(currentTrajectoryGuess, numberOfPatchPoints, propagatedStatesInclSTM, defectVector, stateHistory, massParameter);

            deviationNorms = computeMassVaryingDeviationNorms(defectVector, numberOfPatchPoints);

            positionDeviationNorm = deviationNorms(0);
            velocityTotalDeviationNorm = deviationNorms(1);
            velocityInteriorDeviationNorm = deviationNorms(2);
            velocityExteriorDeviationNorm = deviationNorms(3);
            timeDeviationNorm = deviationNorms(4);
            massDeviationNorm = deviationNorms(5);

            std::cout << "\nLevel II Applied, remaining deviations are: " << std::endl
                      << "Position deviation: " << positionDeviationNorm << std::endl
                      << "Velocity deviation: " << velocityTotalDeviationNorm << std::endl
                      << "Velocity int. deviation: " << velocityInteriorDeviationNorm << std::endl
                      << "Velocity ext. deviation: " << velocityExteriorDeviationNorm << std::endl
                      << "Time deviation: " << timeDeviationNorm << std::endl
                      << "Mass deviation: " << massDeviationNorm << std::endl;

            auto stopLII = std::chrono::high_resolution_clock::now();
            auto durationLII = std::chrono::duration_cast<std::chrono::seconds>(stopLII - startLII);
            timeLII = durationLII.count();

            writeMassRefinementDataToFile(librationPointNr, currentTrajectoryGuess(6), currentTrajectoryGuess(7), amplitude, numberOfPatchPoints, correctionTime,
                                     stateHistory, currentTrajectoryGuess, deviationNorms, propagatedStatesInclSTM, numberOfCycles+1, 2, numberOfLevelICorrections, timeLII);
             numberOfCycles++;

        }



    }

    outputVector = currentTrajectoryGuess;

    return outputVector;
}

