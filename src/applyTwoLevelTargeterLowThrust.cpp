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
#include "propagateTLTOrbitAugmented.h"
#include "propagateMassVaryingOrbitAugmented.h"
#include "propagateOrbitAugmented.h"
#include "computeLevel1TLTCorrection.h"
#include "computeLevel2TLTCorrection.h"
#include "computeLevel2Correction.h"

#include "computeLevel2MassRefinementCorrection.h"



void writeTLTDataToFile(const int librationPointNr, const double accelerationMagnitude, const double alpha, const double amplitude, const int numberOfPatchPoints, const double correctionTime,
                              std::map< double, Eigen::VectorXd > stateHistory, const Eigen::VectorXd stateVector, Eigen::VectorXd deviations, const Eigen::MatrixXd propagatedStatesInclSTMThrust, const Eigen::MatrixXd propagatedStatesInclSTMCoast,
                              const int cycleNumber, const int correctorLevel, const int numberOfCorrections, const double correctionDuration )
{
    Eigen::VectorXd deviationsAndDuration = Eigen::VectorXd::Zero(7);
    deviationsAndDuration.segment(0,5) = deviations;
    deviationsAndDuration(5) = correctionDuration;
    deviationsAndDuration(6) = numberOfCorrections;

    std::string fileNameStringStateVector;
    std::string fileNameStringStateHistory;
    std::string fileNameStringDeviations;
    std::string fileNameStringPropagatedStatesThrust;
    std::string fileNameStringPropagatedStatesCoast;

    std::string directoryString = "../data/raw/two_level_targeter/";

    fileNameStringStateVector = ("L" + std::to_string(librationPointNr) + "_" + std::to_string(accelerationMagnitude)
                                 + "_" + std::to_string(alpha) + "_" + std::to_string(amplitude) + "_" + std::to_string(numberOfPatchPoints) + "_" + std::to_string(correctionTime) + "_" +  std::to_string(cycleNumber) + "_" +  std::to_string(correctorLevel) + "_stateVectors.txt");
    fileNameStringStateHistory = ("L" + std::to_string(librationPointNr) + "_" + std::to_string(accelerationMagnitude)
                                  + "_" + std::to_string(alpha) + "_" + std::to_string(amplitude) + "_" + std::to_string(numberOfPatchPoints) + "_" + std::to_string(correctionTime) + "_" +  std::to_string(cycleNumber) + "_" +  std::to_string(correctorLevel) + "_stateHistory.txt");
    fileNameStringDeviations = ("L" + std::to_string(librationPointNr) + "_" + std::to_string(accelerationMagnitude)
                                + "_" + std::to_string(alpha) + "_" + std::to_string(amplitude) + "_" + std::to_string(numberOfPatchPoints) + "_" + std::to_string(correctionTime) + "_" +  std::to_string(cycleNumber) + "_" +  std::to_string(correctorLevel) + "_deviations.txt");
    fileNameStringPropagatedStatesThrust = ("L" + std::to_string(librationPointNr) + "_" + std::to_string(accelerationMagnitude)
                                + "_" + std::to_string(alpha) + "_" + std::to_string(amplitude) + "_" + std::to_string(numberOfPatchPoints) + "_" + std::to_string(correctionTime) + "_" +  std::to_string(cycleNumber) + "_" +  std::to_string(correctorLevel) + "_propagatedStatesThrust.txt");
    fileNameStringPropagatedStatesCoast = ("L" + std::to_string(librationPointNr) + "_" + std::to_string(accelerationMagnitude)
                                + "_" + std::to_string(alpha) + "_" + std::to_string(amplitude) + "_" + std::to_string(numberOfPatchPoints) + "_" + std::to_string(correctionTime) + "_" +  std::to_string(cycleNumber) + "_" +  std::to_string(correctorLevel) + "_propagatedStatesCoast.txt");

    Eigen::VectorXd propagatedStatesThrust = propagatedStatesInclSTMThrust.block(0,0,10*(numberOfPatchPoints-1),1);
    Eigen::VectorXd propagatedStatesCoast = propagatedStatesInclSTMCoast.block(0,0,10*(numberOfPatchPoints-1),1);


    tudat::input_output::writeDataMapToTextFile( stateHistory, fileNameStringStateHistory, directoryString );
    tudat::input_output::writeMatrixToFile( deviationsAndDuration, fileNameStringDeviations, 16, directoryString);
    tudat::input_output::writeMatrixToFile( stateVector, fileNameStringStateVector, 16, directoryString);
    tudat::input_output::writeMatrixToFile( propagatedStatesThrust, fileNameStringPropagatedStatesThrust, 16, directoryString);
    tudat::input_output::writeMatrixToFile( propagatedStatesCoast, fileNameStringPropagatedStatesCoast, 16, directoryString);

}

Eigen::VectorXd rewriteInputGuess(const Eigen::VectorXd initialStateVector, const int numberOfPatchPoints)
{
    // define output vector
    Eigen::VectorXd outputVector = Eigen::VectorXd(12*numberOfPatchPoints);
    outputVector.setZero();
    extern double maximumThrust;

    for(int i = 0; i < numberOfPatchPoints; i++)
    {
        // Extract and compute needed information
        Eigen::VectorXd inputState = initialStateVector.segment(i*11,10);
        Eigen::VectorXd outputState = Eigen::VectorXd::Zero(12);
        double inputTime = initialStateVector(i*11+10);
        double inputTimeNextPoint = initialStateVector((i+1)*11+10);
        double gamma = std::asin(std::sqrt(  inputState(6) / maximumThrust ) );

        // add position and velocity state variables to output state
        outputState.segment(0,6) = inputState.segment(0,6);

        // add gamma to the output state
        outputState(6) = gamma;

        // add alpha, beta and mass to the output state
        outputState.segment(7,3) = inputState.segment(7,3);

        // Add next patch point time as burn point to have a coast arc;
        outputState(10) = inputTime;

        // Add next patch point time as burn point to have a coast arc;
        outputState(11) = inputTimeNextPoint;

        outputVector.segment(i*12,12) = outputState;

    }

    return outputVector;

}

void computeTwoLevelDeviations(Eigen::VectorXd inputStateVector, const int numberOfPatchPoints, Eigen::MatrixXd& propagatedStatesInclSTMCoast, Eigen::MatrixXd& propagatedStatesInclSTMThrust, Eigen::VectorXd& defectVector, std::map< double, Eigen::VectorXd >& stateHistory, const double massParameter)
{
    // declare full initial state vector for propagation
    Eigen::MatrixXd initialStateAndSTM = Eigen::MatrixXd::Zero(10,11);
    initialStateAndSTM.block(0,0,10,1) = inputStateVector.segment(0,10);
    initialStateAndSTM.block(0,1,10,10).setIdentity();

    for (int i = 0; i < (numberOfPatchPoints - 1); i++ )
    {
        // define start and end times for the thrust and coast arc
        double initialTimeThrust = inputStateVector(i*12+10); // current patch point time
        double finalTimeThrust = inputStateVector(i*12+11);
        double finalTimeCoast = inputStateVector((i+1)*12+10);  // next patch point time

        // Propagate the thrust arc
        std::pair< Eigen::MatrixXd, double > endStateAndSTMAndTimeThrust = propagateTLTOrbitAugmentedToFinalCondition(initialStateAndSTM, massParameter, finalTimeThrust, 1, stateHistory, 1000, initialTimeThrust);

        Eigen::MatrixXd endStateAndSTMThrust      = endStateAndSTMAndTimeThrust.first;
        double endTimeThrust                  = endStateAndSTMAndTimeThrust.second;
        Eigen::VectorXd stateVectorOnlyThrust = endStateAndSTMThrust.block( 0, 0, 10, 1 );

        // propagate the coast Arc
        Eigen::VectorXd initialStateVectorCoast = stateVectorOnlyThrust;
        initialStateVectorCoast(6) = 0.0;
        double initialTimeCoast = endTimeThrust;

        std::pair< Eigen::MatrixXd, double > endStateAndSTMAndTimeCoast = propagateTLTOrbitAugmentedToFinalCondition(getFullInitialStateAugmented(initialStateVectorCoast) , massParameter, finalTimeCoast, 1, stateHistory, 1000, initialTimeCoast);

        Eigen::MatrixXd endStateAndSTMCoast  = endStateAndSTMAndTimeCoast.first;
        double endTimeCoast                  = endStateAndSTMAndTimeCoast.second;
        Eigen::VectorXd stateVectorOnlyCoast = endStateAndSTMCoast.block( 0, 0, 10, 1 );

        // Store information in the propagatedStatesInclSTMThrust propagatedStatesInclSTMThrust
        propagatedStatesInclSTMThrust.block(10*i,0,10,11) = endStateAndSTMThrust;
        propagatedStatesInclSTMCoast.block(10*i,0,10,11) = endStateAndSTMCoast;

        // Select the begin state of the next segment
        initialStateAndSTM.setZero();
        initialStateAndSTM.block(0,0,10,1) = inputStateVector.segment(12*(i+1),10);
        initialStateAndSTM.block(0,1,10,10).setIdentity();

        // Compute deviations w.r.t to the next segment for final loop build in condition to substract w.r.t to final and initial state
        if(i < numberOfPatchPoints - 2)
        {
            defectVector.segment(i*12,10) = inputStateVector.segment(12*(i+1),10) - stateVectorOnlyCoast; // state difference
            defectVector(i*12+10) = finalTimeCoast - endTimeCoast;
            defectVector(i*12+11) = finalTimeThrust - endTimeThrust;


        }else
        {
            // inputStateVector.segment(11*(i+1),10) - stateVectorOnly << std::endl; state difference w.r.t terminal state:D

            // Compare the final state of propagated trajectory to initial patch point to have periodicity!
            defectVector.segment(i*12,10) = inputStateVector.segment(0,10) - stateVectorOnlyCoast;
            //defectVector.segment(i*12,10) = inputStateVector.segment(12*(i+1),10) - stateVectorOnlyCoast;
            defectVector(i*12+10) = finalTimeCoast - endTimeCoast;
            defectVector(i*12+11) = finalTimeThrust - endTimeThrust;

            // Mass continuity test
            defectVector(i*12+9) = inputStateVector(12*(i+1)+9) - stateVectorOnlyCoast(9);

        }


    }
}

Eigen::VectorXd computeTLTDeviationNorms (const Eigen::VectorXd defectVector, const int numberOfPatchPoints )
{

    Eigen::VectorXd outputVector = Eigen::VectorXd::Zero(6);
    Eigen::VectorXd defectVectorAtPatchPoint = Eigen::VectorXd::Zero(12);
    Eigen::VectorXd positionDeviations(3*(numberOfPatchPoints-1));
    Eigen::VectorXd velocityDeviations(3*(numberOfPatchPoints-1));
    Eigen::VectorXd velocityInteriorDeviations(3*(numberOfPatchPoints-2));
    Eigen::VectorXd velocityExteriorDeviations(3);
    Eigen::VectorXd periodDeviations(2*(numberOfPatchPoints-1));
    Eigen::VectorXd massDeviations(numberOfPatchPoints-1);

    for (int i = 0; i < (numberOfPatchPoints - 1); i++ ){

        defectVectorAtPatchPoint = defectVector.segment(i*12,12);

        positionDeviations.segment(i*3,3) = defectVectorAtPatchPoint.segment(0,3);
        velocityDeviations.segment(i*3,3) = defectVectorAtPatchPoint.segment(3,3);
        periodDeviations.segment(i*2,2) =defectVectorAtPatchPoint.segment(10,2);
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

Eigen::VectorXd applyTwoLevelTargeterLowThrust(const int librationPointNr,
                                            const Eigen::VectorXd& initialStateVector,
                                            const double massParameter, const int numberOfPatchPoints,
                                            double maxPositionDeviationFromPeriodicOrbit,
                                            double maxVelocityDeviationFromPeriodicOrbit, const double maxPeriodDeviationFromPeriodicOrbit,
                                            const int maxNumberOfIterations )
{
    std::cout << "\nAPPLY TLT-LT \n" << std::endl;
    double amplitude = 1.0E-4;
    double correctionTime = 0.05;
    double timeINIT = 0.0;
    double timeLI = 0.0;
    double timeLII = 0.0;
    extern double maximumThrust;


    // Declare relevant variables
    Eigen::VectorXd outputVector = Eigen::VectorXd::Zero(11*numberOfPatchPoints);
    Eigen::VectorXd currentTrajectoryGuess = Eigen::VectorXd(12*numberOfPatchPoints);
    Eigen::MatrixXd propagatedStatesInclSTMThrust = Eigen::MatrixXd::Zero(10*(numberOfPatchPoints-1),11);
    Eigen::MatrixXd propagatedStatesInclSTMCoast = Eigen::MatrixXd::Zero(10*(numberOfPatchPoints-1),11);
    Eigen::VectorXd defectVector =  Eigen::VectorXd::Zero(12* (numberOfPatchPoints - 1) );
    Eigen::VectorXd deviationNorms = Eigen::VectorXd::Zero(6);
    std::map< double, Eigen::VectorXd > stateHistory;
    stateHistory.clear();



    // ========= Rewrite stateVector into the desired form, replace f with gamma and add the burn points ====
    currentTrajectoryGuess = rewriteInputGuess(initialStateVector, numberOfPatchPoints);




    // ========= Compute current defects deviations by propagating the currentTrajectoryGuess in CR3BPLT with varying mass  ======= //
    computeTwoLevelDeviations(currentTrajectoryGuess, numberOfPatchPoints, propagatedStatesInclSTMCoast, propagatedStatesInclSTMThrust, defectVector, stateHistory, massParameter);

    deviationNorms = computeTLTDeviationNorms(defectVector, numberOfPatchPoints);

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

    writeTLTDataToFile(librationPointNr, initialStateVector(6), initialStateVector(7), amplitude, numberOfPatchPoints, correctionTime,
                             stateHistory, currentTrajectoryGuess, deviationNorms, propagatedStatesInclSTMThrust, propagatedStatesInclSTMCoast, 0, 0, 0, timeINIT);

    int numberOfCycles = 0;
    // ==== Apply Level 1 Correction ==== //

    while(  positionDeviationNorm > maxPositionDeviationFromPeriodicOrbit or
            velocityTotalDeviationNorm > maxVelocityDeviationFromPeriodicOrbit or
             timeDeviationNorm > maxPeriodDeviationFromPeriodicOrbit ){

        int numberOfLevelICorrections = 0;
        auto startLI = std::chrono::high_resolution_clock::now();
        currentTrajectoryGuess = computeLevel1TLTCorrection(defectVector, propagatedStatesInclSTMThrust, propagatedStatesInclSTMCoast, currentTrajectoryGuess, numberOfPatchPoints, massParameter, numberOfLevelICorrections );

        stateHistory.clear();
        defectVector.setZero();
        propagatedStatesInclSTMThrust.setZero();
        propagatedStatesInclSTMCoast.setZero();

        computeTwoLevelDeviations(currentTrajectoryGuess, numberOfPatchPoints, propagatedStatesInclSTMCoast, propagatedStatesInclSTMThrust, defectVector, stateHistory, massParameter);

        deviationNorms = computeTLTDeviationNorms(defectVector, numberOfPatchPoints);

        double positionDeviationNorm = deviationNorms(0);
        double velocityTotalDeviationNorm = deviationNorms(1);
        double velocityInteriorDeviationNorm = deviationNorms(2);
        double velocityExteriorDeviationNorm = deviationNorms(3);
        double timeDeviationNorm = deviationNorms(4);
        double massDeviationNorm = deviationNorms(5);

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

        writeTLTDataToFile(librationPointNr, initialStateVector(6), initialStateVector(7), amplitude, numberOfPatchPoints, correctionTime,
                                 stateHistory, currentTrajectoryGuess, deviationNorms, propagatedStatesInclSTMThrust, propagatedStatesInclSTMCoast, numberOfCycles+1, 1, numberOfLevelICorrections, timeLI);

        if(positionDeviationNorm > maxPositionDeviationFromPeriodicOrbit or
                   velocityTotalDeviationNorm > maxVelocityDeviationFromPeriodicOrbit or
                timeDeviationNorm > maxPeriodDeviationFromPeriodicOrbit){

            // ==== Apply Level II Correction ==== //
            auto startLII = std::chrono::high_resolution_clock::now();

            Eigen::VectorXd temporaryGuess = Eigen::VectorXd::Zero(11*numberOfPatchPoints);
            Eigen::VectorXd ballisticGuess = Eigen::VectorXd::Zero(11*numberOfPatchPoints);
            Eigen::VectorXd tempDeviation = Eigen::VectorXd::Zero(11*numberOfPatchPoints);
            Eigen::MatrixXd tempPropagatedStatesInclSTMThrust = Eigen::MatrixXd::Zero(10*(numberOfPatchPoints-1),11);


            for (int k = 0; k < numberOfPatchPoints; k++)
            {
                temporaryGuess.segment(k*11,11) = currentTrajectoryGuess.segment(k*12,11);
                tempDeviation.segment(k*11,11) = defectVector.segment(k*12,11);
                temporaryGuess(k*11+6) =  maximumThrust * std::sin(currentTrajectoryGuess(k*12+6) ) * std::sin(currentTrajectoryGuess(k*12+6) );

                if(k < numberOfPatchPoints -1)
                {
                    tempPropagatedStatesInclSTMThrust.block(10*k,0,10,11) = propagatedStatesInclSTMThrust.block(10*k,0,10,11);
                    tempPropagatedStatesInclSTMThrust(10*k+6,0) = maximumThrust * std::sin(propagatedStatesInclSTMThrust(10*k+6,0) ) * std::sin(propagatedStatesInclSTMThrust(10*k+6,0) );
                }
            }

//            std::cout << "tempGuess: \n" << temporaryGuess << std::endl;
//            std::cout << "currentTrajectoryGuess: \n" << currentTrajectoryGuess << std::endl;

            ballisticGuess = computeLevel2Correction(tempDeviation, tempPropagatedStatesInclSTMThrust, temporaryGuess, Eigen::VectorXd::Zero(3), numberOfPatchPoints, massParameter );
            temporaryGuess = computeLevel2MassRefinementCorrection(tempDeviation, tempPropagatedStatesInclSTMThrust, temporaryGuess, numberOfPatchPoints, massParameter );
            currentTrajectoryGuess = computeLevel2TLTCorrection(defectVector, propagatedStatesInclSTMThrust, propagatedStatesInclSTMCoast, currentTrajectoryGuess, numberOfPatchPoints, massParameter);

            stateHistory.clear();
            defectVector.setZero();
            propagatedStatesInclSTMThrust.setZero();
            propagatedStatesInclSTMCoast.setZero();

            //std::cout << "currentTrajectoryGuess: \n" << currentTrajectoryGuess << std::endl;

            computeTwoLevelDeviations(currentTrajectoryGuess, numberOfPatchPoints, propagatedStatesInclSTMCoast, propagatedStatesInclSTMThrust, defectVector, stateHistory, massParameter);

            deviationNorms = computeTLTDeviationNorms(defectVector, numberOfPatchPoints);

            double positionDeviationNorm = deviationNorms(0);
            double velocityTotalDeviationNorm = deviationNorms(1);
            double velocityInteriorDeviationNorm = deviationNorms(2);
            double velocityExteriorDeviationNorm = deviationNorms(3);
            double timeDeviationNorm = deviationNorms(4);
            double massDeviationNorm = deviationNorms(5);

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

            writeTLTDataToFile(librationPointNr, initialStateVector(6), initialStateVector(7), amplitude, numberOfPatchPoints, correctionTime,
                                     stateHistory, currentTrajectoryGuess, deviationNorms, propagatedStatesInclSTMThrust, propagatedStatesInclSTMCoast, numberOfCycles+1, 2, 0, timeLII);



        }


        numberOfCycles++;

    }



    return outputVector;
}
