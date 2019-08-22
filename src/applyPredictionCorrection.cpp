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
#include "computeLevel1Correction.h"
#include "computeLevel2Correction.h"
#include "propagateOrbitAugmented.h"
#include "propagateOrbit.h"
#include "stateDerivativeModel.h"
#include "stateDerivativeModelAugmented.h"

void shiftConvergedTrajectoryGuess(int librationPointNr, Eigen::VectorXd currentTrajectoryGuess, Eigen::VectorXd inputTrajectoryGuess, const Eigen::VectorXd offsetUnitVector, Eigen::VectorXd& convergedTrajectoryGuess, double massParameter, const int numberOfPatchPoints)
{


// Determine Target Angle
    double targetAngle;
    if (librationPointNr < 3)
    {
        targetAngle = atan2( inputTrajectoryGuess(1), inputTrajectoryGuess(0) - (1.0 - massParameter) ) * 180.0/tudat::mathematical_constants::PI;

    } else
    {
        targetAngle = atan2( inputTrajectoryGuess(1), inputTrajectoryGuess(0) - ( - massParameter) ) * 180.0/tudat::mathematical_constants::PI;;
    }
    if (targetAngle < 0)
    {
        targetAngle = targetAngle + 360.0;
    }



 // Determine Target Angle
    double startingAngle;
        if (librationPointNr < 3)
        {
            startingAngle = atan2( currentTrajectoryGuess(1), currentTrajectoryGuess(0) - (1.0 - massParameter) ) * 180.0/tudat::mathematical_constants::PI;;

        } else
        {
            startingAngle = atan2( currentTrajectoryGuess(1), currentTrajectoryGuess(0) - ( - massParameter) ) * 180.0/tudat::mathematical_constants::PI;;
        }
        if (startingAngle < 0)
        {
            startingAngle = startingAngle + 360.0;
        }

 // Determine integration Direction
    int direction;
        if (startingAngle - targetAngle > 0 )
        {
            if ( librationPointNr == 1)
            {
                direction = 1;
            }
            if (librationPointNr == 2)
            {
                direction = -1;
            }
        } else if (startingAngle - targetAngle < 0 )
        {
            if ( librationPointNr == 1)
            {
                direction = -1;
            }
            if (librationPointNr == 2)
            {
                direction = 1;
            }

        } else
        {
            convergedTrajectoryGuess = currentTrajectoryGuess;
            return;
        }



    Eigen::MatrixXd initialStateAndSTM = Eigen::MatrixXd::Zero(10,11);
    initialStateAndSTM.block(0,0,10,1) = currentTrajectoryGuess.segment(0,10);
    initialStateAndSTM.block(0,1,10,10).setIdentity();

    std::map< double, Eigen::VectorXd > stateHistoryShift2;
    std::pair< Eigen::MatrixXd, double > shiftedStateInclSTMandTime = propagateOrbitAugmentedToFullRevolutionCondition( initialStateAndSTM,librationPointNr,
                                                                                                  massParameter, targetAngle, direction, stateHistoryShift2, -1
                                                                                                  ,currentTrajectoryGuess(10) );

    Eigen::MatrixXd shiftedStateInclSTM = shiftedStateInclSTMandTime.first;
    double shiftedTime                  = shiftedStateInclSTMandTime.second;
    Eigen::VectorXd stateVectorOnly     = shiftedStateInclSTM.block( 0, 0, 10, 1 );

    // determine shifting Time
    double timeOfIntegration = shiftedTime - currentTrajectoryGuess(10);

    convergedTrajectoryGuess.segment(0,10) = stateVectorOnly;
    convergedTrajectoryGuess(10) = shiftedTime;

    double shiftedAngle;
        if (librationPointNr < 3)
        {
            shiftedAngle = atan2( convergedTrajectoryGuess(1), convergedTrajectoryGuess(0) - (1.0 - massParameter) );

        } else
        {
            shiftedAngle = atan2( convergedTrajectoryGuess(1), convergedTrajectoryGuess(0) - ( - massParameter) );
        }



    convergedTrajectoryGuess.segment(11*(numberOfPatchPoints-1),10) = stateVectorOnly;
    convergedTrajectoryGuess(11*(numberOfPatchPoints-1)+10) = currentTrajectoryGuess(11*(numberOfPatchPoints-1)+10) + timeOfIntegration;


    for (int i = 1; i < (numberOfPatchPoints - 1); i++ )
    {
        // Select the begin state of the next segment
        initialStateAndSTM.setZero();
        initialStateAndSTM.block(0,0,10,1) = currentTrajectoryGuess.segment(11*i,10);
        initialStateAndSTM.block(0,1,10,10).setIdentity();

        double initialTime = currentTrajectoryGuess(i*11+10);
        double finalTime = initialTime + timeOfIntegration;

        std::map< double, Eigen::VectorXd > stateHistoryShift;
        std::pair< Eigen::MatrixXd, double > endStateAndSTMAndTime = propagateOrbitAugmentedToFinalConditionSmallIntervals(
                    initialStateAndSTM, massParameter, finalTime, direction, stateHistoryShift, -1, initialTime);

        Eigen::MatrixXd endStateAndSTM      = endStateAndSTMAndTime.first;
        double endTime                  = endStateAndSTMAndTime.second;
        Eigen::VectorXd stateVectorOnly = endStateAndSTM.block( 0, 0, 10, 1 );

        convergedTrajectoryGuess.segment(i*11,10) = stateVectorOnly;
        convergedTrajectoryGuess(i*11+10) = endTime;

    }

    // shift the time of all patch points so initial patch point time is zero
    double deltaTime = convergedTrajectoryGuess(10);
    for (int i = 0; i < (numberOfPatchPoints ); i++ )
    {

          convergedTrajectoryGuess(11*i+10) = convergedTrajectoryGuess(11*i+10);

    }

}

void writeCorrectorDataToFile(const int librationPointNr, const double accelerationMagnitude, const double alpha, const double amplitude, const int numberOfPatchPoints, const double correctionTime,
                              std::map< double, Eigen::VectorXd > stateHistory, const Eigen::VectorXd stateVector, Eigen::VectorXd deviations, const Eigen::MatrixXd propagatedStatesInclSTM,
                              const int cycleNumber, const int correctorLevel, const double numberOfCorrections, const double correctionDuration )
{
//std::cout << "\n== check input of writing function: " << std::endl
//          << "libPointNr: " << librationPointNr << std::endl
//          << "alt: " << accelerationMagnitude << std::endl
//          << "alpha: " << alpha << std::endl
//          << "A: " << amplitude << std::endl
//          << "stateHistory size: " << stateHistory.size() << std::endl
//          << "stateVector: \n" << stateVector << std::endl
//          << "deviations: \n" << deviations << std::endl
//          << "cycleNumber: " << cycleNumber << std::endl
//          << "correctorLevel: " << correctorLevel << std::endl
//          << "numberOfCorrections: " << numberOfCorrections << std::endl
//          << "correctionDuration: " << correctionDuration << std::endl;

    Eigen::VectorXd deviationsAndDuration = Eigen::VectorXd::Zero(7);
    deviationsAndDuration.segment(0,5) = deviations;
    deviationsAndDuration(5) = correctionDuration;
    deviationsAndDuration(6) = numberOfCorrections;

    std::string fileNameStringStateVector;
    std::string fileNameStringStateHistory;
    std::string fileNameStringDeviations;
    std::string fileNameStringPropagatedStates;



    std::string directoryString = "../data/raw/tlt_corrector/";

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

Eigen::VectorXd computeDeviationNorms (const Eigen::VectorXd defectVector, const int numberOfPatchPoints )
{

    Eigen::VectorXd outputVector = Eigen::VectorXd::Zero(5);
    Eigen::VectorXd defectVectorAtPatchPoint = Eigen::VectorXd::Zero(11);
    Eigen::VectorXd positionDeviations(3*(numberOfPatchPoints-1));
    Eigen::VectorXd velocityDeviations(3*(numberOfPatchPoints-1));
    Eigen::VectorXd velocityInteriorDeviations(3*(numberOfPatchPoints-2));
    Eigen::VectorXd velocityExteriorDeviations(3);
    Eigen::VectorXd periodDeviations(numberOfPatchPoints-1);


    for (int i = 0; i < (numberOfPatchPoints - 1); i++ ){

        defectVectorAtPatchPoint = defectVector.segment(i*11,11);

        positionDeviations.segment(i*3,3) = defectVectorAtPatchPoint.segment(0,3);
        velocityDeviations.segment(i*3,3) = defectVectorAtPatchPoint.segment(3,3);
        periodDeviations(i) =defectVectorAtPatchPoint(10);

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

    return outputVector;

}

void computeOrbitDeviations(Eigen::VectorXd inputStateVector, const int numberOfPatchPoints, Eigen::MatrixXd& propagatedStatesInclSTM, Eigen::VectorXd& defectVector, std::map< double, Eigen::VectorXd >& stateHistory, const double massParameter  )
{

    Eigen::MatrixXd initialStateAndSTM = Eigen::MatrixXd::Zero(10,11);
    initialStateAndSTM.block(0,0,10,1) = inputStateVector.segment(0,10);
    initialStateAndSTM.block(0,1,10,10).setIdentity();

    for (int i = 0; i < (numberOfPatchPoints - 1); i++ )
    {
        double initialTime = inputStateVector(i+(i+1)*10);
        double finalTime = inputStateVector((i+1)+(i+2)*10);

        std::pair< Eigen::MatrixXd, double > endStateAndSTMAndTime = propagateOrbitAugmentedToFinalCondition(
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


            // Compare the final state of propagated trajectory to initial patch point to have periodicity!
            defectVector.segment(i*11,10) = inputStateVector.segment(0,10) - stateVectorOnly;
            //defectVector.segment(i*11,10) = inputStateVector.segment(11*(i+1),10) - stateVectorOnly;

            defectVector(i*11+10) = finalTime - endTime;
        }



        // Store the state and STM at the end of propagation in the propagatedStatesInclSTM matrix
        propagatedStatesInclSTM.block(10*i,0,10,11) = endStateAndSTM;

       }

}

Eigen::VectorXd applyPredictionCorrection(const int librationPointNr,
                                            const Eigen::VectorXd& initialStateVector,
                                            const double massParameter, const int numberOfPatchPoints,
                                            double maxPositionDeviationFromPeriodicOrbit,
                                            double maxVelocityDeviationFromPeriodicOrbit, const double maxPeriodDeviationFromPeriodicOrbit,
                                            const int maxNumberOfIterations )
{
    std::cout << "\nAPPLY PREDICTION CORRECTION\n" << std::endl;
    double amplitude = 1.0E-1;
    double correctionTime = 0.05;
    double timeINIT = 0.0;
    double timeLI = 0.0;
    double timeLII = 0.0;

    // Determine unitVector
    Eigen::VectorXd offsetUnitVector = Eigen::VectorXd::Zero(3);
    offsetUnitVector = (initialStateVector.segment(0,3)).normalized();

//    std::cout << "offsetUnitVector: \n" << offsetUnitVector << std::endl
//               << "positionVector: \n" << initialStateVector.segment(0,3) << std::endl
//               << "offsetUnitVector NORM: " << offsetUnitVector.norm() << std::endl
//               << "positionVector NORM: " << initialStateVector.segment(0,3).norm() << std::endl
//               << "inner product offsetUnit: " << offsetUnitVector.transpose() * initialStateVector.segment(3,3) << std::endl
//               << "inner product posVel: " << initialStateVector.segment(0,3).transpose() * initialStateVector.segment(3,3)  << std::endl;

    // == Define the relevant variables == //
    Eigen::VectorXd outputVector = Eigen::VectorXd::Zero(25 + (11 * numberOfPatchPoints));
    Eigen::VectorXd currentTrajectoryGuess =  Eigen::VectorXd::Zero(11*numberOfPatchPoints);
    Eigen::VectorXd convergedTrajectoryGuess =  Eigen::VectorXd::Zero(11*numberOfPatchPoints);
    Eigen::MatrixXd propagatedStatesInclSTM = Eigen::MatrixXd::Zero(10*numberOfPatchPoints,11);
    Eigen::VectorXd defectVector =  Eigen::VectorXd::Zero(11* (numberOfPatchPoints - 1) );
    Eigen::VectorXd deviationNorms = Eigen::VectorXd::Zero(5);
    std::map< double, Eigen::VectorXd > stateHistory;
    stateHistory.clear();

    currentTrajectoryGuess = initialStateVector;

    // ========= Compute current defects deviations by propagating the initialStateVector in CR3BPLT  ======= //
    auto startINIT = std::chrono::high_resolution_clock::now();
    computeOrbitDeviations(currentTrajectoryGuess, numberOfPatchPoints, propagatedStatesInclSTM, defectVector, stateHistory, massParameter);

    deviationNorms = computeDeviationNorms(defectVector, numberOfPatchPoints);

    double positionDeviationNorm = deviationNorms(0);
    double velocityTotalDeviationNorm = deviationNorms(1);
    double velocityInteriorDeviationNorm = deviationNorms(2);
    double velocityExteriorDeviationNorm = deviationNorms(3);
    double timeDeviationNorm = deviationNorms(4);

    std::cout << "\nDeviations at the start of correction procedure" << std::endl
              << "Position deviation: " << positionDeviationNorm << std::endl
              << "Velocity deviation: " << velocityTotalDeviationNorm << std::endl
              << "Velocity int. deviation: " << velocityInteriorDeviationNorm << std::endl
              << "Velocity ext. deviation: " << velocityExteriorDeviationNorm << std::endl
              << "Time deviation: " << timeDeviationNorm << std::endl;

    auto stopINIT = std::chrono::high_resolution_clock::now();
    auto durationINIT = std::chrono::duration_cast<std::chrono::seconds>(stopINIT - startINIT);
    timeINIT = durationINIT.count();

    writeCorrectorDataToFile(librationPointNr, currentTrajectoryGuess(6), currentTrajectoryGuess(7), amplitude, numberOfPatchPoints, correctionTime,
                             stateHistory, currentTrajectoryGuess, deviationNorms, propagatedStatesInclSTM, 0, 0, 0, timeINIT);

    int numberOfCorrections = 0;
    while (positionDeviationNorm > maxPositionDeviationFromPeriodicOrbit or
           velocityTotalDeviationNorm > maxVelocityDeviationFromPeriodicOrbit or
           timeDeviationNorm > maxPeriodDeviationFromPeriodicOrbit ){

        std::cout << "\nSTART TLT CORRECTION CYCLE " << numberOfCorrections + 1 << "." << std::endl;

        if( numberOfCorrections > maxNumberOfIterations ){

            std::cout << "Predictor Corrector did not converge within maxNumberOfIterations" << std::endl;
            return outputVector = Eigen::VectorXd::Zero(25+11*numberOfPatchPoints);
        }

        // ==== Apply Level 1 Correction ==== //

        int numberOfCorrectionsLevel1 = 0;
        bool applyLevel1Correction = false;

        auto startLI = std::chrono::high_resolution_clock::now();
        while (positionDeviationNorm > maxPositionDeviationFromPeriodicOrbit or applyLevel1Correction){

            Eigen::VectorXd correctionVectorLevel1 = Eigen::VectorXd::Zero(11*numberOfPatchPoints);
            correctionVectorLevel1 = computeLevel1Correction(defectVector, propagatedStatesInclSTM, currentTrajectoryGuess, numberOfPatchPoints );

            //std::cout << "correctionVectorLevel1: " << correctionVectorLevel1 << std::endl;

            currentTrajectoryGuess = currentTrajectoryGuess + correctionVectorLevel1;

            // ========= Compute the defects of the corrected trajectory  ======= //
            defectVector.setZero();
            propagatedStatesInclSTM.setZero();
            stateHistory.clear();
            computeOrbitDeviations(currentTrajectoryGuess, numberOfPatchPoints, propagatedStatesInclSTM, defectVector, stateHistory, massParameter);

            deviationNorms = computeDeviationNorms(defectVector, numberOfPatchPoints);

            positionDeviationNorm = deviationNorms(0);
            velocityTotalDeviationNorm = deviationNorms(1);
            velocityInteriorDeviationNorm = deviationNorms(2);
            velocityExteriorDeviationNorm = deviationNorms(3);
            timeDeviationNorm = deviationNorms(4);

            std::cout << "\nLevel I Correction applied, remaining deviations are: " << std::endl
                      << "Level 1 Corrections: " << numberOfCorrectionsLevel1 + 1 << std::endl
                      << "Position deviation: " << positionDeviationNorm << std::endl
                      << "Velocity deviation: " << velocityTotalDeviationNorm << std::endl
                      << "Velocity int. deviation: " << velocityInteriorDeviationNorm << std::endl
                      << "Velocity ext. deviation: " << velocityExteriorDeviationNorm << std::endl
                      << "Time deviation: " << timeDeviationNorm << std::endl;

            numberOfCorrectionsLevel1++;
            applyLevel1Correction = false;

        }
        auto stopLI = std::chrono::high_resolution_clock::now();
        auto durationLI = std::chrono::duration_cast<std::chrono::seconds>(stopLI - startLI);
        timeLI = durationLI.count();

        std::cout << "\nLevel I Converged after " << numberOfCorrectionsLevel1 << " corrections" << std::endl;

        writeCorrectorDataToFile(librationPointNr, currentTrajectoryGuess(6), currentTrajectoryGuess(7), amplitude, numberOfPatchPoints, correctionTime,
                                 stateHistory, currentTrajectoryGuess, deviationNorms, propagatedStatesInclSTM, numberOfCorrections+1, 1, numberOfCorrectionsLevel1, timeLI);


        // ========= Apply Level II correction if all constraints are not met  ======= //

        if( positionDeviationNorm > maxPositionDeviationFromPeriodicOrbit or
                   velocityTotalDeviationNorm > maxVelocityDeviationFromPeriodicOrbit or
                timeDeviationNorm > maxPeriodDeviationFromPeriodicOrbit )
        {

            auto startLII = std::chrono::high_resolution_clock::now();

            Eigen::VectorXd correctionVectorLevel2 = Eigen::VectorXd::Zero(11*numberOfPatchPoints);
            correctionVectorLevel2 = computeLevel2Correction(defectVector, propagatedStatesInclSTM, currentTrajectoryGuess, offsetUnitVector, numberOfPatchPoints, massParameter );

            currentTrajectoryGuess = currentTrajectoryGuess + correctionVectorLevel2;

            // ========= Compute the defects of the corrected trajectory  ======= //
            defectVector.setZero();
            propagatedStatesInclSTM.setZero();
            stateHistory.clear();
            computeOrbitDeviations(currentTrajectoryGuess, numberOfPatchPoints, propagatedStatesInclSTM, defectVector, stateHistory, massParameter);

            deviationNorms = computeDeviationNorms(defectVector, numberOfPatchPoints);

            positionDeviationNorm = deviationNorms(0);
            velocityTotalDeviationNorm = deviationNorms(1);
            velocityInteriorDeviationNorm = deviationNorms(2);
            velocityExteriorDeviationNorm = deviationNorms(3);
            timeDeviationNorm = deviationNorms(4);

            std::cout << "\nLevel II Correction applied, remaining deviations are: " << std::endl
                      << "Position deviation: " << positionDeviationNorm << std::endl
                      << "Velocity deviation: " << velocityTotalDeviationNorm << std::endl
                      << "Velocity int. deviation: " << velocityInteriorDeviationNorm << std::endl
                      << "Velocity ext. deviation: " << velocityExteriorDeviationNorm << std::endl
                      << "Time deviation: " << timeDeviationNorm << std::endl;

            if (positionDeviationNorm < maxPositionDeviationFromPeriodicOrbit)
            {
                applyLevel1Correction = true;
            }

            auto stopLII = std::chrono::high_resolution_clock::now();
            auto durationLII = std::chrono::duration_cast<std::chrono::seconds>(stopLII - startLII);
            timeLII = durationLII.count();

            writeCorrectorDataToFile(librationPointNr, currentTrajectoryGuess(6), currentTrajectoryGuess(7), amplitude, numberOfPatchPoints, correctionTime,
                                     stateHistory, currentTrajectoryGuess, deviationNorms, propagatedStatesInclSTM, numberOfCorrections+1, 2, 0, timeLII);

        }


        numberOfCorrections++;
    }

    std::cout << "\nTRAJECTORY CONVERGED AFTER " << numberOfCorrections << " TLT CORRECTION CYCLES, REMAINING DEVIATIONS: "<< std::endl
              << "Position deviation: " << positionDeviationNorm << std::endl
              << "Velocity deviation: " << velocityTotalDeviationNorm << std::endl
              << "Velocity int. deviation: " << velocityInteriorDeviationNorm << std::endl
              << "Velocity ext. deviation: " << velocityExteriorDeviationNorm << std::endl
              << "Time deviation: " << timeDeviationNorm << std::endl;


    // Reshift first patch point to time zero and check phase condition

    currentTrajectoryGuess.segment(11*(numberOfPatchPoints-1), 10) = currentTrajectoryGuess.segment(0,10);

    //std::cout << "output DiffCor First State: \n" << currentTrajectoryGuess.segment(0,11) << std::endl;
    //std::cout << "output DiffCor final State: \n" << currentTrajectoryGuess.segment(11*(numberOfPatchPoints-1),11) << std::endl;
    //std::cout << "Difference Initial and final PP : \n" << currentTrajectoryGuess.segment(0,11) - currentTrajectoryGuess.segment(11*(numberOfPatchPoints-1),11) << std::endl;



    //std::cout << "defectVector after shifting: \n" << defectVector << std::endl;


    shiftConvergedTrajectoryGuess(librationPointNr, currentTrajectoryGuess, initialStateVector, offsetUnitVector, convergedTrajectoryGuess, massParameter, numberOfPatchPoints );

    //std::cout << "output shifted First State: \n" << convergedTrajectoryGuess.segment(0,11) << std::endl;
    //std::cout << "output shifted Final State: \n" << convergedTrajectoryGuess.segment(11*(numberOfPatchPoints-1),11) << std::endl;




    // ========= Compute the defects of the corrected trajectory  ======= //
    defectVector.setZero();
    propagatedStatesInclSTM.setZero();
    stateHistory.clear();
    computeOrbitDeviations(convergedTrajectoryGuess, numberOfPatchPoints, propagatedStatesInclSTM, defectVector, stateHistory, massParameter);

    deviationNorms = computeDeviationNorms(defectVector, numberOfPatchPoints);

    positionDeviationNorm = deviationNorms(0);
    velocityTotalDeviationNorm = deviationNorms(1);
    velocityInteriorDeviationNorm = deviationNorms(2);
    velocityExteriorDeviationNorm = deviationNorms(3);
    timeDeviationNorm = deviationNorms(4);

    //std::cout << "defectVector after shifting: \n" << defectVector << std::endl;

    std::cout << "\nDEVIATIONS AFTER SHIFTING: " << std::endl
              << "Position deviation: " << positionDeviationNorm << std::endl
              << "Velocity deviation: " << velocityTotalDeviationNorm << std::endl
              << "Velocity int. deviation: " << velocityInteriorDeviationNorm << std::endl
              << "Velocity ext. deviation: " << velocityExteriorDeviationNorm << std::endl
              << "Time deviation: " << timeDeviationNorm << std::endl;


    // Store relevant info in the outputVector

    Eigen::VectorXd initialCondition = convergedTrajectoryGuess.segment(0,10);
    Eigen::VectorXd finalCondition   = propagatedStatesInclSTM.block(10*(numberOfPatchPoints-2),0,10,1);
    double orbitalPeriod = convergedTrajectoryGuess(11*(numberOfPatchPoints-1) + 10) - convergedTrajectoryGuess(10);

    double hamiltonianInitialCondition  = computeHamiltonian( massParameter, initialCondition);
    double hamiltonianEndState          = computeHamiltonian( massParameter, finalCondition  );

    // The output vector consists of:
    // 1. Corrected initial state vector, including orbital period and energy
    // 2. Full period state vector, including currentTime of integration and energy
    // 3. numberOfIterations
    // 4. the complete shifted converged guess

    outputVector = Eigen::VectorXd::Zero(25+11*numberOfPatchPoints);
    outputVector.segment(0,10) = initialCondition;
    outputVector(10) = orbitalPeriod;
    outputVector(11) = hamiltonianInitialCondition;
    outputVector.segment(12,10) = finalCondition;
    outputVector(22) = convergedTrajectoryGuess(11*(numberOfPatchPoints) + 10);
    outputVector(23) = hamiltonianEndState;
    outputVector(24) = numberOfCorrections;
    outputVector.segment(25,11*numberOfPatchPoints) = convergedTrajectoryGuess;

    return outputVector;
}

// test deviations at full perio
//    std::map< double, Eigen::VectorXd > stateHistoryShift;
//    std::pair< Eigen::MatrixXd, double > endStateAndSTMAndTime = propagateOrbitAugmentedToFinalCondition(
//                getFullInitialStateAugmented(currentTrajectoryGuess.segment(0,10)), massParameter,
//                currentTrajectoryGuess(11*(numberOfPatchPoints-1)+10), 1, stateHistoryShift, -1, currentTrajectoryGuess(10));

//    Eigen::MatrixXd endStateAndSTM = endStateAndSTMAndTime.first;
//    double endTime = endStateAndSTMAndTime.second;
//    Eigen::MatrixXd stateVectorEnd = endStateAndSTM.block(0,0,10,1);

//    std::pair< Eigen::MatrixXd, double > endStateAndSTMAndTimeShift = propagateOrbitAugmentedToFinalCondition(
//                getFullInitialStateAugmented(convergedTrajectoryGuess.segment(0,10)), massParameter,
//                convergedTrajectoryGuess(11*(numberOfPatchPoints-1)+10), 1, stateHistoryShift, -1, convergedTrajectoryGuess(10));

//    Eigen::MatrixXd endStateAndSTMShift = endStateAndSTMAndTimeShift.first;
//    double endTimeShift = endStateAndSTMAndTimeShift.second;
//    Eigen::MatrixXd stateVectorEndShift = endStateAndSTMShift.block(0,0,10,1);

//    std::cout << "\n=== Check the unshifted error at full period: == " << std::endl
//              << "finalTime: " << endTime << std::endl
//              << "deviation of finTime: " << currentTrajectoryGuess(11*(numberOfPatchPoints-1)+10) - endTime << std::endl
//              << "deviation between initial and final State: \n" << currentTrajectoryGuess.segment(0,10) - stateVectorEnd << std::endl;

//    std::cout << "\n=== Check the unshifted error at full period: == " << std::endl
//              << "finalTime: " << endTimeShift << std::endl
//              << "deviation of finTime: " << convergedTrajectoryGuess(11*(numberOfPatchPoints-1)+10) - endTimeShift << std::endl
//              << "deviation between initial and final State: \n" << convergedTrajectoryGuess.segment(0,10) - stateVectorEndShift << std::endl;
