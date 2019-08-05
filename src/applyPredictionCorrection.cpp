#include <cmath>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Dense>

#include <chrono>

#include <boost/function.hpp>

#include "Tudat/Astrodynamics/BasicAstrodynamics/celestialBodyConstants.h"
#include "Tudat/Astrodynamics/Gravitation/librationPoint.h"
#include "Tudat/Astrodynamics/Gravitation/jacobiEnergy.h"

#include "createLowThrustInitialConditions.h"
#include "computeLevel1Correction.h"
#include "computeLevel2Correction.h"
#include "propagateOrbitAugmented.h"
#include "propagateOrbit.h"
#include "stateDerivativeModel.h"
#include "stateDerivativeModelAugmented.h"

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

void computeOrbitDeviations(Eigen::VectorXd inputStateVector, const int numberOfPatchPoints, Eigen::MatrixXd& propagatedStatesInclSTM, Eigen::VectorXd& defectVector, const double massParameter  )
{

    Eigen::MatrixXd initialStateAndSTM = Eigen::MatrixXd::Zero(10,11);
    initialStateAndSTM.block(0,0,10,1) = inputStateVector.segment(0,10);
    initialStateAndSTM.block(0,1,10,10).setIdentity();

    for (int i = 0; i < (numberOfPatchPoints - 1); i++ )
    {
        double initialTime = inputStateVector(i+(i+1)*10);
        double finalTime = inputStateVector((i+1)+(i+2)*10);

        std::map< double, Eigen::VectorXd > segmentStateHistory;
        segmentStateHistory.clear();


        std::pair< Eigen::MatrixXd, double > endStateAndSTMAndTime = propagateOrbitAugmentedToFinalCondition(
                    initialStateAndSTM, massParameter, finalTime, 1, segmentStateHistory, -1, initialTime);

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

    // == Define the relevant variables == //
    Eigen::VectorXd outputVector = Eigen::VectorXd::Zero(25 + (11 * numberOfPatchPoints));
    Eigen::VectorXd currentTrajectoryGuess =  Eigen::VectorXd::Zero(11*numberOfPatchPoints);
    Eigen::MatrixXd propagatedStatesInclSTM = Eigen::MatrixXd::Zero(10*numberOfPatchPoints,11);
    Eigen::VectorXd defectVector =  Eigen::VectorXd::Zero(11* (numberOfPatchPoints - 1) );
    Eigen::VectorXd deviationNorms = Eigen::VectorXd::Zero(5);

    currentTrajectoryGuess = initialStateVector;

    // ========= Compute current defects deviations by propagating the initialStateVector in CR3BPLT  ======= //
    computeOrbitDeviations(currentTrajectoryGuess, numberOfPatchPoints, propagatedStatesInclSTM, defectVector, massParameter);

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

        while (positionDeviationNorm > maxPositionDeviationFromPeriodicOrbit or applyLevel1Correction){

            Eigen::VectorXd correctionVectorLevel1 = Eigen::VectorXd::Zero(11*numberOfPatchPoints);
            correctionVectorLevel1 = computeLevel1Correction(defectVector, propagatedStatesInclSTM, currentTrajectoryGuess, numberOfPatchPoints );

            //std::cout << "correctionVectorLevel1: " << correctionVectorLevel1 << std::endl;

            currentTrajectoryGuess = currentTrajectoryGuess + correctionVectorLevel1;

            // ========= Compute the defects of the corrected trajectory  ======= //
            defectVector.setZero();
            propagatedStatesInclSTM.setZero();
            computeOrbitDeviations(currentTrajectoryGuess, numberOfPatchPoints, propagatedStatesInclSTM, defectVector, massParameter);

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

        std::cout << "\nLevel I Converged after " << numberOfCorrectionsLevel1 << " corrections" << std::endl;

        // ========= Apply Level II correction if all constraints are not met  ======= //

        if( positionDeviationNorm > maxPositionDeviationFromPeriodicOrbit or
                   velocityTotalDeviationNorm > maxVelocityDeviationFromPeriodicOrbit or
                timeDeviationNorm > maxPeriodDeviationFromPeriodicOrbit )
        {

            Eigen::VectorXd correctionVectorLevel2 = Eigen::VectorXd::Zero(11*numberOfPatchPoints);
            correctionVectorLevel2 = computeLevel2Correction(defectVector, propagatedStatesInclSTM, currentTrajectoryGuess, numberOfPatchPoints, massParameter );

            currentTrajectoryGuess = currentTrajectoryGuess + correctionVectorLevel2;

            // ========= Compute the defects of the corrected trajectory  ======= //
            defectVector.setZero();
            propagatedStatesInclSTM.setZero();
            computeOrbitDeviations(currentTrajectoryGuess, numberOfPatchPoints, propagatedStatesInclSTM, defectVector, massParameter);

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
        }

        numberOfCorrections++;
    }

    std::cout << "\nTRAJECTORY CONVERGED AFTER " << numberOfCorrections << " TLT CORRECTION CYCLES, REMAINING DEVIATIONS: "<< std::endl
              << "Position deviation: " << positionDeviationNorm << std::endl
              << "Velocity deviation: " << velocityTotalDeviationNorm << std::endl
              << "Velocity int. deviation: " << velocityInteriorDeviationNorm << std::endl
              << "Velocity ext. deviation: " << velocityExteriorDeviationNorm << std::endl
              << "Time deviation: " << timeDeviationNorm << std::endl;




    outputVector = Eigen::VectorXd::Zero(25+11*numberOfPatchPoints);
    return outputVector;



}

