#include <fstream>
#include <iomanip>
#include <iostream>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>
#include <Eigen/Dense>
#include <boost/function.hpp>

#include "Tudat/Astrodynamics/BasicAstrodynamics/celestialBodyConstants.h"
#include "Tudat/Astrodynamics/Gravitation/librationPoint.h"
#include "Tudat/Astrodynamics/Gravitation/jacobiEnergy.h"

#include "createInitialConditions.h"
#include "createLowThrustInitialConditions.h"
#include "applyDifferentialCorrection.h"
#include "applyPredictionCorrection.h"
#include "checkEigenvalues.h"
#include "propagateOrbit.h"
#include "propagateOrbitAugmented.h"
#include "morimotoFirstOrderApproximation.h"
#include "floquetApproximation.h"
//#include "richardsonThirdOrderApproximationMultiple.h"
#include "applyDifferentialCorrectionAugmented.h"

void appendResultsVectorAugmented(const double hamiltonian, const double orbitalPeriod, const Eigen::VectorXd& initialStateVector,
        const Eigen::MatrixXd& stateVectorInclSTM, std::vector< Eigen::VectorXd >& initialConditions )
{
    Eigen::VectorXd tempInitialCondition = Eigen::VectorXd( 112 );

    // Add Jacobi energy and orbital period
    tempInitialCondition( 0 ) = hamiltonian;
    tempInitialCondition( 1 ) = orbitalPeriod;

    // Add initial condition of periodic solution
    for (int i = 0; i <= 9; i++){
        tempInitialCondition( i + 2 ) = initialStateVector( i );
    }

    // Add Monodromy matrix
    tempInitialCondition( 12 ) = stateVectorInclSTM(0, 1);
    tempInitialCondition( 13 ) = stateVectorInclSTM(0, 2);
    tempInitialCondition( 14 ) = stateVectorInclSTM(0, 3);
    tempInitialCondition( 15 ) = stateVectorInclSTM(0, 4);
    tempInitialCondition( 16 ) = stateVectorInclSTM(0, 5);
    tempInitialCondition( 17 ) = stateVectorInclSTM(0, 6);
    tempInitialCondition( 18 ) = stateVectorInclSTM(0, 7);
    tempInitialCondition( 19 ) = stateVectorInclSTM(0, 8);
    tempInitialCondition( 20 ) = stateVectorInclSTM(0, 9);
    tempInitialCondition( 21 ) = stateVectorInclSTM(0, 10);

    tempInitialCondition( 22 ) = stateVectorInclSTM(1, 1);
    tempInitialCondition( 23 ) = stateVectorInclSTM(1, 2);
    tempInitialCondition( 24 ) = stateVectorInclSTM(1, 3);
    tempInitialCondition( 25 ) = stateVectorInclSTM(1, 4);
    tempInitialCondition( 26 ) = stateVectorInclSTM(1, 5);
    tempInitialCondition( 27 ) = stateVectorInclSTM(1, 6);
    tempInitialCondition( 28 ) = stateVectorInclSTM(1, 7);
    tempInitialCondition( 29 ) = stateVectorInclSTM(1, 8);
    tempInitialCondition( 30 ) = stateVectorInclSTM(1, 9);
    tempInitialCondition( 31 ) = stateVectorInclSTM(1, 10);

    tempInitialCondition( 32 ) = stateVectorInclSTM(2, 1);
    tempInitialCondition( 33 ) = stateVectorInclSTM(2, 2);
    tempInitialCondition( 34 ) = stateVectorInclSTM(2, 3);
    tempInitialCondition( 35 ) = stateVectorInclSTM(2, 4);
    tempInitialCondition( 36 ) = stateVectorInclSTM(2, 5);
    tempInitialCondition( 37 ) = stateVectorInclSTM(2, 6);
    tempInitialCondition( 38 ) = stateVectorInclSTM(2, 7);
    tempInitialCondition( 39 ) = stateVectorInclSTM(2, 8);
    tempInitialCondition( 40 ) = stateVectorInclSTM(2, 9);
    tempInitialCondition( 41 ) = stateVectorInclSTM(2, 10);

    tempInitialCondition( 42 ) = stateVectorInclSTM(3, 1);
    tempInitialCondition( 43 ) = stateVectorInclSTM(3, 2);
    tempInitialCondition( 44 ) = stateVectorInclSTM(3, 3);
    tempInitialCondition( 45 ) = stateVectorInclSTM(3, 4);
    tempInitialCondition( 46 ) = stateVectorInclSTM(3, 5);
    tempInitialCondition( 47 ) = stateVectorInclSTM(3, 6);
    tempInitialCondition( 48 ) = stateVectorInclSTM(3, 7);
    tempInitialCondition( 49 ) = stateVectorInclSTM(3, 8);
    tempInitialCondition( 50 ) = stateVectorInclSTM(3, 9);
    tempInitialCondition( 51 ) = stateVectorInclSTM(3, 10);

    tempInitialCondition( 52 ) = stateVectorInclSTM(4, 1);
    tempInitialCondition( 53 ) = stateVectorInclSTM(4, 2);
    tempInitialCondition( 54 ) = stateVectorInclSTM(4, 3);
    tempInitialCondition( 55 ) = stateVectorInclSTM(4, 4);
    tempInitialCondition( 56 ) = stateVectorInclSTM(4, 5);
    tempInitialCondition( 57 ) = stateVectorInclSTM(4, 6);
    tempInitialCondition( 58 ) = stateVectorInclSTM(4, 7);
    tempInitialCondition( 59 ) = stateVectorInclSTM(4, 8);
    tempInitialCondition( 60 ) = stateVectorInclSTM(4, 9);
    tempInitialCondition( 61 ) = stateVectorInclSTM(4, 10);

    tempInitialCondition( 62 ) = stateVectorInclSTM(5, 1);
    tempInitialCondition( 63 ) = stateVectorInclSTM(5, 2);
    tempInitialCondition( 64 ) = stateVectorInclSTM(5, 3);
    tempInitialCondition( 65 ) = stateVectorInclSTM(5, 4);
    tempInitialCondition( 66 ) = stateVectorInclSTM(5, 5);
    tempInitialCondition( 67 ) = stateVectorInclSTM(5, 6);
    tempInitialCondition( 68 ) = stateVectorInclSTM(5, 7);
    tempInitialCondition( 69 ) = stateVectorInclSTM(5, 8);
    tempInitialCondition( 70 ) = stateVectorInclSTM(5, 9);
    tempInitialCondition( 71 ) = stateVectorInclSTM(5, 10);

    tempInitialCondition( 72 ) = stateVectorInclSTM(6, 1);
    tempInitialCondition( 73 ) = stateVectorInclSTM(6, 2);
    tempInitialCondition( 74 ) = stateVectorInclSTM(6, 3);
    tempInitialCondition( 75 ) = stateVectorInclSTM(6, 4);
    tempInitialCondition( 76 ) = stateVectorInclSTM(6, 5);
    tempInitialCondition( 77 ) = stateVectorInclSTM(6, 6);
    tempInitialCondition( 78 ) = stateVectorInclSTM(6, 7);
    tempInitialCondition( 79 ) = stateVectorInclSTM(6, 8);
    tempInitialCondition( 80 ) = stateVectorInclSTM(6, 9);
    tempInitialCondition( 81 ) = stateVectorInclSTM(6, 10);

    tempInitialCondition( 82 ) = stateVectorInclSTM(7, 1);
    tempInitialCondition( 83 ) = stateVectorInclSTM(7, 2);
    tempInitialCondition( 84 ) = stateVectorInclSTM(7, 3);
    tempInitialCondition( 85 ) = stateVectorInclSTM(7, 4);
    tempInitialCondition( 86 ) = stateVectorInclSTM(7, 5);
    tempInitialCondition( 87 ) = stateVectorInclSTM(7, 6);
    tempInitialCondition( 88 ) = stateVectorInclSTM(7, 7);
    tempInitialCondition( 89 ) = stateVectorInclSTM(7, 8);
    tempInitialCondition( 90 ) = stateVectorInclSTM(7, 9);
    tempInitialCondition( 91 ) = stateVectorInclSTM(7, 10);

    tempInitialCondition( 92 ) = stateVectorInclSTM(8, 1);
    tempInitialCondition( 93 ) = stateVectorInclSTM(8, 2);
    tempInitialCondition( 94 ) = stateVectorInclSTM(8, 3);
    tempInitialCondition( 95 ) = stateVectorInclSTM(8, 4);
    tempInitialCondition( 96 ) = stateVectorInclSTM(8, 5);
    tempInitialCondition( 97 ) = stateVectorInclSTM(8, 6);
    tempInitialCondition( 98 ) = stateVectorInclSTM(8, 7);
    tempInitialCondition( 99 ) = stateVectorInclSTM(8, 8);
    tempInitialCondition( 100 ) = stateVectorInclSTM(8, 9);
    tempInitialCondition( 101 ) = stateVectorInclSTM(8, 10);

    tempInitialCondition( 102 ) = stateVectorInclSTM(9, 1);
    tempInitialCondition( 103 ) = stateVectorInclSTM(9, 2);
    tempInitialCondition( 104 ) = stateVectorInclSTM(9, 3);
    tempInitialCondition( 105 ) = stateVectorInclSTM(9, 4);
    tempInitialCondition( 106 ) = stateVectorInclSTM(9, 5);
    tempInitialCondition( 107 ) = stateVectorInclSTM(9, 6);
    tempInitialCondition( 108 ) = stateVectorInclSTM(9, 7);
    tempInitialCondition( 109 ) = stateVectorInclSTM(9, 8);
    tempInitialCondition( 110 ) = stateVectorInclSTM(9, 9);
    tempInitialCondition( 111 ) = stateVectorInclSTM(9, 10);


    initialConditions.push_back(tempInitialCondition);
}

void appendDifferentialCorrectionResultsVectorAugmented(
        const double hamiltonianFullPeriod,  const Eigen::VectorXd& differentialCorrectionResult,
        std::vector< Eigen::VectorXd >& differentialCorrections )
{

    Eigen::VectorXd tempDifferentialCorrection = Eigen::VectorXd( 13 );



    tempDifferentialCorrection( 0 ) = differentialCorrectionResult( 24 );  // numberOfIterations
    tempDifferentialCorrection( 1 ) = hamiltonianFullPeriod;  // jacobiEnergyHalfPeriod
    tempDifferentialCorrection( 2 ) = differentialCorrectionResult( 22 );  // currentTime
    for (int i = 12; i <= 21; i++)
    {
        tempDifferentialCorrection( i - 9 ) = differentialCorrectionResult( i );  // FullPeriodStateVector
    }



    differentialCorrections.push_back(tempDifferentialCorrection);

}

void appendContinuationStatesVectorAugmented(const int orbitNumber, const int numberOfPatchPoints, const double hamiltonianInitialCondition,
                                             const Eigen::VectorXd& differentialCorrectionResult, std::vector< Eigen::VectorXd >& statesContinuation)
{
    Eigen::VectorXd tempStatesContinuation = Eigen::VectorXd( 2 + 11*numberOfPatchPoints  );
    tempStatesContinuation( 0 ) = orbitNumber;  // numberOfIterations
    tempStatesContinuation( 1 ) = hamiltonianInitialCondition;  // Hamiltonian Initial Condition

    for (int i = 25; i < ( 11*numberOfPatchPoints + 25 ); i++)
    {
        tempStatesContinuation( i - 23 ) = differentialCorrectionResult( i );  // InitialStateVectors
    }

    statesContinuation.push_back(tempStatesContinuation);
}


Eigen::VectorXd getEarthMoonInitialGuessParameters ( const int librationPointNr, const std::string& orbitType, const double accelerationMagnitude, const double accelerationAngle, const double accelerationAngle2, const int continuationIndex, const int guessIteration )
{

   //Define the output vector
    Eigen::VectorXd initialGuessParameters(4);
    initialGuessParameters.setZero();

    //Define difference in initial guesses for acceleration and angle
    double deltaAcceleration = 1.0e-10;
    double deltaAngle = 1.0;
    double deltaAngle2 = 1.0;

    // Continuation for Orbital period, set different amplitudes and identical thrust parameters
    if (continuationIndex == 1) {

        if( guessIteration == 0 )
        {
            if (orbitType == "horizontal")
            {
                if (librationPointNr == 1)
                {
                    initialGuessParameters(0) = 1.0e-4;
                }
                else if (librationPointNr == 2)
                {
                    initialGuessParameters(0) = 1.0e-4;
                }
            }
            else if (orbitType == "vertical")
            {
                if (librationPointNr == 1)
                {
                    initialGuessParameters(0) = 1.0e-1;
                }
                else if (librationPointNr == 2)
                {
                    initialGuessParameters(0) = 1.0e-1;
                }
            }
            else if (orbitType == "halo")
            {
                if (librationPointNr == 1)
                {
                    initialGuessParameters(0) = -1.1e-1;
                }
                else if (librationPointNr == 2)
                {
                    initialGuessParameters(0) = 1.5e-1;
                }
            }
        }
        else if( guessIteration == 1 )
        {

            if (orbitType == "horizontal")
            {
                if (librationPointNr == 1)
                {
                    initialGuessParameters(0) = 1.0e-4;
                }
                else if (librationPointNr == 2)
                {
                    initialGuessParameters(0) = 1.0e-3;
                }
            }
            else if (orbitType == "vertical")
            {
                if (librationPointNr == 1)
                {
                    initialGuessParameters(0) = 2.0e-1;
                }
                else if (librationPointNr == 2)
                {
                    initialGuessParameters(0) = 2.0e-1;
                }
            }
            else if (orbitType == "halo")
            {
                if (librationPointNr == 1)
                {
                    initialGuessParameters(0) = -1.2e-1;
                }
                else if (librationPointNr == 2)
                {
                    initialGuessParameters(0) = 1.6e-1;
                }
            }
        }

        initialGuessParameters(1) = accelerationMagnitude;
        initialGuessParameters(2) = accelerationAngle;
        initialGuessParameters(3) = accelerationAngle2;
    }

    // Continuation for acceleration Magnitude, set identical amplitudes and acceleration angles
    else if (continuationIndex == 8) {
            if (orbitType == "horizontal")
            {
                if (librationPointNr == 1)
                {
                    initialGuessParameters(0) = 5.e-5;
                }
                else if (librationPointNr == 2)
                {
                    initialGuessParameters(0) = 1.0e-4;
                }
            }
            else if (orbitType == "vertical")
            {
                if (librationPointNr == 1)
                {
                    initialGuessParameters(0) = 1.0e-1;
                }
                else if (librationPointNr == 2)
                {
                    initialGuessParameters(0) = 1.0e-1;
                }
            }
            else if (orbitType == "halo")
            {
                if (librationPointNr == 1)
                {
                    initialGuessParameters(0) = -1.1e-1;
                }
                else if (librationPointNr == 2)
                {
                    initialGuessParameters(0) = 1.5e-1;
                }
            }

        if ( guessIteration == 0 ) {

            initialGuessParameters(1) = accelerationMagnitude;
        }
        else if (guessIteration == 1) {

            initialGuessParameters(1) = accelerationMagnitude + deltaAcceleration;

        }

           initialGuessParameters(2) = accelerationAngle;
           initialGuessParameters(3) = accelerationAngle2;
        }


    // Continuation for accelerationAngle1, set identical acceleration magnitude, amplitudes and accelerationAngle2
    else if (continuationIndex == 9) {

        if (orbitType == "horizontal")
        {
            if (librationPointNr == 1)
            {
                initialGuessParameters(0) = 1.0e-2;
            }
            else if (librationPointNr == 2)
            {
                initialGuessParameters(0) = 1.0e-4;
            }
        }
        else if (orbitType == "vertical")
        {
            if (librationPointNr == 1)
            {
                initialGuessParameters(0) = 1.0e-1;
            }
            else if (librationPointNr == 2)
            {
                initialGuessParameters(0) = 1.0e-1;
            }
        }
        else if (orbitType == "halo")
        {
            if (librationPointNr == 1)
            {
                initialGuessParameters(0) = -1.1e-1;
            }
            else if (librationPointNr == 2)
            {
                initialGuessParameters(0) = 1.5e-1;
            }
        }

        initialGuessParameters(1) = accelerationMagnitude;

        if ( guessIteration == 0 ) {

            initialGuessParameters(2) = accelerationAngle;

        }
        else if (guessIteration == 1) {

            initialGuessParameters(2) = accelerationAngle + deltaAngle;

        }

        initialGuessParameters(3) = accelerationAngle2;

    }

    // Continuation for accelerationAngle2, set identical acceleration magnitude, amplitudes and accelerationAngle1
    else if (continuationIndex == 10) {

        if (orbitType == "horizontal")
        {
            if (librationPointNr == 1)
            {
                initialGuessParameters(0) = 1.0e-6;
            }
            else if (librationPointNr == 2)
            {
                initialGuessParameters(0) = 1.0e-4;
            }
        }
        else if (orbitType == "vertical")
        {
            if (librationPointNr == 1)
            {
                initialGuessParameters(0) = 1.0e-1;
            }
            else if (librationPointNr == 2)
            {
                initialGuessParameters(0) = 1.0e-1;
            }
        }
        else if (orbitType == "halo")
        {
            if (librationPointNr == 1)
            {
                initialGuessParameters(0) = -1.1e-1;
            }
            else if (librationPointNr == 2)
            {
                initialGuessParameters(0) = 1.5e-1;
            }
        }

        initialGuessParameters(1) = accelerationMagnitude;
        initialGuessParameters(2) = accelerationAngle;

        if ( guessIteration == 0 ) {

            initialGuessParameters(3) = accelerationAngle2;

        }
        else if (guessIteration == 1) {

            initialGuessParameters(3) = accelerationAngle2 + deltaAngle2;

        }

    }

    return initialGuessParameters;
}

Eigen::VectorXd getLowThrustInitialStateVectorGuess( const int librationPointNr, const std::string& orbitType, const double accelerationMagnitude, const double accelerationAngle, const double accelerationAngle2, const double initialMass, const int continuationIndex, const int numberOfPatchPoints, const int guessIteration,
                                            const boost::function< Eigen::VectorXd( const int librationPointNr, const std::string& orbitType, const double accelerationMagnitude, const double accelerationAngle, const double accelerationAngle2, const int continuationIndex, const int guessIteration ) > getInitialGuessParameters )
{
    Eigen::VectorXd lowThrustInitialStateVectorGuess = Eigen::VectorXd::Zero(numberOfPatchPoints*11);
    Eigen::VectorXd initialGuessParameters(4);

    initialGuessParameters = getInitialGuessParameters(librationPointNr, orbitType, accelerationMagnitude, accelerationAngle, accelerationAngle2, continuationIndex, guessIteration );

    //lowThrustInitialStateVectorGuess = richardsonThirdOrderApproximationMultiple( "horizontal", librationPointNr,
    //                                                                              initialGuessParameters(0), initialGuessParameters(1), initialGuessParameters(2), initialGuessParameters(3), initialMass, numberOfPatchPoints );

    //lowThrustInitialStateVectorGuess =  morimotoFirstOrderApproximation(librationPointNr, initialGuessParameters(0), initialGuessParameters(1), initialGuessParameters(2), initialGuessParameters(3), initialMass, numberOfPatchPoints );
    Eigen::ArrayXd amplitudeArray = Eigen::ArrayXd::LinSpaced( 91, 1.0E-5, 1.0E-4);
    std::cout << "Array: \n" << amplitudeArray << std::endl;
    std::cout << "Array1: \n" << amplitudeArray(1) << std::endl;

    for(int k =0;  k < 91; k++)
    {
        lowThrustInitialStateVectorGuess = floquetApproximation( librationPointNr, orbitType, amplitudeArray(k), initialGuessParameters(1), initialGuessParameters(2), initialGuessParameters(3), initialMass, numberOfPatchPoints );
    }

    //Eigen::ArrayXd alphaArray = Eigen::ArrayXd::LinSpaced( 360, 0, 359);
    //std::cout << "Array: \n" << alphaArray << std::endl;
    //std::cout << "Array1: \n" << alphaArray(1) << std::endl;

    //for(int k =0;  k < 360; k++)
    //{
    //    lowThrustInitialStateVectorGuess = floquetApproximation( 1, orbitType, 1.0E-4, 0.01, alphaArray(k), initialGuessParameters(3), initialMass, numberOfPatchPoints );
    //}

    //lowThrustInitialStateVectorGuess = floquetApproximation( 1, orbitType, 1.0E-4, 0.01, 225, initialGuessParameters(3), initialMass, numberOfPatchPoints );



    return lowThrustInitialStateVectorGuess;
}


double computeHamiltonian (const double massParameter, const Eigen::VectorXd stateVector) {

    double Hamiltonian;
    double jacobiEnergy;
    double alpha;
    double beta;
    double innerProduct;

    jacobiEnergy = tudat::gravitation::computeJacobiEnergy( massParameter, stateVector.segment(0,6) );
    alpha = stateVector(7) * tudat::mathematical_constants::PI / 180.0;
    beta = stateVector(8) * tudat::mathematical_constants::PI / 180.0;
    innerProduct = stateVector(0) * stateVector(6) * std::cos( alpha ) * std::cos( beta )
            + stateVector(1) * stateVector(6) * std::sin( alpha ) * std::cos( beta ) + stateVector(2) * stateVector(6) * std::sin( beta );;
    Hamiltonian = -0.5 * jacobiEnergy - innerProduct;
    return Hamiltonian;

}

Eigen::MatrixXd getCorrectedAugmentedInitialState( const Eigen::VectorXd& initialStateGuess, double targetHamiltonian, const int orbitNumber,
                                          const int librationPointNr, const std::string& orbitType, const double massParameter, const int numberOfPatchPoints, const bool hamiltonianConstraint,
                                          std::vector< Eigen::VectorXd >& initialConditions,
                                          std::vector< Eigen::VectorXd >& differentialCorrections,
                                          std::vector< Eigen::VectorXd >& statesContinuation,
                                                   const double maxPositionDeviationFromPeriodicOrbit, double maxVelocityDeviationFromPeriodicOrbit, const double maxPeriodDeviationFromPeriodicOrbit ) {

    //Eigen::MatrixXd stateVectorInclSTM = Eigen::MatrixXd::Zero(8,9);
    Eigen::VectorXd initialStateVector(10);

    // Correct state vector guess
    Eigen::VectorXd differentialCorrectionResult = applyPredictionCorrection(
                librationPointNr, initialStateGuess, targetHamiltonian, massParameter, numberOfPatchPoints, hamiltonianConstraint,
                maxPositionDeviationFromPeriodicOrbit, maxVelocityDeviationFromPeriodicOrbit, maxPeriodDeviationFromPeriodicOrbit );
    //std::cout << "DC result: " << differentialCorrectionResult << std::endl;

    initialStateVector = differentialCorrectionResult.segment( 0, 10 );
    double orbitalPeriod = differentialCorrectionResult( 10 );

    // Propagate the initialStateVector for a full period and write output to file.
    std::map< double, Eigen::VectorXd > stateHistory;
    Eigen::MatrixXd stateVectorInclSTM = propagateOrbitAugmentedToFinalCondition(
                getFullInitialStateAugmented( initialStateVector ), massParameter, orbitalPeriod, 1, stateHistory, 1000, 0.0 ).first;

    writeStateHistoryToFileAugmented( stateHistory, initialStateVector(6), initialStateVector(7), initialStateVector(8), differentialCorrectionResult(11), orbitNumber, librationPointNr, orbitType, 1000, false );

    // Save results
    double hamiltonianFullPeriod = computeHamiltonian( massParameter, stateVectorInclSTM.block(0,0,10,1));

    appendDifferentialCorrectionResultsVectorAugmented( hamiltonianFullPeriod, differentialCorrectionResult, differentialCorrections );

    appendContinuationStatesVectorAugmented( orbitNumber, numberOfPatchPoints, differentialCorrectionResult(11), differentialCorrectionResult, statesContinuation);

    appendResultsVectorAugmented( hamiltonianFullPeriod, orbitalPeriod, initialStateVector, stateVectorInclSTM, initialConditions );

return stateVectorInclSTM;
}

void writeFinalResultsToFilesAugmented( const int librationPointNr, const std::string& orbitType, const int continuationIndex, const double accelerationMagnitude, const double accelerationAngle, const double accelerationAngle2, const double familyHamiltonian, const int numberOfPatchPoints,
                               std::vector< Eigen::VectorXd > initialConditions,
                               std::vector< Eigen::VectorXd > differentialCorrections,
                               std::vector< Eigen::VectorXd > statesContinuation)
{
    std::string directoryPath;
    if (continuationIndex == 1) {
        //directoryPath = "/Users/Sjors/Documents/thesisSoftware/tudatBundle/tudatApplications/thesisProject/data/raw/orbits/augmented/varying_energy/";
        directoryPath = "../data/raw/orbits/augmented/varying_hamiltonian/";

    } else if (continuationIndex == 6) {
        //directoryPath = "/Users/Sjors/Documents/thesisSoftware/tudatBundle/tudatApplications/thesisProject/data/raw/orbits/augmented/varying_acceleration/";
        directoryPath = "../data/raw/orbits/augmented/varying_acceleration/";

    } else if (continuationIndex == 7) {
        //directoryPath = "/Users/Sjors/Documents/thesisSoftware/tudatBundle/tudatApplications/thesisProject/data/raw/orbits/augmented/varying_alpha/";
        directoryPath = "../data/raw/orbits/augmented/varying_alpha/";

    } else {
        //directoryPath = "/Users/Sjors/Documents/thesisSoftware/tudatBundle/tudatApplications/thesisProject/data/raw/orbits/augmented/varying_beta/";
        directoryPath = "../data/raw/orbits/augmented/varying_beta/";

    }

    // Write variables to the right precision via ostringstream

    std::ostringstream ssAccelerationMagnitude;
    ssAccelerationMagnitude << std::fixed <<std::setprecision(13) << accelerationMagnitude;
    std::string stringAccelerationMagnitude = ssAccelerationMagnitude.str();

    std::ostringstream ssAccelerationAngle1;
    ssAccelerationAngle1 << std::fixed <<  std::setprecision(13) << accelerationAngle;
    std::string stringAccelerationAngle1 = ssAccelerationAngle1.str();

    std::ostringstream ssAccelerationAngle2;
    ssAccelerationAngle2 << std::fixed << std::setprecision(13) << accelerationAngle2;
    std::string stringAccelerationAngle2 = ssAccelerationAngle2.str();

    std::ostringstream ssHamiltonian;
    ssHamiltonian << std::fixed << std::setprecision(13) << familyHamiltonian;
    std::string stringHamiltonian = ssHamiltonian.str();

    // remove initial Conditions file that already exist
    if (continuationIndex == 1 ) {

        remove((directoryPath + "L" + std::to_string(librationPointNr) + "_" + orbitType + "_" + stringAccelerationMagnitude + "_" + stringAccelerationAngle1 + "_" + stringAccelerationAngle2 + "_initial_conditions.txt").c_str());

    } else if ( continuationIndex == 6) {

        remove((directoryPath + "L" + std::to_string(librationPointNr) + "_" + orbitType + "_" + stringAccelerationAngle1 + "_" + stringAccelerationAngle2 + "_" + stringHamiltonian + "_initial_conditions.txt").c_str());

    } else if (continuationIndex == 7 ) {

        remove((directoryPath + "L" + std::to_string(librationPointNr) + "_" + orbitType + "_" + stringAccelerationMagnitude + "_" + stringAccelerationAngle2 + "_" + stringHamiltonian + "_initial_conditions.txt").c_str());

    } else {

        remove((directoryPath + "L" + std::to_string(librationPointNr) + "_" + orbitType + "_" + stringAccelerationMagnitude + "_" + stringAccelerationAngle1 + "_" + stringHamiltonian + "_initial_conditions.txt").c_str());
    }

    // Create initial Conditions file that
    std::ofstream textFileInitialConditions;

    if (continuationIndex == 1 ) {

        textFileInitialConditions.open((directoryPath + "L" + std::to_string(librationPointNr) + "_" + orbitType + "_" + stringAccelerationMagnitude + "_" + stringAccelerationAngle1 + "_" + stringAccelerationAngle2 + "_initial_conditions.txt"));

    } else if ( continuationIndex == 6) {

        textFileInitialConditions.open((directoryPath + "L" + std::to_string(librationPointNr) + "_" + orbitType + "_" + stringAccelerationAngle1 + "_" + stringAccelerationAngle2 + "_" + stringHamiltonian + "_initial_conditions.txt"));

    } else if (continuationIndex == 7 ) {

        textFileInitialConditions.open((directoryPath + "L" + std::to_string(librationPointNr) + "_" + orbitType + "_" + stringAccelerationMagnitude + "_" + stringAccelerationAngle2 + "_" + stringHamiltonian + "_initial_conditions.txt"));

    } else {

        textFileInitialConditions.open((directoryPath + "L" + std::to_string(librationPointNr) + "_" + orbitType + "_" + stringAccelerationMagnitude + "_" + stringAccelerationAngle1 + "_" + stringHamiltonian + "_initial_conditions.txt"));
    }

    textFileInitialConditions.precision(std::numeric_limits<double>::digits10);

    // Remove differential Corrections file that already exist
    if (continuationIndex == 1 ) {

        remove((directoryPath + "L" + std::to_string(librationPointNr) + "_" + orbitType + "_" + stringAccelerationMagnitude + "_" + stringAccelerationAngle1 + "_" + stringAccelerationAngle2 + "_differential_correction.txt").c_str());

    } else if ( continuationIndex == 6) {

        remove((directoryPath + "L" + std::to_string(librationPointNr) + "_" + orbitType + "_" + stringAccelerationAngle1 + "_" + stringAccelerationAngle2 + "_" + stringHamiltonian + "_differential_correction.txt").c_str());

    } else if (continuationIndex == 7 ) {

        remove((directoryPath + "L" + std::to_string(librationPointNr) + "_" + orbitType + "_" + stringAccelerationMagnitude + "_" + stringAccelerationAngle2 + "_" + stringHamiltonian + "_differential_correction.txt").c_str());

    } else {

        remove((directoryPath + "L" + std::to_string(librationPointNr) + "_" + orbitType + "_" + stringAccelerationMagnitude + "_" + stringAccelerationAngle1 + "_" + stringHamiltonian + "_differential_correction.txt").c_str());
    }

    std::ofstream textFileDifferentialCorrection;

    if (continuationIndex == 1 ) {

        textFileDifferentialCorrection.open((directoryPath + "L" + std::to_string(librationPointNr) + "_" + orbitType + "_" + stringAccelerationMagnitude + "_" + stringAccelerationAngle1 + "_" + stringAccelerationAngle2 + "_differential_correction.txt"));

    } else if ( continuationIndex == 6) {

        textFileDifferentialCorrection.open((directoryPath + "L" + std::to_string(librationPointNr) + "_" + orbitType + "_" + stringAccelerationAngle1 + "_" + stringAccelerationAngle2 + "_" + stringHamiltonian + "_differential_correctiontxt"));

    } else if (continuationIndex == 7 ) {

        textFileDifferentialCorrection.open((directoryPath + "L" + std::to_string(librationPointNr) + "_" + orbitType + "_" + stringAccelerationMagnitude + "_" + stringAccelerationAngle2 + "_" + stringHamiltonian + "_differential_correction.txt"));

    } else {

        textFileDifferentialCorrection.open((directoryPath + "L" + std::to_string(librationPointNr) + "_" + orbitType + "_" + stringAccelerationMagnitude + "_" + stringAccelerationAngle1 + "_" + stringHamiltonian + "_differential_correction.txt"));
    }

    textFileDifferentialCorrection.precision(std::numeric_limits<double>::digits10);

    // Remove states Continuation file that already exist
    if (continuationIndex == 1 ) {

        remove((directoryPath + "L" + std::to_string(librationPointNr) + "_" + orbitType + "_" + stringAccelerationMagnitude + "_" + stringAccelerationAngle1 + "_" + stringAccelerationAngle2 + "_states_continuation.txt").c_str());

    } else if ( continuationIndex == 6) {

        remove((directoryPath + "L" + std::to_string(librationPointNr) + "_" + orbitType + "_" + stringAccelerationAngle1 + "_" + stringAccelerationAngle2 + "_" + stringHamiltonian + "_states_continuation.txt").c_str());

    } else if (continuationIndex == 7 ) {

        remove((directoryPath + "L" + std::to_string(librationPointNr) + "_" + orbitType + "_" + stringAccelerationMagnitude + "_" + stringAccelerationAngle2 + "_" + stringHamiltonian + "_states_continuation.txt").c_str());

    } else {

        remove((directoryPath + "L" + std::to_string(librationPointNr) + "_" + orbitType + "_" + stringAccelerationMagnitude + "_" + stringAccelerationAngle1 + "_" + stringHamiltonian + "_states_continuation.txt").c_str());
    }

    std::ofstream textFileStatesContinuation;

    if (continuationIndex == 1 ) {

        textFileStatesContinuation.open((directoryPath + "L" + std::to_string(librationPointNr) + "_" + orbitType + "_" + stringAccelerationMagnitude + "_" + stringAccelerationAngle1 + "_" + stringAccelerationAngle2 + "_states_continuation.txt"));

    } else if ( continuationIndex == 6) {

        textFileStatesContinuation.open((directoryPath + "L" + std::to_string(librationPointNr) + "_" + orbitType + "_" + stringAccelerationAngle1 + "_" + stringAccelerationAngle2 + "_" + stringHamiltonian + "_states_continuation.txt"));

    } else if (continuationIndex == 7 ) {

        textFileStatesContinuation.open((directoryPath + "L" + std::to_string(librationPointNr) + "_" + orbitType + "_" + stringAccelerationMagnitude + "_" + stringAccelerationAngle2 + "_" + stringHamiltonian + "_states_continuation.txt"));

    } else {

        textFileStatesContinuation.open((directoryPath + "L" + std::to_string(librationPointNr) + "_" + orbitType + "_" + stringAccelerationMagnitude + "_" + stringAccelerationAngle1 + "_" + stringHamiltonian + "_states_continuation.txt"));
    }

    textFileStatesContinuation.precision(std::numeric_limits<double>::digits10);

    // Write initial conditions to file
    for (unsigned int i=0; i<initialConditions.size(); i++) {

        textFileInitialConditions << std::left << std::scientific                                          << std::setw(25)
                                  << initialConditions[i][0]  << std::setw(25) << initialConditions[i][1]  << std::setw(25)
                                  << initialConditions[i][2]  << std::setw(25) << initialConditions[i][3]  << std::setw(25)
                                  << initialConditions[i][4]  << std::setw(25) << initialConditions[i][5]  << std::setw(25)
                                  << initialConditions[i][6]  << std::setw(25) << initialConditions[i][7]  << std::setw(25)
                                  << initialConditions[i][8]  << std::setw(25) << initialConditions[i][9]  << std::setw(25)
                                  << initialConditions[i][10] << std::setw(25) << initialConditions[i][11] << std::setw(25)
                                  << initialConditions[i][12] << std::setw(25) << initialConditions[i][13] << std::setw(25)
                                  << initialConditions[i][14] << std::setw(25) << initialConditions[i][15] << std::setw(25)
                                  << initialConditions[i][16] << std::setw(25) << initialConditions[i][17] << std::setw(25)
                                  << initialConditions[i][18] << std::setw(25) << initialConditions[i][19] << std::setw(25)
                                  << initialConditions[i][20] << std::setw(25) << initialConditions[i][21] << std::setw(25)
                                  << initialConditions[i][22] << std::setw(25) << initialConditions[i][23] << std::setw(25)
                                  << initialConditions[i][24] << std::setw(25) << initialConditions[i][25] << std::setw(25)
                                  << initialConditions[i][26] << std::setw(25) << initialConditions[i][27] << std::setw(25)
                                  << initialConditions[i][28] << std::setw(25) << initialConditions[i][29] << std::setw(25)
                                  << initialConditions[i][30] << std::setw(25) << initialConditions[i][31] << std::setw(25)
                                  << initialConditions[i][32] << std::setw(25) << initialConditions[i][33] << std::setw(25)
                                  << initialConditions[i][34] << std::setw(25) << initialConditions[i][35] << std::setw(25)
                                  << initialConditions[i][36] << std::setw(25) << initialConditions[i][37] << std::setw(25)
                                  << initialConditions[i][38] << std::setw(25) << initialConditions[i][39] << std::setw(25)
                                  << initialConditions[i][40] << std::setw(25) << initialConditions[i][41] << std::setw(25)
                                  << initialConditions[i][44] << std::setw(25) << initialConditions[i][45] << std::setw(25)
                                  << initialConditions[i][46] << std::setw(25) << initialConditions[i][47] << std::setw(25)
                                  << initialConditions[i][48] << std::setw(25) << initialConditions[i][49] << std::setw(25)
                                  << initialConditions[i][50] << std::setw(25) << initialConditions[i][51] << std::setw(25)
                                  << initialConditions[i][52] << std::setw(25) << initialConditions[i][53] << std::setw(25)
                                  << initialConditions[i][54] << std::setw(25) << initialConditions[i][55] << std::setw(25)
                                  << initialConditions[i][56] << std::setw(25) << initialConditions[i][57] << std::setw(25)
                                  << initialConditions[i][58] << std::setw(25) << initialConditions[i][59] << std::setw(25)
                                  << initialConditions[i][60] << std::setw(25) << initialConditions[i][61] << std::setw(25)
                                  << initialConditions[i][62] << std::setw(25) << initialConditions[i][63] << std::setw(25)
                                  << initialConditions[i][64] << std::setw(25) << initialConditions[i][65] << std::setw(25)
                                  << initialConditions[i][66] << std::setw(25) << initialConditions[i][67] << std::setw(25)
                                  << initialConditions[i][68] << std::setw(25) << initialConditions[i][69] << std::setw(25)
                                  << initialConditions[i][70] << std::setw(25) << initialConditions[i][71] << std::setw(25)
                                  << initialConditions[i][72] << std::setw(25) << initialConditions[i][73] << std::setw(25)
                                  << initialConditions[i][74] << std::setw(25) << initialConditions[i][75] << std::setw(25)
                                  << initialConditions[i][76] << std::setw(25) << initialConditions[i][77] << std::setw(25)
                                  << initialConditions[i][78] << std::setw(25) << initialConditions[i][79] << std::setw(25)
                                  << initialConditions[i][80] << std::setw(25) << initialConditions[i][81] << std::setw(25)
                                  << initialConditions[i][82] << std::setw(25) << initialConditions[i][83] << std::setw(25)
                                  << initialConditions[i][84] << std::setw(25) << initialConditions[i][85] << std::setw(25)
                                  << initialConditions[i][86] << std::setw(25) << initialConditions[i][87] << std::setw(25)
                                  << initialConditions[i][88] << std::setw(25) << initialConditions[i][89] << std::setw(25)
                                  << initialConditions[i][90] << std::setw(25) << initialConditions[i][91] << std::setw(25)
                                  << initialConditions[i][92] << std::setw(25) << initialConditions[i][93] << std::setw(25)
                                  << initialConditions[i][94] << std::setw(25) << initialConditions[i][95] << std::setw(25)
                                  << initialConditions[i][96] << std::setw(25) << initialConditions[i][97] << std::setw(25)
                                  << initialConditions[i][98] << std::setw(25) << initialConditions[i][99] << std::setw(25)
                                  << initialConditions[i][100] << std::setw(25) << initialConditions[i][101] << std::setw(25)
                                  << initialConditions[i][102] << std::setw(25) << initialConditions[i][103] << std::setw(25)
                                  << initialConditions[i][104] << std::setw(25) << initialConditions[i][105] << std::setw(25)
                                  << initialConditions[i][106] << std::setw(25) << initialConditions[i][107] << std::setw(25)
                                  << initialConditions[i][108] << std::setw(25) << initialConditions[i][109] << std::setw(25)
                                  << initialConditions[i][110] << std::setw(25) << initialConditions[i][111] << std::endl;


        textFileDifferentialCorrection << std::left << std::scientific   << std::setw(25)
                                       << differentialCorrections[i][0]  << std::setw(25) << differentialCorrections[i][1 ]  << std::setw(25)
                                       << differentialCorrections[i][2]  << std::setw(25) << differentialCorrections[i][3]  << std::setw(25)
                                       << differentialCorrections[i][4]  << std::setw(25) << differentialCorrections[i][5]  << std::setw(25)
                                       << differentialCorrections[i][6]  << std::setw(25) << differentialCorrections[i][7]  << std::setw(25)
                                       << differentialCorrections[i][8]  << std::setw(25) << differentialCorrections[i][9]  << std::setw(25)
                                       << differentialCorrections[i][10]  << std::endl;

        for (int test = 0; test < (2+11*numberOfPatchPoints); test++)
                {

                    textFileStatesContinuation << std::left << std::scientific   << std::setw(25)
                                                   << statesContinuation[i][test];
                }

                textFileStatesContinuation << std::left << std::scientific   << std::endl;
    }
}

double getDefaultArcLengthAugmented(
        const double  distanceIncrement,
        const Eigen::VectorXd& currentState, const int continuationIndex )
{

   return distanceIncrement / ( currentState.segment( 0, 1 ).norm( ) );
}

bool checkTerminationAugmented( const std::vector< Eigen::VectorXd >& differentialCorrections,
                       const Eigen::MatrixXd& stateVectorInclSTM, const std::string orbitType, const int librationPointNr,
                       const double maxEigenvalueDeviation )
{
    // Check termination conditions
    bool continueNumericalContinuation = true;
    if ( differentialCorrections.at( differentialCorrections.size( ) - 1 ).segment(0, 10) == Eigen::VectorXd::Zero(10) )
    {
        continueNumericalContinuation = false;
        std::cout << "\n\nNUMERICAL CONTINUATION STOPPED DUE TO EXCEEDING MAXIMUM NUMBER OF ITERATIONS\n\n" << std::endl;
    }
    else
    {

        // Check eigenvalue condition (at least one pair equalling a real one)
        // Exception for the horizontal Lyapunov family in Earth-Moon L2: eigenvalue may be of module one instead of a real one to compute a more extensive family
        continueNumericalContinuation = false;
        if ( ( librationPointNr == 2 ) && ( orbitType == "horizontal" ) )
        {
            continueNumericalContinuation = checkEigenvalues( stateVectorInclSTM, maxEigenvalueDeviation, true );
        }
        else
        {
            continueNumericalContinuation = checkEigenvalues( stateVectorInclSTM, maxEigenvalueDeviation, false );
        }
    }
    return continueNumericalContinuation;
}

void createLowThrustInitialConditions( const int librationPointNr, const std::string& orbitType, const int continuationIndex, const double accelerationMagnitude, const double accelerationAngle,
                                       const double accelerationAngle2, const double initialMass, const double familyHamiltonian,
                              const double massParameter, const int numberOfPatchPoints, const double maxPositionDeviationFromPeriodicOrbit, const double maxVelocityDeviationFromPeriodicOrbit, const double maxPeriodDeviationFromPeriodicOrbit, const double maxEigenvalueDeviation,
                              const boost::function< double( const Eigen::VectorXd&, const int ) > pseudoArcLengthFunctionAugmented ) {

    std::cout << "\nCreate initial conditions:\n" << std::endl;

    // Set output maximum precision
    std::cout.precision(std::numeric_limits<double>::digits10);

    // Initialize state vectors and orbital periods
    Eigen::VectorXd initialStateVector = Eigen::VectorXd::Zero( 11*numberOfPatchPoints );
    Eigen::MatrixXd stateVectorInclSTM = Eigen::MatrixXd::Zero( 10, 11 );

    Eigen::VectorXd linearApproximationResultIteration1 = Eigen::VectorXd::Zero(11*numberOfPatchPoints);
    Eigen::VectorXd linearApproximationResultIteration2 = Eigen::VectorXd::Zero(11*numberOfPatchPoints);



    std::vector< Eigen::VectorXd > initialConditions;
    std::vector< Eigen::VectorXd > differentialCorrections;
    std::vector< Eigen::VectorXd > statesContinuation;

    // Obtain ballistic initial guesses and refine them
    linearApproximationResultIteration1 = getLowThrustInitialStateVectorGuess(librationPointNr, orbitType, accelerationMagnitude, accelerationAngle, accelerationAngle2, initialMass, continuationIndex, numberOfPatchPoints, 0);
//    linearApproximationResultIteration2 = getLowThrustInitialStateVectorGuess(librationPointNr, orbitType, accelerationMagnitude, accelerationAngle, accelerationAngle2, initialMass, continuationIndex, numberOfPatchPoints, 1);


//    stateVectorInclSTM =  getCorrectedAugmentedInitialState(
//                linearApproximationResultIteration1, computeHamiltonian( massParameter, linearApproximationResultIteration1.segment(0,10)), 0,
//               librationPointNr, orbitType, massParameter, numberOfPatchPoints, false, initialConditions, differentialCorrections, statesContinuation,
//                maxPositionDeviationFromPeriodicOrbit, maxVelocityDeviationFromPeriodicOrbit );

//    if ( continuationIndex == 1 ) {

//        stateVectorInclSTM =  getCorrectedAugmentedInitialState(
//                    linearApproximationResultIteration2, computeHamiltonian( massParameter, linearApproximationResultIteration1.segment(0,10)), 1,
//                   librationPointNr, orbitType, massParameter, numberOfPatchPoints, false, initialConditions, differentialCorrections, statesContinuation,
//                    maxPositionDeviationFromPeriodicOrbit, maxVelocityDeviationFromPeriodicOrbit );
//    }

//    // Set exit parameters of continuation procedure
//    int maximumNumberOfInitialConditions = 2000;
//    int numberOfInitialConditions;
//    if ( continuationIndex == 1 ) {

//        numberOfInitialConditions = 2;
//    }

// //    } else {

// //        numberOfInitialConditions = 1;
// //    }

//    // Generate periodic orbits until termination
// //    double orbitalPeriod  = 0.0, periodIncrement = 0.0, pseudoArcLengthCorrection = 0.0;
// //    int orderOfMagnitude = 5, minimumIncrementOrderOfMagnitude = 10;
//    double pseudoArcLengthCorrection = 0.0;
//    bool continueNumericalContinuation = true;
//    Eigen::VectorXd stateIncrement(11*numberOfPatchPoints+1);
//    stateIncrement.setZero();
//    double targetHamiltonian;

//    while( ( numberOfInitialConditions < maximumNumberOfInitialConditions ) && continueNumericalContinuation)
//    {

//        std::cout << "========== Numerical continuation Status Update ========== "<< std::endl
//                << "Creating initial guess number "  << numberOfInitialConditions + 1 << std::endl
//                << "Continuating along continuation index "  << continuationIndex << std::endl
//                << "============================================================ " << std::endl;

// //        double incrementContinuationParameter =  pow(10,(static_cast<float>(-orderOfMagnitude)));


//            //std::cout << "============" << std::endl
//            //          << "X^{n}: \n" << statesContinuation[ statesContinuation.size( ) - 2 ].segment( 2, 11*numberOfPatchPoints ) << std::endl
//            //          << "X^{n+1}: \n" << statesContinuation[ statesContinuation.size( ) - 1 ].segment( 2, 11*numberOfPatchPoints ) << std::endl;
//            //std::cout << "Hamiltonian n: "  << statesContinuation[ statesContinuation.size( ) - 2 ](1) << std::endl;
//            //std::cout << "Hamiltonian n+1: "  << statesContinuation[ statesContinuation.size( ) - 1 ](1) << std::endl;


//            // Determine increments to state and time
//            stateIncrement.segment(1,11*numberOfPatchPoints) = statesContinuation[ statesContinuation.size( ) - 1 ].segment( 2, 11*numberOfPatchPoints ) -
//                    statesContinuation[ statesContinuation.size( ) - 2 ].segment( 2, 11*numberOfPatchPoints );
//            stateIncrement(0) = statesContinuation[ statesContinuation.size( ) - 1 ]( 1 ) -
//                    statesContinuation[ statesContinuation.size( ) - 2 ]( 1 );

//            //std::cout << "stateIncrement: \n" << stateIncrement << std::endl;

//            pseudoArcLengthCorrection =
//                    pseudoArcLengthFunctionAugmented( stateIncrement, continuationIndex );

//            //std::cout << "pseudoArcLengthCorrection: " << pseudoArcLengthCorrection << std::endl;

//            // Apply numerical continuation
//            initialStateVector = statesContinuation[ statesContinuation.size( ) - 1 ].segment( 2, 11*numberOfPatchPoints ) +
//                    stateIncrement.segment(1,11*numberOfPatchPoints) * pseudoArcLengthCorrection;

//            targetHamiltonian = statesContinuation[ statesContinuation.size( ) - 1 ](1) + stateIncrement(0) * pseudoArcLengthCorrection;





//                stateVectorInclSTM =  getCorrectedAugmentedInitialState( initialStateVector, targetHamiltonian, numberOfInitialConditions,
//                                                                         librationPointNr, orbitType, massParameter, numberOfPatchPoints, false,
//                                                                         initialConditions, differentialCorrections, statesContinuation,
//                                                                         maxPositionDeviationFromPeriodicOrbit, maxVelocityDeviationFromPeriodicOrbit, maxPeriodDeviationFromPeriodicOrbit);

//                numberOfInitialConditions += 1;

//                continueNumericalContinuation = checkTerminationAugmented(differentialCorrections, stateVectorInclSTM, orbitType, librationPointNr, maxEigenvalueDeviation );

// //                if ( continuationIndex != 1 && continueNumericalContinuation == false && orderOfMagnitude > minimumIncrementOrderOfMagnitude )
// //                  {
// //                            orderOfMagnitude = orderOfMagnitude + 1;
// //                            continueNumericalContinuation = true;
// //                        }

// //        // Copy previous solution as current solution
// //        //initialStateVector = initialConditions[ initialConditions.size( ) - 1 ].segment( 2, 10 );
// //        //orbitalPeriod = initialConditions[ initialConditions.size( ) - 1 ](1);

// //        // Provide increment to the continuation parameter
// //        //if (continuationIndex == 1) {
// //        //   orbitalPeriod = orbitalPeriod + incrementContinuationParameter ;
// //        //} else {

// //        //   initialStateVector( continuationIndex - 2 ) = initialStateVector( continuationIndex - 2 ) + incrementContinuationParameter;
// //        //
// //        //}




// //        // impleement pseudo arc length for potential further guesses

// //        // Adapt first and second intiial guesses from Earth Moon Parameters

// //        // T orbit: switch amplitudes so it seeds to the left!

// //        // acceleration largest amplitude and one initial guess, start at 10^-4

// //        // alpha largest amplitude and start at alpha


// //        std::cout << "============" << std::endl
// //                  << "X^{n}: \n" << initialConditions[ initialConditions.size( ) - 2 ].segment( 2, 10 ) << std::endl
// //                  << "X^{n+1}: \n" << initialConditions[ initialConditions.size( ) - 1 ].segment( 2, 10 ) << std::endl;




//    }


//    writeFinalResultsToFilesAugmented( librationPointNr, orbitType, continuationIndex, accelerationMagnitude, accelerationAngle, accelerationAngle2, familyHamiltonian, numberOfPatchPoints, initialConditions, differentialCorrections, statesContinuation );

}
