#include <fstream>
#include <iomanip>
#include <iostream>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>
#include <Eigen/Dense>
#include <boost/function.hpp>
#include <random>

#include "Tudat/Astrodynamics/BasicAstrodynamics/celestialBodyConstants.h"
#include "Tudat/Astrodynamics/Gravitation/librationPoint.h"
#include "Tudat/Astrodynamics/Gravitation/jacobiEnergy.h"

#include "createInitialConditions.h"
#include "createLowThrustInitialConditions.h"
#include "applyDifferentialCorrection.h"
#include "applyCollocation.h"
#include "applyPredictionCorrection.h"
#include "applyMassRefinement.h"
#include "applyTwoLevelTargeterLowThrust.h"
#include "applyPredictionCorrection.h"
#include "checkEigenvalues.h"
#include "propagateOrbit.h"
#include "propagateOrbitAugmented.h"
#include "floquetApproximation.h"
#include "createEquilibriumLocations.h"
#include "stateDerivativeModelAugmented.h"
#include "computeCollocationCorrection.h"
#include "applyMeshRefinement.h"
#include "interpolatePolynomials.h"
#include "refineOrbitHamiltonian.h"
#include "initialiseContinuationFromTextFile.h"

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
        std::vector< Eigen::VectorXd >& differentialCorrections, const Eigen::VectorXd deviationsNorms )
{

    Eigen::VectorXd tempDifferentialCorrection = Eigen::VectorXd( 21 );


    tempDifferentialCorrection( 0 ) = differentialCorrectionResult( 24 );  // numberOfIterations
    tempDifferentialCorrection( 1 ) = differentialCorrectionResult( 25 );  // maximumErrorPerSegment
    tempDifferentialCorrection( 2 ) = differentialCorrectionResult( 26 );  // DeltaErrorDistribution
    tempDifferentialCorrection( 3 ) = hamiltonianFullPeriod;  // HamiltonianFullPeriod
    tempDifferentialCorrection( 4 ) = differentialCorrectionResult( 22 );  // currentTime
    tempDifferentialCorrection.segment(5,5) = deviationsNorms;
    tempDifferentialCorrection.segment(10,10) = differentialCorrectionResult.segment(12,10);


    differentialCorrections.push_back(tempDifferentialCorrection);

}

void appendContinuationStatesVectorAugmented(const int orbitNumber, const int numberOfPatchPoints, const double hamiltonianInitialCondition, const double orbitalPeriod,
                                             const Eigen::VectorXd& differentialCorrectionResult, std::vector< Eigen::VectorXd >& statesContinuation)
{
    Eigen::VectorXd tempStatesContinuation = Eigen::VectorXd( 3 + 11*numberOfPatchPoints  );
    tempStatesContinuation( 0 ) = orbitNumber;  // numberOfIterations
    tempStatesContinuation( 1 ) = hamiltonianInitialCondition;  // Hamiltonian Initial Condition
    tempStatesContinuation( 2 ) = orbitalPeriod;

    for (int i = 0; i < ( 11*numberOfPatchPoints ); i++)
    {
        tempStatesContinuation(i + 3) = differentialCorrectionResult(i);
    }

    statesContinuation.push_back(tempStatesContinuation);
}


Eigen::VectorXd getEarthMoonInitialGuessParameters ( const int librationPointNr, const std::string& orbitType, const double accelerationMagnitude, const double accelerationAngle, const double accelerationAngle2, const int continuationIndex, const int guessIteration )
{

   //Define the output vector
    Eigen::VectorXd initialGuessParameters(4);
    initialGuessParameters.setZero();

    //Define difference in initial guesses for acceleration and angle
    double deltaAcceleration = 1.0e-5;
    double deltaAngle = 1.0E-1;
    double deltaAngle2 = 1.0E-1;

    // Continuation for Orbital period, set different amplitudes and identical thrust parameters
    if (continuationIndex == 1) {

        if( guessIteration == 0 )
        {
            if (orbitType == "horizontal")
            {
                if (librationPointNr == 1)
                {
                    initialGuessParameters(0) = 1.0e-5;
                }
                else if (librationPointNr == 2)
                {
                    initialGuessParameters(0) = 1.0e-5;
                }
                else if (librationPointNr == 3)
                {
                    initialGuessParameters(0) = 1.0e-5;
                }
                else if (librationPointNr == 4)
                {
                    initialGuessParameters(0) = 1.0e-5;
                } else
                {
                    initialGuessParameters(0) = 1.0e-5;
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
                    initialGuessParameters(0) = 1.0e-4;
                }
                else if (librationPointNr == 3)
                {
                    initialGuessParameters(0) = 1.0e-4;
                }
                else if (librationPointNr == 4)
                {
                    initialGuessParameters(0) = 1.0e-4;
                }
                else if (librationPointNr == 5)
                {
                    initialGuessParameters(0) = 1.0e-4;
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
    else if (continuationIndex == 6) {
            if (orbitType == "horizontal")
            {
                if (librationPointNr == 1)
                {
                    initialGuessParameters(0) = 9.0e-3;
                }
                else if (librationPointNr == 2)
                {
                    initialGuessParameters(0) = 1.0e-5;
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
    else if (continuationIndex == 7) {

        if (orbitType == "horizontal")
        {
            if (librationPointNr == 1)
            {
                initialGuessParameters(0) = 1.0e-5;
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
    else if (continuationIndex == 8) {

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

Eigen::VectorXd getLowThrustInitialStateVectorGuess( const int librationPointNr, const double ySign, const std::string& orbitType, const double accelerationMagnitude, const double accelerationAngle, const double accelerationAngle2, const double initialMass, const int continuationIndex, const int numberOfPatchPoints, const int guessIteration,
                                            const boost::function< Eigen::VectorXd( const int librationPointNr, const std::string& orbitType, const double accelerationMagnitude, const double accelerationAngle, const double accelerationAngle2, const int continuationIndex, const int guessIteration ) > getInitialGuessParameters )
{
    Eigen::VectorXd lowThrustInitialStateVectorGuess = Eigen::VectorXd::Zero(numberOfPatchPoints*11);
    Eigen::VectorXd initialGuessParameters(4);


    initialGuessParameters = getInitialGuessParameters(librationPointNr, orbitType, accelerationMagnitude, accelerationAngle, accelerationAngle2, continuationIndex, guessIteration );

    lowThrustInitialStateVectorGuess = floquetApproximation( librationPointNr, ySign, orbitType, initialGuessParameters(0), initialGuessParameters(1), initialGuessParameters(2), initialGuessParameters(3), initialMass, numberOfPatchPoints );
   //lowThrustInitialStateVectorGuess = floquetApproximation( librationPointNr, orbitType, 1.0E-4, initialGuessParameters(1), initialGuessParameters(2), initialGuessParameters(3), initialMass, numberOfPatchPoints );

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

Eigen::MatrixXd getCollocatedAugmentedInitialState( const Eigen::MatrixXd& initialOddPoints, const int orbitNumber,
                                          const int librationPointNr, const std::string& orbitType, const int continuationIndex, const Eigen::VectorXd previousDesignVector, const double massParameter, const int numberOfPatchPoints, int& numberOfCollocationPoints,
                                          std::vector< Eigen::VectorXd >& initialConditions,
                                          std::vector< Eigen::VectorXd >& differentialCorrections,
                                          std::vector< Eigen::VectorXd >& statesContinuation,
                                                   const double maxPositionDeviationFromPeriodicOrbit, double maxVelocityDeviationFromPeriodicOrbit, const double maxPeriodDeviationFromPeriodicOrbit )
{
    Eigen::VectorXd initialCollocationGuess;
    Eigen::VectorXd initialStateVector(10);


    // Apply collocation
    Eigen::VectorXd deviationNorms(5);
    Eigen::VectorXd collocatedGuess;
    Eigen::VectorXd collocatedNodes;
    Eigen::VectorXd collocatedDefects;

    Eigen::VectorXd collocationResult = applyCollocation(initialOddPoints, massParameter, numberOfCollocationPoints, collocatedGuess, collocatedNodes, deviationNorms, collocatedDefects, continuationIndex, previousDesignVector,
                                                         maxPositionDeviationFromPeriodicOrbit, maxVelocityDeviationFromPeriodicOrbit, maxPeriodDeviationFromPeriodicOrbit);

    int numberOfOddPoints = (numberOfCollocationPoints-1)*3 + 1;
    initialStateVector = collocationResult.segment( 0, 10 );
    double orbitalPeriod = collocationResult( 10 );
    double hamiltonianFullPeriodColloc = collocationResult( 23 );

    // Propagate the initialStateVector for a full period and write output to file.

    std::map< double, Eigen::VectorXd > stateHistory;
    Eigen::MatrixXd stateVectorInclSTM = propagateOrbitAugmentedToFinalCondition(
                getFullInitialStateAugmented( initialStateVector ), massParameter, orbitalPeriod, 1, stateHistory, 1000, 0.0 ).first;

    // compute Deviations resulting from full period and MS!
    Eigen::VectorXd fullPeriodDeviations = initialStateVector.block(0,0,10,1) - stateVectorInclSTM.block(0,0,10,1);

    Eigen::VectorXd defectVectorMS(11*(numberOfCollocationPoints-1));
    std::map< double, Eigen::VectorXd > stateHistoryMS;
    Eigen::MatrixXd propagatedStatesMS(10*(numberOfCollocationPoints-1),11);


    //defectVectorMS.setZero(); stateHistoryMS.clear(); propagatedStatesMS.setZero();
    //computeOrbitDeviations( collocatedNodes, numberOfCollocationPoints, propagatedStatesMS, defectVectorMS, stateHistoryMS, massParameter);
    //const int magnitudeNoiseOffset = 0;
    //const double amplitude = 9.0E-3;
    //Eigen::VectorXd collcationSegmentErrors = computeSegmentErrors( collocatedGuess, initialOddPoints.block(6,0,4,1), numberOfCollocationPoints);
    //writeTrajectoryErrorDataToFile(numberOfCollocationPoints, fullPeriodDeviations, defectVectorMS, collocatedDefects, collcationSegmentErrors, magnitudeNoiseOffset, amplitude );

    writeStateHistoryToFileAugmented( stateHistory, initialStateVector(6), initialStateVector(7), initialStateVector(8), collocationResult(11), orbitNumber, librationPointNr, orbitType, 1000, false );

    // Save results
    double hamiltonianFullPeriod = computeHamiltonian( massParameter, stateVectorInclSTM.block(0,0,10,1));

    Eigen::VectorXd defectVector(11*numberOfCollocationPoints);
    Eigen::MatrixXd propagatedStates(10*(numberOfCollocationPoints-1),11);
    std::map< double, Eigen::VectorXd > stateHistoryTemp;

    appendDifferentialCorrectionResultsVectorAugmented( hamiltonianFullPeriod, collocationResult, differentialCorrections, deviationNorms );

    Eigen::VectorXd collocationResultWithStates(25+11*numberOfCollocationPoints);
    collocationResultWithStates.segment(0,25) = collocationResult;
    collocationResultWithStates.segment(25,11*numberOfCollocationPoints) = collocatedNodes;

    appendContinuationStatesVectorAugmented( orbitNumber, numberOfOddPoints, collocationResult(11), orbitalPeriod, collocatedGuess, statesContinuation);

    appendResultsVectorAugmented( hamiltonianFullPeriod, orbitalPeriod, initialStateVector, stateVectorInclSTM, initialConditions );


    return stateVectorInclSTM;

}


Eigen::MatrixXd getCorrectedAugmentedInitialState( const Eigen::VectorXd& initialStateGuess, double targetHamiltonian, const int orbitNumber,
                                          const int librationPointNr, const std::string& orbitType, const double massParameter, const int numberOfPatchPoints, const int numberOfCollocationPoints, const bool hamiltonianConstraint,
                                          std::vector< Eigen::VectorXd >& initialConditions,
                                          std::vector< Eigen::VectorXd >& differentialCorrections,
                                          std::vector< Eigen::VectorXd >& statesContinuation,
                                                   const double maxPositionDeviationFromPeriodicOrbit, double maxVelocityDeviationFromPeriodicOrbit, const double maxPeriodDeviationFromPeriodicOrbit ) {

    //Eigen::MatrixXd stateVectorInclSTM = Eigen::MatrixXd::Zero(8,9);
    Eigen::VectorXd initialStateVector(10);
    int numberOfOddPoints = (numberOfCollocationPoints-1)*3 + 1;

    // Correct state vector guess
    Eigen::VectorXd differentialCorrectionResult = applyPredictionCorrection(
                librationPointNr, initialStateGuess, massParameter,  numberOfPatchPoints,
                maxPositionDeviationFromPeriodicOrbit, maxVelocityDeviationFromPeriodicOrbit, maxPeriodDeviationFromPeriodicOrbit );

    initialStateVector = differentialCorrectionResult.segment( 0, 10 );
    double orbitalPeriod = differentialCorrectionResult( 10 );
    double hamiltonianFullPeriodDiffCorr = differentialCorrectionResult( 23 );


    // Propagate the initialStateVector for a full period and write output to file.
    std::map< double, Eigen::VectorXd > stateHistory;
    Eigen::MatrixXd stateVectorInclSTM = propagateOrbitAugmentedToFinalCondition(
                getFullInitialStateAugmented( initialStateVector ), massParameter, orbitalPeriod, 1, stateHistory, 1000, 0.0 ).first;


    writeStateHistoryToFileAugmented( stateHistory, initialStateVector(6), initialStateVector(7), initialStateVector(8), differentialCorrectionResult(11), orbitNumber, librationPointNr, orbitType, 1000, false );



    // Save results
    double hamiltonianFullPeriod = computeHamiltonian( massParameter, stateVectorInclSTM.block(0,0,10,1));

    Eigen::VectorXd defectVector(11*numberOfPatchPoints);
    Eigen::MatrixXd propagatedStates(10*(numberOfPatchPoints-1),11);
    std::map< double, Eigen::VectorXd > stateHistoryTemp;

    computeOrbitDeviations(differentialCorrectionResult.segment(25,11*numberOfPatchPoints), numberOfPatchPoints, propagatedStates, defectVector, stateHistoryTemp, massParameter);
    Eigen::VectorXd deviationNorms = computeDeviationNorms(defectVector,numberOfPatchPoints);

    Eigen::VectorXd DCResultsVector = Eigen::VectorXd(27); DCResultsVector.setZero();
    DCResultsVector.segment(0,25) = differentialCorrectionResult.segment(0,25);

    appendDifferentialCorrectionResultsVectorAugmented( hamiltonianFullPeriodDiffCorr, DCResultsVector, differentialCorrections, deviationNorms );
    Eigen::VectorXd statesConvergedGuess = differentialCorrectionResult.segment(25,11*numberOfPatchPoints);

    // store the multiple shooting converged guesses as converged colllocated formats in the
    Eigen::VectorXd redistributedNodes = redstributeNodesOverTrajectory( statesConvergedGuess, numberOfPatchPoints, numberOfCollocationPoints, massParameter);
    Eigen::MatrixXd oddNodesMatrix((11*(numberOfCollocationPoints-1)), 4 );
    computeOddPoints(redistributedNodes, oddNodesMatrix, numberOfCollocationPoints, massParameter, true);
    Eigen::VectorXd convergedGuessCollocationFormat = rewriteOddPointsToVector(oddNodesMatrix, numberOfCollocationPoints);

    appendContinuationStatesVectorAugmented( orbitNumber, numberOfOddPoints, differentialCorrectionResult(11), orbitalPeriod, convergedGuessCollocationFormat, statesContinuation);



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
    ssAccelerationMagnitude << std::fixed <<std::setprecision(11) << accelerationMagnitude;
    std::string stringAccelerationMagnitude = ssAccelerationMagnitude.str();

    std::ostringstream ssAccelerationAngle1;
    ssAccelerationAngle1 << std::fixed <<  std::setprecision(11) << accelerationAngle;
    std::string stringAccelerationAngle1 = ssAccelerationAngle1.str();

    std::ostringstream ssAccelerationAngle2;
    ssAccelerationAngle2 << std::fixed << std::setprecision(11) << accelerationAngle2;
    std::string stringAccelerationAngle2 = ssAccelerationAngle2.str();

    std::ostringstream ssHamiltonian;
    ssHamiltonian << std::fixed << std::setprecision(11) << familyHamiltonian;
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

        textFileDifferentialCorrection.open((directoryPath + "L" + std::to_string(librationPointNr) + "_" + orbitType + "_" + stringAccelerationAngle1 + "_" + stringAccelerationAngle2 + "_" + stringHamiltonian + "_differential_correction.txt"));

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
    std::cout << "initialConditions Size: " << initialConditions.size() << std::endl;
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
                                  << initialConditions[i][42] << std::setw(25) << initialConditions[i][43] << std::setw(25)
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
                                       << differentialCorrections[i][10] << std::setw(25) << differentialCorrections[i][11] << std::setw(25)
                                       << differentialCorrections[i][12] << std::setw(25) << differentialCorrections[i][13] << std::setw(25)
                                       << differentialCorrections[i][14] << std::setw(25) << differentialCorrections[i][15] << std::setw(25)
                                       << differentialCorrections[i][16] << std::setw(25) << differentialCorrections[i][17] << std::setw(25)
                                       << differentialCorrections[i][18] << std::setw(25) << differentialCorrections[i][19] << std::setw(25) << std::endl;
        //std::cout << "i: " << std::endl;
        //std::cout << "size statesContinuation[i]: "  << (statesContinuation[i]).size() << std::endl;
        //std::cout << "size statesContinuation[i] - 2: "  << (statesContinuation[i]).size() - 2<< std::endl;


        for (int test = 0; test < ( (statesContinuation[i]).size() ) ; test++)
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
    double scalingFactor;
    double rho_n;
   if (continuationIndex == 1)
   {
        scalingFactor = 1.0;
        //std::cout << "check rho_n elements: \n" <<  currentState.segment( 1, 3 ) << std::endl;
        rho_n = currentState.segment( 1, 3 ).norm( );
   }

   if (continuationIndex == 6)
   {
       scalingFactor = 2.0;
       rho_n = currentState.segment( 7, 1 ).norm( );

   }

   if (continuationIndex == 7)
   {
       scalingFactor = 2.0E3;
       rho_n = currentState.segment( 8, 1 ).norm( );

   }

   return (distanceIncrement * scalingFactor) / ( rho_n );
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

Eigen::VectorXd computeHamiltonianVaryingStateIncrement(const Eigen::VectorXd initialStateVector, const int numberOfCollocationPoints, const double massParameter)
{
    Eigen::VectorXd outputVector(initialStateVector.rows());;
    outputVector.setZero();

    const double currentHamiltonian = computeHamiltonian( massParameter, initialStateVector.segment(0,10));

    for(int i = 0; i < numberOfCollocationPoints; i++)
    {
        Eigen::VectorXd localIncrement = Eigen::VectorXd::Zero(11);
        localIncrement.setZero();


        Eigen::VectorXd currentNode = initialStateVector.segment(i*11,11);
        Eigen::VectorXcd statePositionAndVelocity = currentNode.segment(0,6);
        Eigen::VectorXd thrustAndMassParameters = currentNode.segment(6,4);
        double epsilon = 1.0E-10;
        std::complex<double> increment(0.0,epsilon);


        for(int j = 0; j < 6; j++)
        {
            Eigen::VectorXcd columnDesignVector = statePositionAndVelocity;
            columnDesignVector(j) = columnDesignVector(j) + increment;
            double localDerivative = computeHamiltonianDerivativeUsingComplexStep( columnDesignVector, thrustAndMassParameters, currentHamiltonian, epsilon, massParameter  );
            localIncrement(j,0) = localDerivative;
        }

        outputVector.segment(i*11,11) = localIncrement;
    }



    return outputVector;

}

void createLowThrustInitialConditions( const int librationPointNr, const double ySign, const std::string& orbitType, const int continuationIndex, const double accelerationMagnitude, const double accelerationAngle,
                                       const double accelerationAngle2, const double initialMass, const double familyHamiltonian,
                              const double massParameter, const int numberOfPatchPoints, const int initialNumberOfCollocationPoints, const double maxPositionDeviationFromPeriodicOrbit, const double maxVelocityDeviationFromPeriodicOrbit, const double maxPeriodDeviationFromPeriodicOrbit, const double maxEigenvalueDeviation,
                              const boost::function< double( const Eigen::VectorXd&, const int ) > pseudoArcLengthFunctionAugmented ) {

    bool startContinuationFromTextFile = true;
    std::cout << "\nCreate initial conditions:" << std::endl;
    std::cout << "Start continuation from text file: " << startContinuationFromTextFile << "\n"<<std::endl;


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
    int numberOfCollocationPoints;
    // Obtain ballistic initial guesses and refine them

    if (continuationIndex == 1 and startContinuationFromTextFile == false)
    {
        linearApproximationResultIteration1 = getLowThrustInitialStateVectorGuess(librationPointNr, ySign, orbitType, accelerationMagnitude, accelerationAngle, accelerationAngle2, initialMass, continuationIndex, numberOfPatchPoints, 0);
        linearApproximationResultIteration2 = getLowThrustInitialStateVectorGuess(librationPointNr, ySign, orbitType, accelerationMagnitude, accelerationAngle, accelerationAngle2, initialMass, continuationIndex, numberOfPatchPoints, 1);

        stateVectorInclSTM =  getCorrectedAugmentedInitialState(
                    linearApproximationResultIteration1, computeHamiltonian( massParameter, linearApproximationResultIteration1.segment(0,10)), 0,
                   librationPointNr, orbitType, massParameter, numberOfPatchPoints, initialNumberOfCollocationPoints,false, initialConditions, differentialCorrections, statesContinuation,
                    maxPositionDeviationFromPeriodicOrbit, maxVelocityDeviationFromPeriodicOrbit );
        stateVectorInclSTM =  getCorrectedAugmentedInitialState(
                    linearApproximationResultIteration2, computeHamiltonian( massParameter, linearApproximationResultIteration1.segment(0,10)), 1,
                   librationPointNr, orbitType, massParameter, numberOfPatchPoints, initialNumberOfCollocationPoints, false, initialConditions, differentialCorrections, statesContinuation,
                    maxPositionDeviationFromPeriodicOrbit, maxVelocityDeviationFromPeriodicOrbit );
    } else if (startContinuationFromTextFile == false)
    {
        std::cout << "StatesContinuationVector: computed" << std::endl;

        Eigen::VectorXd statesContinuationVector = refineOrbitHamiltonian(librationPointNr, orbitType, accelerationMagnitude,accelerationAngle, accelerationAngle2,
                                                                          familyHamiltonian, massParameter, continuationIndex, numberOfCollocationPoints);

        std::cout << "StatesContinuationVector: computed" << std::endl;
        // Compute the interior points and nodes for each segment, this is the input for the getCollocated State
        Eigen::MatrixXd oddNodesMatrix((11*(numberOfCollocationPoints-1)), 4 );
        computeOddPoints(statesContinuationVector, oddNodesMatrix, numberOfCollocationPoints, massParameter, false);

        Eigen::VectorXd hamiltonianVector = Eigen::VectorXd::Zero(1);
        hamiltonianVector(0) = familyHamiltonian;
        stateVectorInclSTM = getCollocatedAugmentedInitialState(oddNodesMatrix, 0, librationPointNr, orbitType, continuationIndex, hamiltonianVector,
                                                                massParameter, numberOfPatchPoints, numberOfCollocationPoints, initialConditions,
                                                                differentialCorrections, statesContinuation, maxPositionDeviationFromPeriodicOrbit, maxVelocityDeviationFromPeriodicOrbit, maxPeriodDeviationFromPeriodicOrbit);

          }


// ============ CONTINUATION PROCEDURE ================== //
    // Set exit parameters of continuation procedure
    int maximumNumberOfInitialConditions = 55;
    int numberOfInitialConditions;
    if (continuationIndex == 1)
    {
        numberOfInitialConditions = 2;
    } else
    {
        numberOfInitialConditions = 1;
    }

    // Generate periodic orbits until termination
    double orbitalPeriod  = 0.0, periodIncrement = 0.0;
    int orderOfMagnitude, minimumIncrementOrderOfMagnitude;
     orderOfMagnitude = 5;
     minimumIncrementOrderOfMagnitude = 7;


    double pseudoArcLengthCorrection = 0.0;
    bool continueNumericalContinuation = true;
    double targetHamiltonian;
    bool maxThrustOrFullRevolutionReached = false;
    double initialAngle = 0.0;
    Eigen::VectorXd adaptedIncrementVector = Eigen::VectorXd::Zero(6);

    if (continuationIndex == 1)
    {
        numberOfCollocationPoints = initialNumberOfCollocationPoints;

        if (startContinuationFromTextFile == true)
        {
            Eigen::VectorXd statesContinuationVectorFirstGuess;
            Eigen::VectorXd statesContinuationVectorSecondGuess;
            int numberOfCollocationPointsFirstGuess;
            int numberOfCollocationPointsSecondGuess;

           // ORBIT ID // HAMILTONIAN
           // 000      // -1.594170534760879
           // 001      // -1.594170243726332
           // 498      // -1.516776129756689
           // 499      // -1.516655149071308
           //
            initialiseContinuationFromTextFile( librationPointNr, orbitType, accelerationMagnitude, accelerationAngle, accelerationAngle2,
                                                -1.594170534760879, -1.594170243726332, ySign, massParameter,
                                                statesContinuationVectorFirstGuess, statesContinuationVectorSecondGuess,
                                                numberOfCollocationPointsFirstGuess, numberOfCollocationPointsSecondGuess, adaptedIncrementVector, numberOfInitialConditions);

            // Compute the interior points and nodes for each segment, this is the input for the getCollocated State
          Eigen::MatrixXd oddNodesMatrixFirst((11*(numberOfCollocationPointsFirstGuess-1)), 4 );
          computeOddPoints(statesContinuationVectorFirstGuess, oddNodesMatrixFirst, numberOfCollocationPointsFirstGuess, massParameter, false);

          Eigen::MatrixXd oddNodesMatrixSecond((11*(numberOfCollocationPointsSecondGuess-1)), 4 );
          computeOddPoints(statesContinuationVectorSecondGuess, oddNodesMatrixSecond, numberOfCollocationPointsSecondGuess, massParameter, false);

          int orbitNumberFirstGuess = numberOfInitialConditions-2;
          int orbitNumberSecondGuess = numberOfInitialConditions-1;


            Eigen::MatrixXd stateVectorInclSTMFirst = getCollocatedAugmentedInitialState(oddNodesMatrixFirst, orbitNumberFirstGuess, librationPointNr, orbitType, 1, adaptedIncrementVector,
                                                                                    massParameter, numberOfCollocationPointsFirstGuess, numberOfCollocationPointsFirstGuess, initialConditions, differentialCorrections,
                                                                                    statesContinuation, 1.0E-12, 1.0E-12, 1.0E-12);
            Eigen::MatrixXd stateVectorInclSTMSecond = getCollocatedAugmentedInitialState(oddNodesMatrixSecond, orbitNumberSecondGuess, librationPointNr, orbitType, 1, adaptedIncrementVector,
                                                                                    massParameter, numberOfCollocationPointsFirstGuess, numberOfCollocationPointsFirstGuess, initialConditions, differentialCorrections,
                                                                                    statesContinuation, 1.0E-12, 1.0E-12, 1.0E-12);

            numberOfCollocationPoints = numberOfCollocationPointsSecondGuess;
        }

    }

    while( ( numberOfInitialConditions < maximumNumberOfInitialConditions ) && continueNumericalContinuation)
    {

        std::cout << "========== Numerical continuation Status Update L" << librationPointNr << "_" << orbitType << "_" << "acc = " << accelerationMagnitude << ", alpha = " << accelerationAngle <<  " ========== "<< std::endl
                << "Creating initial guess number "  << numberOfInitialConditions + 1 << std::endl
                << "Continuating along continuation index "  << continuationIndex << std::endl
                << "============================================================ " << std::endl;

         double incrementContinuationParameter =  pow(10,(static_cast<float>(-orderOfMagnitude)));

          if(continuationIndex == 1)
          {
              int numberOfStates =  numberOfStates = 3*(numberOfCollocationPoints-1)+1;

              Eigen::VectorXd initialStateVectorContinuation (11* numberOfStates);
              initialStateVectorContinuation.setZero();


              initialStateVectorContinuation = statesContinuation[ statesContinuation.size( ) - 1 ].segment( 3, 11*numberOfStates );

              if (numberOfInitialConditions == 2)
              {
                 Eigen::VectorXd increment = Eigen::VectorXd::Zero(6);
                 Eigen::VectorXd fullEquilibriumLocation = Eigen::VectorXd::Zero(6);
                 fullEquilibriumLocation.segment(0,2) = createEquilibriumLocations(1, accelerationMagnitude, accelerationAngle, "acceleration", ySign, massParameter );
                 increment = initialStateVectorContinuation.segment(0,6) - fullEquilibriumLocation;
                 adaptedIncrementVector = 10.0 *increment / (increment.norm());

                 //adaptedIncrementVector = computeStateDerivativeAugmented(0.0, getFullInitialStateAugmented( initialStateVectorContinuation.segment(0,10) ) ).block(0,0,6,1);

                 std::cout << "fullEquilibriumLocation: \n" << fullEquilibriumLocation << std::endl;
                 std::cout << "increment: \n" << increment << std::endl;
                 std::cout << "adaptedIncrementVector: \n" << adaptedIncrementVector << std::endl;
                 std::cout << "adaptedIncrementVector.norm: " << adaptedIncrementVector.norm() << std::endl;
                 std::cout << "adaptedIncrementVector * 5: " << adaptedIncrementVector* 5 << std::endl;
                 std::cout << "adaptedIncrementVector * 5: " << (adaptedIncrementVector* 5).norm() << std::endl;

                    //std::cout << "first state of most recent guess: \n" << initialStateVectorContinuation.segment(0,6) << std::endl;
                    //std::cout << "adaptedIncrementVector: \n" << adaptedIncrementVector << std::endl;



              }

              // SHOULD BE MINUS 1 BUT FOR CONSTRUCTION IS -2
              Eigen::VectorXd previousDesignVector = (statesContinuation[ statesContinuation.size( ) - 2 ].segment( 3, 11*numberOfStates ));



              // perform numerical continuation
              Eigen::VectorXd stateIncrement(11*numberOfStates+1);
              stateIncrement.setZero();


             if ( statesContinuation[ statesContinuation.size( ) - 1 ].size() == statesContinuation[ statesContinuation.size( ) - 2 ].size() )
             {
                 std::cout << "number of patch point of guesses is similar!: " << std::endl
                            << "size statesContinuation[size-2]: " << statesContinuation[ statesContinuation.size( ) - 2 ].size() << std::endl
                            << "size statesContinuation[size-1]: " << statesContinuation[ statesContinuation.size( ) - 1 ].size() << std::endl;

//                 stateIncrement.segment(1,11*numberOfStates) = statesContinuation[ statesContinuation.size( ) - 1 ].segment( 3, 11*numberOfStates ) -
//                                     statesContinuation[ statesContinuation.size( ) - 2 ].segment( 3, 11*numberOfStates );
//                 stateIncrement(0) = statesContinuation[ statesContinuation.size( ) - 1 ]( 1 ) -
//                                     statesContinuation[ statesContinuation.size( ) - 2 ]( 1 );
//                 previousDesignVector = statesContinuation[ statesContinuation.size( ) - 1 ].segment( 3, 11*numberOfStates );

                 Eigen::VectorXd stateIncrementInterpolation(11*numberOfStates);  stateIncrementInterpolation.setZero();

                  stateIncrement(0) = statesContinuation[ statesContinuation.size( ) - 1 ]( 1 ) -
                                      statesContinuation[ statesContinuation.size( ) - 2 ]( 1 );

                   Eigen::VectorXd previousGuess = statesContinuation[ statesContinuation.size( ) - 2 ];
                   Eigen::VectorXd currentGuess = statesContinuation[ statesContinuation.size( ) - 1 ];


                   computeStateIncrementFromInterpolation(previousGuess, currentGuess, stateIncrementInterpolation );
                   stateIncrement.segment(1,11*numberOfStates) = stateIncrementInterpolation;

                   // define previousDesignVector
                   int previousNumberOfStates = (static_cast<int>(statesContinuation[ statesContinuation.size( ) - 1 ].size()) - 3)/11;
                   previousDesignVector = statesContinuation[ statesContinuation.size( ) - 1 ].segment( 3, 11*previousNumberOfStates );



             } else
             {
                 std::cout << "number of patch point of guesses has been changed!!: " << std::endl
                            << "size statesContinuation[size-2]: " << statesContinuation[ statesContinuation.size( ) - 2 ].size() << std::endl
                            << "size statesContinuation[size-1]: " << statesContinuation[ statesContinuation.size( ) - 1 ].size() << std::endl;

                 Eigen::VectorXd stateIncrementInterpolation(11*numberOfStates);  stateIncrementInterpolation.setZero();

                 stateIncrement(0) = statesContinuation[ statesContinuation.size( ) - 1 ]( 1 ) -
                                     statesContinuation[ statesContinuation.size( ) - 2 ]( 1 );

                 Eigen::VectorXd previousGuess = statesContinuation[ statesContinuation.size( ) - 2 ];
                 Eigen::VectorXd currentGuess = statesContinuation[ statesContinuation.size( ) - 1 ];

                 computeStateIncrementFromInterpolation(previousGuess, currentGuess, stateIncrementInterpolation );

                 stateIncrement.segment(1,11*numberOfStates) = stateIncrementInterpolation;

                 // define previousDesignVector
                 int previousNumberOfStates = (static_cast<int>(statesContinuation[ statesContinuation.size( ) - 1 ].size()) - 3)/11;
                 previousDesignVector = statesContinuation[ statesContinuation.size( ) - 1 ].segment( 3, 11*previousNumberOfStates );


             }
//                Eigen::Vector6d x1 = statesContinuation[ statesContinuation.size( ) - 1 ].segment( 3, 6 );
//                Eigen::Vector6d x0 = statesContinuation[ statesContinuation.size( ) - 2 ].segment( 3, 6 );
//                Eigen::Vector6d difference = x1-x0;
//                Eigen::Vector6d x0dot = computeStateDerivativeAugmented(0.0, getFullInitialStateAugmented( statesContinuation[ statesContinuation.size( ) - 2 ].segment( 3, 10 ))).block(0,0,6,1);


//                std::cout << "\n===== TESTING INTEGRAL PHASE CONSTRAINT =====" << std::endl
//                           << "x1: \n" << x1 << std::endl
//                           << "x0: \n" << x0 << std::endl
//                           << "difference: \n" << difference << std::endl
//                           << "x0dot: \n" << x0dot << std::endl
//                           << "x1^T x0dot: " << x1.transpose()*x0dot << std::endl
//                           << "difference.transpose x0: " << difference.transpose()*x0dot << std::endl;


                 double pseudoArcLengthCorrection = pseudoArcLengthFunctionAugmented( stateIncrement.segment(0,11), continuationIndex );


                initialStateVectorContinuation = initialStateVectorContinuation + pseudoArcLengthCorrection* stateIncrement.segment(1,numberOfStates*11);


                 // Compute the interior points and nodes for each segment, this is the input for the getCollocated State
              Eigen::MatrixXd oddNodesMatrix((11*(numberOfCollocationPoints-1)), 4 );
              computeOddPoints(initialStateVectorContinuation, oddNodesMatrix, numberOfCollocationPoints, massParameter, false);


              stateVectorInclSTM = getCollocatedAugmentedInitialState( oddNodesMatrix, numberOfInitialConditions, librationPointNr, orbitType, continuationIndex, adaptedIncrementVector,
                                                                       massParameter, numberOfPatchPoints, numberOfCollocationPoints, initialConditions,
                                                                       differentialCorrections, statesContinuation, maxPositionDeviationFromPeriodicOrbit, maxVelocityDeviationFromPeriodicOrbit, maxPeriodDeviationFromPeriodicOrbit);


          }
          if (continuationIndex == 6)
          {

              int numberOfStates =  3*(numberOfCollocationPoints-1)+1;

              Eigen::VectorXd initialStateVectorContinuation (11* numberOfStates);
              initialStateVectorContinuation.setZero();
              initialStateVectorContinuation = statesContinuation[statesContinuation.size()-1].segment(3,11*numberOfStates);

              // Compute the interior points and nodes for each segment, this is the input for the getCollocated State
              Eigen::MatrixXd oddNodesMatrix((11*(numberOfCollocationPoints-1)), 4 );
              computeOddPoints(initialStateVectorContinuation, oddNodesMatrix, numberOfCollocationPoints, massParameter, false);


               //Add thrust increment to all nodes and interior Points!
               for(int i = 0; i < (numberOfCollocationPoints-1); i ++)
               {
                   for(int j = 0; j < 4; j++)
                   {
                       if( oddNodesMatrix(11*i+continuationIndex,j) + incrementContinuationParameter > 0.1)
                       {
                            oddNodesMatrix(11*i+continuationIndex,j) = 0.1;
                       } else {

                           oddNodesMatrix(11*i+continuationIndex,j) = oddNodesMatrix(11*i+continuationIndex,j) + incrementContinuationParameter;

                       }

                   }

               }


              Eigen::VectorXd previousDesignVector(1); previousDesignVector.setZero();
              previousDesignVector(0) = familyHamiltonian;
              stateVectorInclSTM = getCollocatedAugmentedInitialState( oddNodesMatrix, numberOfInitialConditions, librationPointNr, orbitType, continuationIndex, previousDesignVector,
                                                                       massParameter, numberOfPatchPoints, numberOfCollocationPoints, initialConditions,
                                                                       differentialCorrections, statesContinuation, maxPositionDeviationFromPeriodicOrbit, maxVelocityDeviationFromPeriodicOrbit, maxPeriodDeviationFromPeriodicOrbit);



          }
          if (continuationIndex == 7)
          {
             if(numberOfInitialConditions == 1)
             {
                initialAngle = stateVectorInclSTM(0,7);
             }

             int numberOfStates =  3*(numberOfCollocationPoints-1)+1;

             Eigen::VectorXd initialStateVectorContinuation (11* numberOfStates);
             initialStateVectorContinuation.setZero();
             initialStateVectorContinuation = statesContinuation[statesContinuation.size()-1].segment(3,11*numberOfStates);

             // Compute the interior points and nodes for each segment, this is the input for the getCollocated State
             Eigen::MatrixXd oddNodesMatrix((11*(numberOfCollocationPoints-1)), 4 );
             computeOddPoints(initialStateVectorContinuation, oddNodesMatrix, numberOfCollocationPoints, massParameter, false);


              //Add thrust increment to all nodes and interior Points!
              for(int i = 0; i < (numberOfCollocationPoints-1); i ++)
              {
                  for(int j = 0; j < 4; j++)
                  {
                      if( ( std::abs(stateVectorInclSTM(0,7) - initialAngle )) >= 360)
                      {
                           oddNodesMatrix(11*i+continuationIndex,j) = initialAngle +360;
                      } else {

                          oddNodesMatrix(11*i+continuationIndex,j) = oddNodesMatrix(11*i+continuationIndex,j) + incrementContinuationParameter;

                      }

                  }

              }


             Eigen::VectorXd previousDesignVector(1); previousDesignVector.setZero();
             previousDesignVector(0) = familyHamiltonian;
             stateVectorInclSTM = getCollocatedAugmentedInitialState( oddNodesMatrix, numberOfInitialConditions, librationPointNr, orbitType, continuationIndex, previousDesignVector,
                                                                      massParameter, numberOfPatchPoints, numberOfCollocationPoints, initialConditions,
                                                                      differentialCorrections, statesContinuation, maxPositionDeviationFromPeriodicOrbit, maxVelocityDeviationFromPeriodicOrbit, maxPeriodDeviationFromPeriodicOrbit);

          }

          continueNumericalContinuation = checkTerminationAugmented(differentialCorrections, stateVectorInclSTM, orbitType, librationPointNr, maxEigenvalueDeviation );
          std::cout << "continueNumericalContinuation: " << continueNumericalContinuation << std::endl;

            if(continuationIndex == 6 or continuationIndex == 7)
            {
                if (stateVectorInclSTM(0,6) > 0.1)
                {
                    continueNumericalContinuation = 0;
                    maxThrustOrFullRevolutionReached = true;
                }

                if( ( std::abs(stateVectorInclSTM(0,7) - initialAngle )) >= 360)
                {
                    continueNumericalContinuation = 0;
                    maxThrustOrFullRevolutionReached = true;
                }

            }

            if(continuationIndex == 7)
            {

            }

            std::cout << "continueNumericalContinuation: " << continueNumericalContinuation << std::endl;


           if ( continuationIndex != 1 && continueNumericalContinuation == false && orderOfMagnitude > minimumIncrementOrderOfMagnitude && maxThrustOrFullRevolutionReached == false )
                  {
                         orderOfMagnitude = orderOfMagnitude + 1;
                         continueNumericalContinuation = true;
                   }

           numberOfInitialConditions += 1;



    }

    writeFinalResultsToFilesAugmented( librationPointNr, orbitType, continuationIndex, accelerationMagnitude, accelerationAngle, accelerationAngle2, familyHamiltonian, numberOfPatchPoints, initialConditions, differentialCorrections, statesContinuation );

}



 //propagateAndSaveCollocationProcedure(oddNodesMatrix, Eigen::VectorXd::Zero(numberOfCollocationPoints-1), Eigen::VectorXd::Zero(4), numberOfCollocationPoints, 1, massParameter);

//              // Generate random noise
//              int numberOfNodes = 3*(numberOfCollocationPoints-1)+1;
//              int numberOfStates2 = numberOfNodes*6;
//              Eigen::VectorXd noiseVector(numberOfStates2);
//              noiseVector.setZero();

//              std::random_device                  rand_dev;
//              std::mt19937                        generator(rand_dev());
//              std::uniform_real_distribution<double>  distr(1.0E-12, 1.0E-05);

//              std::random_device                  rand_dev2;
//              std::mt19937                        generator2(rand_dev2());
//              std::uniform_real_distribution<double>  distr2(0, 1);

//              for(int i = 0; i < numberOfStates2; i++)
//                   {

//                       double sign = 0.0;
//                       if ( distr2(generator2) < 0.5 )
//                       {
//                            sign = -1.0;
//                        } else
//                         {
//                            sign = 1.0;
//                          }
//                       if ( (i+1) % 3 == 0 and i > 0)
//                          {
//                            noiseVector(i) = 0.0;
//                          } else
//                          {
//                            noiseVector(i) = sign*distr(generator);

//                          }

//                     }
