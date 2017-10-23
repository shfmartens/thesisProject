#include <fstream>
#include <iomanip>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>
#include <Eigen/Dense>

#include "Tudat/Astrodynamics/BasicAstrodynamics/celestialBodyConstants.h"
#include "Tudat/Astrodynamics/Gravitation/librationPoint.h"
#include "Tudat/Astrodynamics/Gravitation/jacobiEnergy.h"

#include "createInitialConditions.h"
#include "applyDifferentialCorrection.h"
#include "checkEigenvalues.h"
#include "propagateOrbit.h"
#include "richardsonThirdOrderApproximation.h"

void appendResultsVector(const double jacobiEnergy, const double orbitalPeriod, const Eigen::VectorXd& initialStateVector,
        const Eigen::VectorXd& stateVectorInclSTM, std::vector< Eigen::VectorXd >& initialConditions )
{
    Eigen::VectorXd tempInitialCondition = Eigen::VectorXd( 44 );

    // Add Jacobi energy and orbital period
    tempInitialCondition( 0 ) = jacobiEnergy;
    tempInitialCondition( 1 ) = orbitalPeriod;

    // Add initial condition of periodic solution
    for (int i = 0; i <= 5; i++){
        tempInitialCondition( i + 2 ) = initialStateVector( i );
    }

    // Add Monodromy matrix
    for (int i = 6; i <= 41; i++){
        tempInitialCondition( i + 2 ) = stateVectorInclSTM(i);
    }

    initialConditions.push_back(tempInitialCondition);
}

void appendDifferentialCorrectionResultsVector(
        const double jacobiEnergyHalfPeriod,  const Eigen::VectorXd& differentialCorrectionResult,
        std::vector< Eigen::VectorXd >& differentialCorrections )
{

    Eigen::VectorXd tempDifferentialCorrection = Eigen::VectorXd( 9 );
    tempDifferentialCorrection( 0 ) = differentialCorrectionResult( 14 );  // numberOfIterations
    tempDifferentialCorrection( 1 ) = jacobiEnergyHalfPeriod;  // jacobiEnergyHalfPeriod
    tempDifferentialCorrection( 2 ) = differentialCorrectionResult( 13 );  // currentTime
    for (int i = 7; i <= 12; i++)
    {
        tempDifferentialCorrection( i - 4 ) = differentialCorrectionResult( i );  // halfPeriodStateVector
    }

    differentialCorrections.push_back(tempDifferentialCorrection);

}

double getEarthMoonAmplitude( const int librationPointNr, const std::string& orbitType, const int guessIteration )
{
    double amplitude;
    if( guessIteration == 0 )
    {
        if (orbitType == "horizontal")
        {
            if (librationPointNr == 1)
            {
                amplitude = 1.0e-3;
            }
            else if (librationPointNr == 2)
            {
                amplitude = 1.0e-4;
            }
        }
        else if (orbitType == "vertical")
        {
            if (librationPointNr == 1)
            {
                amplitude = 1.0e-1;
            }
            else if (librationPointNr == 2)
            {
                amplitude = 1.0e-1;
            }
        }
        else if (orbitType == "halo")
        {
            if (librationPointNr == 1)
            {
                amplitude = -1.1e-1;
            }
            else if (librationPointNr == 2)
            {
                amplitude = 1.5e-1;
            }
        }
    }
    else if( guessIteration == 1 )
    {

        if (orbitType == "horizontal")
        {
            if (librationPointNr == 1)
            {
                amplitude = 1.0e-4;
            }
            else if (librationPointNr == 2)
            {
                amplitude = 1.0e-3;
            }
        }
        else if (orbitType == "vertical")
        {
            if (librationPointNr == 1)
            {
                amplitude = 2.0e-1;
            }
            else if (librationPointNr == 2)
            {
                amplitude = 2.0e-1;
            }
        }
        else if (orbitType == "halo")
        {
            if (librationPointNr == 1)
            {
                amplitude = -1.2e-1;
            }
            else if (librationPointNr == 2)
            {
                amplitude = 1.6e-1;
            }
        }
    }

    return amplitude;
}

Eigen::Vector7d getInitialStateVectorGuess( const int librationPointNr, const std::string& orbitType, const int guessIteration,
                                            const boost::function< double( const int librationPointNr, const std::string& orbitType, const int guessIteration ) > getAmplitude )
{
    double amplitude = getAmplitude( librationPointNr, orbitType, guessIteration );
    Eigen::Vector7d richardsonThirdOrderApproximationResult = richardsonThirdOrderApproximation(orbitType, librationPointNr, amplitude);

    return richardsonThirdOrderApproximationResult;
}

Eigen::MatrixXd getCorrectedInitialState( const Eigen::Vector6d& initialStateGuess, double orbitalPeriod, const int orbitNumber,
                                          const int librationPointNr, std::string orbitType, const double massParameter,
                                          std::vector< Eigen::VectorXd >& initialConditions,
                                          std::vector< Eigen::VectorXd >& differentialCorrections,
                                          const double maxPositionDeviationFromPeriodicOrbit, double maxVelocityDeviationFromPeriodicOrbit )
{
    Eigen::Vector6d initialStateVector = initialStateGuess;

    // Correct state vector guess
    Eigen::VectorXd differentialCorrectionResult = applyDifferentialCorrection(
                librationPointNr, orbitType, initialStateVector, orbitalPeriod, massParameter,
                maxPositionDeviationFromPeriodicOrbit, maxVelocityDeviationFromPeriodicOrbit );
    initialStateVector = differentialCorrectionResult.segment( 0, 6 );
    orbitalPeriod = differentialCorrectionResult( 6 );

    // Propagate the initialStateVector for a full period and write output to file.
    std::map< double, Eigen::Vector6d > stateHistory;
    Eigen::MatrixXd stateVectorInclSTM = propagateOrbitToFinalCondition(
                getFullInitialState( initialStateVector ), massParameter, orbitalPeriod, 1, stateHistory, 1000, 0.0 ).first;
    writeStateHistoryToFile( stateHistory, orbitNumber, orbitType, librationPointNr, 1000, false );

    // Save results
    double jacobiEnergyHalfPeriod = tudat::gravitation::computeJacobiEnergy( massParameter, differentialCorrectionResult.segment( 7, 6 ) );
    appendDifferentialCorrectionResultsVector( jacobiEnergyHalfPeriod, differentialCorrectionResult, differentialCorrections );

    double jacobiEnergy = tudat::gravitation::computeJacobiEnergy( massParameter, stateVectorInclSTM.block( 0, 0, 6, 1 ));
    appendResultsVector( jacobiEnergy, orbitalPeriod, initialStateVector, stateVectorInclSTM, initialConditions );

    return stateVectorInclSTM;

}

void writeFinalResultsToFiles( const int librationPointNr, const std::string orbitType,
                               std::vector< Eigen::VectorXd > initialConditions,
                               std::vector< Eigen::VectorXd > differentialCorrections )
{
    // Prepare file for initial conditions
    remove(("../data/raw/orbits/L" + std::to_string(librationPointNr) + "_" + orbitType + "_initial_conditions.txt").c_str());
    std::ofstream textFileInitialConditions;
    textFileInitialConditions.open(("../data/raw/orbits/L" + std::to_string(librationPointNr) + "_" + orbitType + "_initial_conditions.txt"));
    textFileInitialConditions.precision(std::numeric_limits<double>::digits10);

    // Prepare file for differential correction
    remove(("../data/raw/orbits/L" + std::to_string(librationPointNr) + "_" + orbitType + "_differential_correction.txt").c_str());
    std::ofstream textFileDifferentialCorrection;
    textFileDifferentialCorrection.open(("../data/raw/orbits/L" + std::to_string(librationPointNr) + "_" + orbitType + "_differential_correction.txt"));
    textFileDifferentialCorrection.precision(std::numeric_limits<double>::digits10);

    // Write initial conditions to file
    for (unsigned int i=0; i<initialConditions.size(); i++) {

        textFileInitialConditions << std::left << std::scientific                                          << std::setw(25)
                                  << initialConditions[i][0]  << std::setw(25) << initialConditions[i][1 ]  << std::setw(25)
                                  << initialConditions[i][2 ]  << std::setw(25) << initialConditions[i][3]  << std::setw(25)
                                  << initialConditions[i][4]  << std::setw(25) << initialConditions[i][5]  << std::setw(25)
                                  << initialConditions[i][6]  << std::setw(25) << initialConditions[i][7]  << std::setw(25)
                                  << initialConditions[i][8]  << std::setw(25) << initialConditions[i][9]  << std::setw(25)
                                  << initialConditions[i][10] << std::setw(25) << initialConditions[i][11 ] << std::setw(25)
                                  << initialConditions[i][12 ] << std::setw(25) << initialConditions[i][13] << std::setw(25)
                                  << initialConditions[i][14] << std::setw(25) << initialConditions[i][15] << std::setw(25)
                                  << initialConditions[i][16] << std::setw(25) << initialConditions[i][17] << std::setw(25)
                                  << initialConditions[i][18] << std::setw(25) << initialConditions[i][19] << std::setw(25)
                                  << initialConditions[i][20] << std::setw(25) << initialConditions[i][21 ] << std::setw(25)
                                  << initialConditions[i][22 ] << std::setw(25) << initialConditions[i][23] << std::setw(25)
                                  << initialConditions[i][24] << std::setw(25) << initialConditions[i][25] << std::setw(25)
                                  << initialConditions[i][26] << std::setw(25) << initialConditions[i][27] << std::setw(25)
                                  << initialConditions[i][28] << std::setw(25) << initialConditions[i][29] << std::setw(25)
                                  << initialConditions[i][30] << std::setw(25) << initialConditions[i][31 ] << std::setw(25)
                                  << initialConditions[i][32 ] << std::setw(25) << initialConditions[i][33] << std::setw(25)
                                  << initialConditions[i][34] << std::setw(25) << initialConditions[i][35] << std::setw(25)
                                  << initialConditions[i][36] << std::setw(25) << initialConditions[i][37] << std::setw(25)
                                  << initialConditions[i][38] << std::setw(25) << initialConditions[i][39] << std::setw(25)
                                  << initialConditions[i][40] << std::setw(25) << initialConditions[i][41 ] << std::setw(25)
                                  << initialConditions[i][42 ] << std::setw(25) << initialConditions[i][43] << std::endl;

        textFileDifferentialCorrection << std::left << std::scientific   << std::setw(25)
                                       << differentialCorrections[i][0]  << std::setw(25) << differentialCorrections[i][1 ]  << std::setw(25)
                                       << differentialCorrections[i][2 ]  << std::setw(25) << differentialCorrections[i][3]  << std::setw(25)
                                       << differentialCorrections[i][4]  << std::setw(25) << differentialCorrections[i][5]  << std::setw(25)
                                       << differentialCorrections[i][6]  << std::setw(25) << differentialCorrections[i][7]  << std::setw(25)
                                       << differentialCorrections[i][8]  << std::setw(25) << std::endl;
    }
}

bool checkTermination( const std::vector< Eigen::VectorXd >& differentialCorrections,
                       const Eigen::MatrixXd& stateVectorInclSTM, const std::string orbitType, const int librationPointNr,
                       const double maxEigenvalueDeviation )
{
    // Check termination condtions

    bool continueNumericalContinuation = true;
    if ( differentialCorrections.at( differentialCorrections.size( ) - 1 ) == Eigen::VectorXd::Zero( 15 ) )
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

void createInitialConditions( const int librationPointNr, const std::string& orbitType,
                              const double massParameter,
                              const double maxPositionDeviationFromPeriodicOrbit, const double maxVelocityDeviationFromPeriodicOrbit,
                              const double maxEigenvalueDeviation )

{
    std::cout << "\nCreate initial conditions:\n" << std::endl;

    // Set output maximum precision
    std::cout.precision(std::numeric_limits<double>::digits10);

    // Initialize state vectors and orbital periods
    Eigen::Vector6d initialStateVector = Eigen::Vector6d::Zero( );
    Eigen::MatrixXd stateVectorInclSTM = Eigen::MatrixXd::Zero( 6, 7 );

    std::vector< Eigen::VectorXd > initialConditions;
    std::vector< Eigen::VectorXd > differentialCorrections;

    // Perform first two iteration
    Eigen::Vector7d richardsonThirdOrderApproximationResultIteration1 =
            getInitialStateVectorGuess( librationPointNr, orbitType, 0 );
    Eigen::Vector7d richardsonThirdOrderApproximationResultIteration2 =
            getInitialStateVectorGuess( librationPointNr, orbitType, 1 );
    stateVectorInclSTM = getCorrectedInitialState(
                richardsonThirdOrderApproximationResultIteration1.segment(0,6), richardsonThirdOrderApproximationResultIteration1( 6 ), 0,
                librationPointNr, orbitType, massParameter, initialConditions, differentialCorrections,
                maxPositionDeviationFromPeriodicOrbit, maxVelocityDeviationFromPeriodicOrbit );
    stateVectorInclSTM = getCorrectedInitialState(
                richardsonThirdOrderApproximationResultIteration2.segment(0,6), richardsonThirdOrderApproximationResultIteration2( 6 ), 1,
                librationPointNr, orbitType, massParameter, initialConditions, differentialCorrections,
                maxPositionDeviationFromPeriodicOrbit, maxVelocityDeviationFromPeriodicOrbit );

    // Set exit parameters of continuation procedure
    int numberOfInitialConditions = 2;
    int maximumNumberOfInitialConditions = 4000;

    // Generate periodic orbits until termination
    double orbitalPeriod  = 0.0, periodIncrement = 0.0, pseudoArcLengthCorrection = 0.0;
    bool continueNumericalContinuation = true;
    Eigen::Vector6d stateIncrement;
    while( ( numberOfInitialConditions < maximumNumberOfInitialConditions ) && continueNumericalContinuation)
    {
        // Determine increments to state and time
        stateIncrement = initialConditions[ initialConditions.size( ) - 1 ].segment( 2, 6 ) -
                initialConditions[ initialConditions.size( ) - 2 ].segment( 2, 6 );
        periodIncrement = initialConditions[ initialConditions.size( ) - 1 ]( 1 ) -
                initialConditions[ initialConditions.size( ) - 2 ]( 1 );
        pseudoArcLengthCorrection = 1e-4 / sqrt(pow(stateIncrement( 0 ),2) + pow(stateIncrement( 1 ),2) + pow(stateIncrement( 2 ),2 ) );

        // Apply numerical continuation
        initialStateVector = initialConditions[ initialConditions.size( ) - 1 ].segment( 2, 6 ) +
                stateIncrement * pseudoArcLengthCorrection;
        orbitalPeriod = initialConditions[ initialConditions.size( ) - 1 ]( 1 ) +
                periodIncrement * pseudoArcLengthCorrection;
        stateVectorInclSTM = getCorrectedInitialState(
                    initialStateVector, orbitalPeriod, numberOfInitialConditions,
                    librationPointNr, orbitType, massParameter, initialConditions, differentialCorrections,
                    maxPositionDeviationFromPeriodicOrbit, maxVelocityDeviationFromPeriodicOrbit );

        continueNumericalContinuation = checkTermination(differentialCorrections, stateVectorInclSTM, orbitType, librationPointNr, maxEigenvalueDeviation );

        numberOfInitialConditions += 1;
    }

    writeFinalResultsToFiles( librationPointNr, orbitType, initialConditions, differentialCorrections );
}
