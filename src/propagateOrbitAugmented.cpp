#include "Tudat/Mathematics/NumericalIntegrators/rungeKuttaVariableStepSizeIntegrator.h"
#include "Tudat/Mathematics/NumericalIntegrators/rungeKuttaCoefficients.h"
#include "Tudat/Mathematics/RootFinders/newtonRaphson.h"
#include "Tudat/Astrodynamics/Gravitation/jacobiEnergy.h"
#include "Tudat/Astrodynamics/BasicAstrodynamics/celestialBodyConstants.h"
#include "Tudat/Astrodynamics/Gravitation/librationPoint.h"
#include "Tudat/InputOutput/basicInputOutput.h"
#include <cmath>
#include <sstream>
#include <iomanip>


#include "propagateOrbitAugmented.h"
#include "stateDerivativeModelAugmented.h"

Eigen::VectorXd redstributeNodesOverTrajectory(const Eigen::VectorXd initialStateVector, const int numberOfPatchPoints,  const int numberOfCollocationPoints, const double massParameter)
{

    Eigen::VectorXd redistributedGuess(11*numberOfCollocationPoints);
    redistributedGuess.setZero();

    Eigen::ArrayXd timeArray = Eigen::ArrayXd::LinSpaced(numberOfCollocationPoints,0,initialStateVector(11*(numberOfPatchPoints-1) + 10 ) );

    // store the first state and time:
    redistributedGuess.segment(0,11) = initialStateVector.segment(0,11);

    // introduce loop variables
    Eigen::VectorXd stateVectorOnly = initialStateVector.segment(0,10);
    double initialTime = timeArray(0);
    double finalTime;
    for(int i = 1; i < numberOfCollocationPoints; i++)
    {
        Eigen::MatrixXd initialStateInclSTM(10,11);
        initialStateInclSTM.block(0,0,10,1) = stateVectorOnly;
        finalTime = timeArray(i);
        std::map< double, Eigen::VectorXd > stateHistory;
        std::pair< Eigen::MatrixXd, double > StateInclSTMandTime = propagateOrbitAugmentedToFinalCondition( initialStateInclSTM,
                                                                                                            massParameter, finalTime, 1, stateHistory, -1, initialTime );
        Eigen::MatrixXd StateInclSTM = StateInclSTMandTime.first;
        initialTime                  = StateInclSTMandTime.second;
        stateVectorOnly              = StateInclSTM.block( 0, 0, 10, 1 );


        redistributedGuess.segment(i*11,10) = stateVectorOnly;
        redistributedGuess(i*11+10) = initialTime;

    }

    return redistributedGuess;
}

Eigen::MatrixXd getFullInitialStateAugmented( const Eigen::VectorXd& initialState )
{
    Eigen::MatrixXd fullInitialState = Eigen::MatrixXd::Zero( 10, 11 );
    fullInitialState.block( 0, 0, 10, 1 ) = initialState;
    fullInitialState.block( 0, 1, 10, 10 ).setIdentity( );
    return fullInitialState;
}

void writeFloquetDataToFile ( const std::map< double, Eigen::VectorXd >& stateHistoryPeriodGuess, Eigen::VectorXd lowThrustInitialStateVectorGuess, const int librationPointNr, const std::string orbitType, const Eigen::VectorXd equilibriumStateVector, const double correctionTime, const double amplitude, Eigen::VectorXd interiorManeuverCorrection )
{
    std::string fileNameStringStateHistory;
    std::string fileNameStringManeuvers;
    std::string fileNameStringResultingGuess;


    std::string directoryString = "../data/raw/initial_guess/";

    fileNameStringStateHistory = ("L" + std::to_string(librationPointNr) + "_" + orbitType + "_" + std::to_string(equilibriumStateVector(6)) + "_" + std::to_string(equilibriumStateVector(7)) + "_" + std::to_string( amplitude ) + "_" + std::to_string(correctionTime) + "_stateHistory.txt");
    fileNameStringManeuvers = ("L" + std::to_string(librationPointNr) + "_" + orbitType + "_" + std::to_string(equilibriumStateVector(6)) + "_" + std::to_string(equilibriumStateVector(7)) + "_" + std::to_string( amplitude ) + "_" + std::to_string(correctionTime) + "_Maneuvers.txt");
    fileNameStringResultingGuess = ("L" + std::to_string(librationPointNr) + "_" + orbitType + "_" + std::to_string(equilibriumStateVector(6)) + "_" + std::to_string(equilibriumStateVector(7)) + "_" + std::to_string( amplitude ) + "_" + std::to_string(correctionTime) + "_resultingGuess.txt");


    tudat::input_output::writeDataMapToTextFile( stateHistoryPeriodGuess, fileNameStringStateHistory, directoryString );
    tudat::input_output::writeMatrixToFile( interiorManeuverCorrection, fileNameStringManeuvers, 16, directoryString);
    tudat::input_output::writeMatrixToFile( lowThrustInitialStateVectorGuess, fileNameStringResultingGuess, 16, directoryString);



}

void writeStateHistoryAndStateVectorsToFile ( const std::map< double, Eigen::VectorXd >& stateHistory, const Eigen::VectorXd stateVectors, const Eigen::VectorXd deviationVector, const Eigen::VectorXd deviationVectorFull,
                                              const int numberOfIterations, const int correctionLevel)
{

    std::string fileNameString;
    std::string fileNameStringStateVectors;
    std::string fileNameStringDeviations;
    std::string fileNameStringDeviationsFull;

    std::string directoryString = "/Users/Sjors/Desktop/debuggingLII/stateHistory/";


    fileNameString = ("propagatedSolution_" + std::to_string(numberOfIterations) + "_" + std::to_string(correctionLevel) + ".txt");
    fileNameStringStateVectors = ("patchPoints_" + std::to_string(numberOfIterations) + "_" + std::to_string(correctionLevel) + ".txt");
    fileNameStringDeviations = ("deviations_" + std::to_string(numberOfIterations) + "_" + std::to_string(correctionLevel) + ".txt");
    fileNameStringDeviationsFull = ("deviationsFull_" + std::to_string(numberOfIterations) + "_" + std::to_string(correctionLevel) + ".txt");



    tudat::input_output::writeDataMapToTextFile( stateHistory, fileNameString, directoryString );
    tudat::input_output::writeMatrixToFile( stateVectors, fileNameStringStateVectors, 16, directoryString);
    tudat::input_output::writeMatrixToFile( deviationVector, fileNameStringDeviations, 16, directoryString);
    tudat::input_output::writeMatrixToFile( deviationVectorFull, fileNameStringDeviationsFull, 16, directoryString);


}


void writeStateHistoryToFileAugmented(
        const std::map< double, Eigen::VectorXd >& stateHistory, const double accelerationMagnitude, const double accelerationAngle, const double accelerationAngle2, const double initialHamiltonian,
        const int orbitId, const int librationPointNr, const std::string& orbitType,
        const int saveEveryNthIntegrationStep, const bool completeInitialConditionsHaloFamily )
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

    std::ostringstream ssHamiltonian;
    ssHamiltonian << std::fixed << std::setprecision(11) << initialHamiltonian;
    std::string stringHamiltonian = ssHamiltonian.str();


    std::string fileNameString;
    std::string directoryString = "../data/raw/orbits/augmented/";
    // Prepare output file
    if (saveEveryNthIntegrationStep != 1000)
    {
        if (completeInitialConditionsHaloFamily == false)
        {
            fileNameString = ("L" +std::to_string(librationPointNr) + "_" + orbitType  + "_" + stringAccelerationMagnitude + "_" + stringAccelerationAngle1 + "_" + stringAccelerationAngle2 + "_" + stringHamiltonian + "_" + std::to_string(saveEveryNthIntegrationStep) + ".txt");
        }
        else
        {
            fileNameString = ("L" +std::to_string(librationPointNr) + "_" + orbitType  + "_" + stringAccelerationMagnitude + "_" + stringAccelerationAngle1 + "_" + stringAccelerationAngle2 + "_" + stringHamiltonian + "_n_" + std::to_string(orbitId) + "_" + std::to_string(saveEveryNthIntegrationStep) + ".txt");
        }
    }
    else
    {
        if (completeInitialConditionsHaloFamily == false)
        {
            fileNameString = ("L" +std::to_string(librationPointNr) + "_" + orbitType  + "_" + stringAccelerationMagnitude + "_" + stringAccelerationAngle1 + "_" + stringAccelerationAngle2 + "_" + stringHamiltonian + "_" + ".txt");
        }
        else
        {
            fileNameString = ("L" +std::to_string(librationPointNr) + "_" + orbitType  + "_" + stringAccelerationMagnitude + "_" + stringAccelerationAngle1 + "_" + stringAccelerationAngle2 + "_" + stringHamiltonian + "_" + "_n_" + ".txt");
        }
    }

    tudat::input_output::writeDataMapToTextFile( stateHistory, fileNameString, directoryString );
}

std::pair< Eigen::MatrixXd, double > propagateOrbitAugmented(
        const Eigen::MatrixXd& stateVectorInclSTM, double massParameter, double currentTime,
        int direction, double initialStepSize, double maximumStepSize )
{
    // Declare variables
    Eigen::MatrixXd outputState = stateVectorInclSTM;
    double stepSize = initialStepSize;

    double minimumStepSize   = std::numeric_limits<double>::epsilon( ); // 2.22044604925031e-16
    const double relativeErrorTolerance = 100.0 * std::numeric_limits<double>::epsilon( ); // 2.22044604925031e-14

    const double absoluteErrorTolerance = 1.0e-24;

    // Create integrator to be used for propagating.
    tudat::numerical_integrators::RungeKuttaVariableStepSizeIntegrator< double, Eigen::MatrixXd > orbitIntegratorAugmented (
                tudat::numerical_integrators::RungeKuttaCoefficients::get( tudat::numerical_integrators::RungeKuttaCoefficients::rungeKuttaFehlberg78 ),
                &computeStateDerivativeAugmented, 0.0, stateVectorInclSTM, minimumStepSize, maximumStepSize, relativeErrorTolerance, absoluteErrorTolerance);


    if (direction > 0)
    {
        Eigen::MatrixXd tempState = orbitIntegratorAugmented.performIntegrationStep(stepSize);
        stepSize                  = orbitIntegratorAugmented.getNextStepSize();
        orbitIntegratorAugmented.rollbackToPreviousState();
        outputState               = orbitIntegratorAugmented.performIntegrationStep(stepSize);
        currentTime              += orbitIntegratorAugmented.getCurrentIndependentVariable();
    }
    else
    {
        Eigen::MatrixXd tempState = orbitIntegratorAugmented.performIntegrationStep(-stepSize);
        stepSize                  = orbitIntegratorAugmented.getNextStepSize();
        orbitIntegratorAugmented.rollbackToPreviousState();
        outputState               = orbitIntegratorAugmented.performIntegrationStep(stepSize);
        currentTime              += orbitIntegratorAugmented.getCurrentIndependentVariable();
    }

    // Return the value of the state and the halfPeriod time.
    return std::make_pair( outputState, currentTime );
}

std::pair< Eigen::MatrixXd, double >  propagateOrbitAugmentedToFinalCondition(
        const Eigen::MatrixXd fullInitialState, const double massParameter, const double finalTime, int direction,
        std::map< double, Eigen::VectorXd >& stateHistory, const int saveFrequency, const double initialTime )
{           
    if( saveFrequency >= 0 )
    {
        stateHistory[ initialTime ] = fullInitialState.block( 0, 0, 10, 1 );
    }

    // Perform first integration step
    std::pair< Eigen::MatrixXd, double > previousState;
    std::pair< Eigen::MatrixXd, double > currentState;
    currentState = propagateOrbitAugmented(fullInitialState, massParameter, initialTime, direction, 1.0E-5, 1.0E-5 );
    double currentTime = currentState.second;
    int stepCounter = 1;
    // Perform integration steps until end of target time of half orbital period
    for (int i = 5; i <= 13; i++)
    {

        double initialStepSize = pow(10,(static_cast<float>(-i)));
        double maximumStepSize = initialStepSize;
        //std::cout << "TEST, i is : " << maximumStepSize  << std::endl;
        //std::cout << "FINAL TIME, ORBPER is : " << finalTime  << std::endl;

        if (direction == 1)
        {

            while (currentTime <= finalTime )
            {
                // Write every nth integration step to file.
                if ( saveFrequency > 0 && ( stepCounter % saveFrequency == 0 ) )
                {
                    stateHistory[ currentTime ] = currentState.first.block( 0, 0, 10, 1 );
                }

                currentTime = currentState.second;
                previousState = currentState;
                currentState = propagateOrbitAugmented(currentState.first, massParameter, currentTime, 1, initialStepSize, maximumStepSize);

                stepCounter++;

                if (currentState.second > finalTime )
                {
                    //std::cout << "TargetTime: " << finalTime << std::endl
                    //          << "currentTime: " << finalTime << std::endl;
                    currentState = previousState;
                    currentTime = currentState.second;
                    break;
                }
            }

        }

        if (direction == -1)
        {
            while (currentTime >= finalTime )
            {
                // Write every nth integration step to file.
                if ( saveFrequency > 0 && ( stepCounter % saveFrequency == 0 ) )
                {
                    stateHistory[ currentTime ] = currentState.first.block( 0, 0, 10, 1 );
                }

                currentTime = currentState.second;
                previousState = currentState;
                currentState = propagateOrbitAugmented(currentState.first, massParameter, currentTime, -1, initialStepSize, maximumStepSize);

                stepCounter++;

                if (currentState.second < finalTime )
                {
                    currentState = previousState;
                    currentTime = currentState.second;
                    break;
                }
            }
        }

    }
    // Add final state after minimizing overshoot
    if ( saveFrequency > 0 )
    {
        stateHistory[ currentTime ] = currentState.first.block( 0, 0, 10, 1 );
    }

    return currentState;
}

std::pair< Eigen::MatrixXd, double >  propagateOrbitAugmentedToFinalConditionSmallIntervals(
        const Eigen::MatrixXd fullInitialState, const double massParameter, const double finalTime, int direction,
        std::map< double, Eigen::VectorXd >& stateHistory, const int saveFrequency, const double initialTime )
{

    // declare and initialize propagation variables
    std::pair< Eigen::MatrixXd, double > previousState;
    std::pair< Eigen::MatrixXd, double > currentState;
    double currentTime = initialTime;
    Eigen::MatrixXd stateVectorInclSTM = fullInitialState;
    Eigen::MatrixXd stateVectorOnly = fullInitialState.block(0,0,10,1);

    currentState.first = fullInitialState;
    currentState.second = currentTime;


    int stepCounter = 0;
    // Perform integration steps until end of target time of half orbital period
    for (int i = 5; i <= 13; i++)
    {

        double initialStepSize = pow(10,(static_cast<float>(-i)));
        double maximumStepSize = initialStepSize;
        //std::cout << "TEST, i is : " << maximumStepSize  << std::endl;
        //std::cout << "FINAL TIME, ORBPER is : " << finalTime  << std::endl;

        if (direction == 1)
        {

            while (currentTime <= finalTime )
            {
                // Write every nth integration step to file.
                if ( saveFrequency > 0 && ( stepCounter % saveFrequency == 0 ) )
                {
                    stateHistory[ currentTime ] = currentState.first.block( 0, 0, 10, 1 );
                }

                currentTime = currentState.second;
                previousState = currentState;
                currentState = propagateOrbitAugmented(currentState.first, massParameter, currentTime, 1, initialStepSize, maximumStepSize);

                stepCounter++;

                if (currentState.second > finalTime )
                {
                    //std::cout << "TargetTime: " << finalTime << std::endl
                    //          << "currentTime: " << finalTime << std::endl;
                    currentState = previousState;
                    currentTime = currentState.second;
                    break;
                }
            }

        }

        if (direction == -1)
        {
            while (currentTime >= finalTime )
            {
                // Write every nth integration step to file.
                if ( saveFrequency > 0 && ( stepCounter % saveFrequency == 0 ) )
                {
                    stateHistory[ currentTime ] = currentState.first.block( 0, 0, 10, 1 );
                }

                currentTime = currentState.second;
                previousState = currentState;
                currentState = propagateOrbitAugmented(currentState.first, massParameter, currentTime, -1, initialStepSize, maximumStepSize);

                stepCounter++;

                if (currentState.second < finalTime )
                {
                    currentState = previousState;
                    currentTime = currentState.second;
                    break;
                }
            }
        }

    }
    // Add final state after minimizing overshoot
    if ( saveFrequency > 0 )
    {
        stateHistory[ currentTime ] = currentState.first.block( 0, 0, 10, 1 );
    }

    return currentState;
}

std::pair< Eigen::MatrixXd, double >  propagateOrbitAugmentedToFinalThetaCondition(
        const Eigen::MatrixXd fullInitialState, const double massParameter, int direction,
        std::map< double, Eigen::VectorXd >& stateHistoryMinimized, const int saveFrequency, const double initialTime )
{
    if( saveFrequency >= 0 )
    {
        stateHistoryMinimized[ initialTime ] = fullInitialState.block( 0, 0, 10, 1 );
    }

    // compute theta of initial state w.r.t. secondary body
    double initialAngleOfOrbit = fmod(atan2( fullInitialState(1,0), fullInitialState(0,0) - (1.0 - massParameter) ) * 180.0 / tudat::mathematical_constants::PI, 360.0 );
    if (initialAngleOfOrbit < 0.0 ) {
        initialAngleOfOrbit = initialAngleOfOrbit + 360.0;
    }
    // Perform first integration step
    std::pair< Eigen::MatrixXd, double > previousState;
    std::pair< Eigen::MatrixXd, double > currentState;
    double thetaSign;
    int thetaSignChanges;
    Eigen::MatrixXd stateVectorInclSTM;
    currentState = propagateOrbitAugmented(fullInitialState, massParameter, initialTime, direction, 1.0E-5, 1.0E-5 );
    Eigen::MatrixXd stateVectorOnly = currentState.first;
    double currentTime = currentState.second;
    double currentAngleOfOrbit = fmod(atan2( stateVectorOnly(1,0), stateVectorOnly(0,0) - (1.0 - massParameter) ) *180.0 / tudat::mathematical_constants::PI, 360.0 ) ;
    if (currentAngleOfOrbit < 0.0 ) {
        currentAngleOfOrbit = currentAngleOfOrbit + 360.0;
    }

    //std::cout << "Begin angle" << initialAngleOfOrbit << std::endl;
    //std::cout << "current angle" << currentAngleOfOrbit  << std::endl;

    thetaSignChanges = 0;
    if (initialAngleOfOrbit - currentAngleOfOrbit > 0.0 ) {
        thetaSign = 1.0;
    } else {
        thetaSign = -1.0;
    }
    int stepCounter = 1;

    // Perform integration steps until end of half orbital period
    for (int i = 5; i <= 13; i++)
    {
        double initialStepSize = pow(10,(static_cast<float>(-i)));
        double maximumStepSize = initialStepSize;
        //std::cout << "Step size maximum: " << initialStepSize << std::endl;

        while (thetaSignChanges <= 1.0 )
        {
            // Write every nth integration step to file.
            if ( saveFrequency > 0 && ( stepCounter % saveFrequency == 0 ) )
            {
                stateHistoryMinimized[ currentTime ] = currentState.first.block( 0, 0, 10, 1 );
            }

                currentTime = currentState.second;
                stateVectorInclSTM = currentState.first;
                previousState = currentState;
                currentState = propagateOrbitAugmented(currentState.first, massParameter, currentTime, 1, initialStepSize, maximumStepSize);
                stateVectorOnly = currentState.first;
                currentAngleOfOrbit = fmod(std::atan2( stateVectorOnly(1,0), stateVectorOnly(0,0) - (1.0 - massParameter) ) * 180.0 / tudat::mathematical_constants::PI, 360.0);
                if (currentAngleOfOrbit < 0.0 ) {
                    currentAngleOfOrbit = currentAngleOfOrbit + 360.0;
                }


                stepCounter++;
                if ( (initialAngleOfOrbit - currentAngleOfOrbit) * thetaSign < 0.0 )
                {

                    thetaSignChanges++;
                    thetaSign = thetaSign*-1.0;
                }




                if (thetaSignChanges > 1.0 )
                {
                    currentState = previousState;
                    currentTime = currentState.second;
                    thetaSignChanges--;
                    thetaSign = thetaSign*-1.0;
                    break;
                }

            }
    }

    // Add final state after minimizing overshoot
    if ( saveFrequency > 0 )
    {
        stateHistoryMinimized[ currentTime ] = currentState.first.block( 0, 0, 10, 1 );
    }
    stateVectorInclSTM = currentState.first;
    //std::cout << "||delta theta|| = " << abs(initialAngleOfOrbit - currentAngleOfOrbit) << ", at end of iterative procedure" << std::endl;
    return currentState;

}

std::pair< Eigen::MatrixXd, double >  propagateOrbitAugmentedToFullRevolutionCondition(
        const Eigen::MatrixXd fullInitialState, const int librationPointNr, const double massParameter, const double finalAngle, int direction,
        std::map< double, Eigen::VectorXd >& stateHistoryMinimized, const int saveFrequency, const double initialTime )
{
    // compute theta of initial state w.r.t. secondary body (L1, L2) or primary body (L3,4,5)
    double currentAngleOfOrbit;
    if (librationPointNr < 3)
    {
        currentAngleOfOrbit = atan2( fullInitialState(1,0), ( fullInitialState(0,0) - (1.0 - massParameter) ) ) * 180.0 / tudat::mathematical_constants::PI;
        if (currentAngleOfOrbit < 0.0 ) {
            currentAngleOfOrbit = currentAngleOfOrbit + 360.0;
        }
    } else
    {
        currentAngleOfOrbit = atan2( fullInitialState(1,0), fullInitialState(0,0) - ( - massParameter) ) * 180.0 / tudat::mathematical_constants::PI;
        if (currentAngleOfOrbit < 0.0 ) {
            currentAngleOfOrbit = currentAngleOfOrbit + 360.0;
        }
    }

    // Determine final angle in [0,360 domain]
    double targetAngle = finalAngle;
    if (targetAngle < 0.0 ) {
        targetAngle = targetAngle + 360.0;
    }

    // Determine thetaSign
    double thetaSign;
    if (( targetAngle - currentAngleOfOrbit ) > 0.0 )
    {
        thetaSign = 1.0;
    } else
    {
        thetaSign = -1.0;
    }


    // declare and initialize propagation variables
    std::pair< Eigen::MatrixXd, double > previousState;
    std::pair< Eigen::MatrixXd, double > currentState;
    double currentTime = initialTime;
    Eigen::MatrixXd stateVectorInclSTM = fullInitialState;
    Eigen::MatrixXd stateVectorOnly = fullInitialState.block(0,0,10,1);

    currentState.first = fullInitialState;
    currentState.second = currentTime;

//    std::cout << "currentState first: \n" << currentState.first << std::endl
//              << "currentState second: " << currentState.second << std::endl;

    int stepCounter = 0;

    for (int i = 5; i <= 13; i++)
    {
        double initialStepSize = pow(10,(static_cast<float>(-i)));
        double maximumStepSize = initialStepSize;

        // Recompute current Angle for enter loop again after rolling back to previous state
        if (librationPointNr < 3)
        {
            currentAngleOfOrbit = fmod(std::atan2( stateVectorOnly(1,0), stateVectorOnly(0,0) - (1.0 - massParameter) ) * 180.0 / tudat::mathematical_constants::PI, 360.0);

        }else
        {
            currentAngleOfOrbit = fmod(std::atan2( stateVectorOnly(1,0), stateVectorOnly(0,0) - ( - massParameter) ) * 180.0 / tudat::mathematical_constants::PI, 360.0);

        }
        if (currentAngleOfOrbit < 0.0 ) {
            currentAngleOfOrbit = currentAngleOfOrbit + 360.0;
        }


        while ( ( targetAngle - currentAngleOfOrbit ) * thetaSign > 0.0  )
        {
            //std::cout << "stepCounter: " << stepCounter << std::endl;
            // Write every nth integration step to file.
            if ( saveFrequency > 0 && ( stepCounter % saveFrequency == 0 ) )
            {
                if (stepCounter == 0)
                {
                   stateHistoryMinimized[ initialTime ] = fullInitialState.block(0,0,10,1);
                }else
                {
                   stateHistoryMinimized[ currentTime ] = currentState.first.block( 0, 0, 10, 1 );
                }

            }

            currentTime = currentState.second;
            stateVectorInclSTM = currentState.first;
            previousState = currentState;
            if (direction == 1)
            {
               currentState = propagateOrbitAugmented(currentState.first, massParameter, currentTime, 1, initialStepSize, maximumStepSize);

            }else
            {

               currentState = propagateOrbitAugmented(currentState.first, massParameter, currentTime, -1, initialStepSize, maximumStepSize);
            }

            stateVectorInclSTM = currentState.first;
            stateVectorOnly = stateVectorInclSTM.block(0,0,10,1);

            if (librationPointNr < 3)
            {
                currentAngleOfOrbit = fmod(std::atan2( stateVectorOnly(1,0), stateVectorOnly(0,0) - (1.0 - massParameter) ) * 180.0 / tudat::mathematical_constants::PI, 360.0);

            }else
            {
                currentAngleOfOrbit = fmod(std::atan2( stateVectorOnly(1,0), stateVectorOnly(0,0) - ( - massParameter) ) * 180.0 / tudat::mathematical_constants::PI, 360.0);

            }
            if (currentAngleOfOrbit < 0.0 ) {
                currentAngleOfOrbit = currentAngleOfOrbit + 360.0;
            }

            stepCounter++;





            if ( ( targetAngle - currentAngleOfOrbit ) * thetaSign < 0.0 )
            {
                currentState = previousState;
                currentTime = currentState.second;
                stateVectorOnly = currentState.first;
                break;
            }
        }
    }

    // Add final state after minimizing overshoot
    if ( saveFrequency > 0 )
    {
        stateHistoryMinimized[ currentTime ] = currentState.first.block( 0, 0, 10, 1 );
    }
    stateVectorInclSTM = currentState.first;
    std::cout << "||delta theta|| = " << abs(targetAngle - currentAngleOfOrbit) << ", at end of iterative procedure" << std::endl;
    return currentState;

}

std::pair< Eigen::MatrixXd, double >  propagateOrbitAugmentedToFullRevolutionOrFinalTime(
        const Eigen::MatrixXd fullInitialState, const int librationPointNr, const double massParameter, const double finalAngle, double finalTime,
        int direction, int& thetaSignChanges, double& thetaSign, bool& fullRevolution, std::map< double, Eigen::VectorXd >& stateHistoryMinimized, const int saveFrequency, const double initialTime )
{
    // save first state
    if( saveFrequency >= 0 )
    {
        stateHistoryMinimized[ initialTime ] = fullInitialState.block( 0, 0, 10, 1 );
    }

    // compute theta of initial state w.r.t. secondary body (L1, L2) or primary body (L3,4,5)
    double currentAngleOfOrbit;
    if (librationPointNr < 3)
    {
        currentAngleOfOrbit = atan2( fullInitialState(1,0), fullInitialState(0,0) - (1.0 - massParameter) ) * 180.0 / tudat::mathematical_constants::PI;
        if (currentAngleOfOrbit < 0.0 and librationPointNr == 1 ) {
            currentAngleOfOrbit = currentAngleOfOrbit + 360.0;
        }
    } else
    {
        currentAngleOfOrbit = atan2( fullInitialState(1,0), fullInitialState(0,0) - ( - massParameter) ) * 180.0 / tudat::mathematical_constants::PI;
        if (currentAngleOfOrbit < 0.0 ) {
            currentAngleOfOrbit = currentAngleOfOrbit + 360.0;
        }
    }

    // Convert reference angle in [0,360 domain]
    double referenceAngle  = finalAngle * 180.0 / tudat::mathematical_constants::PI;
    if (referenceAngle < 0.0 and librationPointNr != 2) {
        referenceAngle = referenceAngle + 360.0;
    }

    // Determine sign change after first propagation step if thetasign absolute value lower than 0.5
    // Perform first integration step
    std::pair< Eigen::MatrixXd, double > previousState;
    std::pair< Eigen::MatrixXd, double > currentState;
    Eigen::MatrixXd stateVectorInclSTM;
    Eigen::MatrixXd stateVectorOnly;
    double currentTime = initialTime;

    // perform first propagation step
    currentState = propagateOrbitAugmented(fullInitialState, massParameter, initialTime, direction, 1.0E-5, 1.0E-5 );
    stateVectorInclSTM = currentState.first;
    stateVectorOnly = stateVectorInclSTM.block(0,0,10,1);
    currentTime = currentState.second;

    if (std::abs(thetaSign) < 0.5)
    {
        // compute theta of initial state w.r.t. secondary body (L1, L2) or primary body (L3,4,5)
        if (librationPointNr < 3)
        {
            currentAngleOfOrbit = atan2( stateVectorOnly(1,0), stateVectorOnly(0,0) - (1.0 - massParameter) ) * 180.0 / tudat::mathematical_constants::PI;
            if (currentAngleOfOrbit < 0.0 and librationPointNr == 1) {
                currentAngleOfOrbit = currentAngleOfOrbit + 360.0;
            }
        } else
        {
            currentAngleOfOrbit = atan2( stateVectorOnly(1,0), stateVectorOnly(0,0) - ( - massParameter) ) * 180.0 / tudat::mathematical_constants::PI;
            if (currentAngleOfOrbit < 0.0 ) {
                currentAngleOfOrbit = currentAngleOfOrbit + 360.0;
            }
        }

//        std::cout << "\n referenceAngle: " << referenceAngle << std::endl
//                  << "currentAngle: " << currentAngleOfOrbit <<std::endl
//                  << "current - referenceAngle: " << currentAngleOfOrbit - referenceAngle << std::endl
//                  << "initialState: \n" << fullInitialState.block(0,0,10,1) << std::endl
//                  << "firstPropState: \n" << stateVectorOnly.block(0,0,10,1) << std::endl;

        if (currentAngleOfOrbit - referenceAngle > 0.0 ) {
            thetaSign = 1.0;
        } else {
            thetaSign = -1.0;
        }

 //       std::cout << "thetaSign: " << thetaSign << std::endl;

    }

    // check if the angle condition is met!
    if ( ( currentAngleOfOrbit - referenceAngle) * thetaSign < 0.0 )
    {
        //std::cout << "The sign of angle has changed "<< std::endl;
        thetaSignChanges++;
        thetaSign = thetaSign*-1.0;
    }

//    std::cout << "\nentering the propagation loop: " << std::endl
//              << "currentTime: " << currentTime << std::endl
//              << "finalTime: " << finalTime << std::endl
//              << "thetaSign: " << thetaSign << std::endl
//              << "thetaSignChanges: " << thetaSignChanges << std::endl;


    int stepCounter = 1;
    for (int i = 5; i <= 13; i++)
    {
        double initialStepSize = pow(10,(static_cast<float>(-i)));
        double maximumStepSize = initialStepSize;
        //std::cout << "Step size maximum: " << initialStepSize << std::endl;

        while (thetaSignChanges < 2 and currentTime <= finalTime )
        {
            // Write every nth integration step to file.
            if ( saveFrequency > 0 && ( stepCounter % saveFrequency == 0 ) )
            {
//                std::cout << "Test Angle progression: "<< atan2( stateVectorOnly(1,0), stateVectorOnly(0,0) - ( - massParameter) ) * 180.0 / tudat::mathematical_constants::PI << std::endl;
//                std::cout << "Difference curr - ref: "<<  (atan2( stateVectorOnly(1,0), stateVectorOnly(0,0) - ( - massParameter) ) * 180.0 / tudat::mathematical_constants::PI  - referenceAngle )<< std::endl;
//                std::cout << "current - ref X corr: " << currentState.first(0,0) - previousState.first(0,0) << std::endl;
//                std::cout << "Condition: "<< thetaSign * (atan2( stateVectorOnly(1,0), stateVectorOnly(0,0) - ( - massParameter) ) * 180.0 / tudat::mathematical_constants::PI  - referenceAngle )<< std::endl;
//                std::cout << "signChanges: "<< thetaSignChanges << std::endl;


                stateHistoryMinimized[ currentTime ] = currentState.first.block( 0, 0, 10, 1 );

            }
                // propagate orbit and compute new angle
                currentTime = currentState.second;
                stateVectorInclSTM = currentState.first;
                previousState = currentState;
                currentState = propagateOrbitAugmented(currentState.first, massParameter, currentTime, 1, initialStepSize, maximumStepSize);
                stateVectorInclSTM = currentState.first;
                stateVectorOnly = stateVectorInclSTM.block(0,0,10,1);

                if (librationPointNr < 3)
                {
                    currentAngleOfOrbit = atan2( stateVectorOnly(1,0), stateVectorOnly(0,0) - (1.0 - massParameter) ) * 180.0 / tudat::mathematical_constants::PI;
                    if (currentAngleOfOrbit < 0.0 and librationPointNr == 1 )
                    {
                        currentAngleOfOrbit = currentAngleOfOrbit + 360.0;
                    }
                } else
                {
                    currentAngleOfOrbit = atan2( stateVectorOnly(1,0), stateVectorOnly(0,0) - ( - massParameter) ) * 180.0 / tudat::mathematical_constants::PI;
                    if (currentAngleOfOrbit < 0.0 ) {
                        currentAngleOfOrbit = currentAngleOfOrbit + 360.0;
                    }
                }

                // check if if a sign change occured, break and roll back if signChanges greater than 1
                stepCounter++;
                if ( ( currentAngleOfOrbit - referenceAngle) * thetaSign < 0.0 )
                {
//                    std::cout << "The sign of angle has changed "<< std::endl;
//                    std::cout << "i: " << i << std::endl;

                    thetaSignChanges++;
                    thetaSign = thetaSign*-1.0;
                }

                if (thetaSignChanges > 1 )
                {
                    currentState = previousState;
                    currentTime = currentState.second;

                    // ensure that the fullRevolution boolean is set to true if overshoot procedure keeps getting activated
                    // by a change in sign
                    if ( ( i < 13 && librationPointNr < 3) or ( i < 10 && librationPointNr > 2) )
                    {
                        thetaSignChanges--;
                        thetaSign = thetaSign*-1.0;
                    }


                    stateVectorInclSTM = currentState.first;
                    stateVectorOnly = stateVectorInclSTM.block(0,0,10,1);

                    //std::cout << "2 thetaSignChanges Happened "<< std::endl;

                    if (librationPointNr < 3)
                    {
                        currentAngleOfOrbit = atan2( stateVectorOnly(1,0), stateVectorOnly(0,0) - (1.0 - massParameter) ) * 180.0 / tudat::mathematical_constants::PI;
                        if (currentAngleOfOrbit < 0.0  and librationPointNr == 1)
                        {
                            currentAngleOfOrbit = currentAngleOfOrbit + 360.0;
                        }
                    } else
                    {
                        currentAngleOfOrbit = atan2( stateVectorOnly(1,0), stateVectorOnly(0,0) - ( - massParameter) ) * 180.0 / tudat::mathematical_constants::PI;

                        if (currentAngleOfOrbit < 0.0 ) {
                            currentAngleOfOrbit = currentAngleOfOrbit + 360.0;
                        }
                    }

                    break;
                }

                // check if final time is exceeded, break and roll back
                if (currentState.second > finalTime )
                {
                    currentState = previousState;
                    currentTime = currentState.second;
                    stateVectorInclSTM = currentState.first;
                    stateVectorOnly = stateVectorInclSTM.block(0,0,10,1);

                    //std::cout << "final time exceeded "<< std::endl;


                    if (librationPointNr < 3)
                    {
                        currentAngleOfOrbit = atan2( stateVectorOnly(1,0), stateVectorOnly(0,0) - (1.0 - massParameter) ) * 180.0 / tudat::mathematical_constants::PI;
                        if (currentAngleOfOrbit < 0.0 and librationPointNr == 1 ) {
                            currentAngleOfOrbit = currentAngleOfOrbit + 360.0;
                        }
                    } else
                    {
                        currentAngleOfOrbit = atan2( stateVectorOnly(1,0), stateVectorOnly(0,0) - ( - massParameter) ) * 180.0 / tudat::mathematical_constants::PI;
                        if (currentAngleOfOrbit < 0.0 ) {
                            currentAngleOfOrbit = currentAngleOfOrbit + 360.0;
                        }
                    }


                    break;
                }
        }


    }

    // Add final state after minimizing overshoot
    if ( saveFrequency > 0 )
    {
        stateHistoryMinimized[ currentTime ] = currentState.first.block( 0, 0, 10, 1 );
    }

    if (thetaSignChanges > 1)
    {
        fullRevolution = true;
    }

    return currentState;

}

std::pair< Eigen::MatrixXd, double >  propagateOrbitAugmentedToFinalSpatialCondition(
        const Eigen::MatrixXd fullInitialState, const double massParameter, const int stateIndex, int direction,
        std::map< double, Eigen::VectorXd >& stateHistoryMinimized, const int saveFrequency, const double initialTime )
{
    if( saveFrequency >= 0 )
    {
        stateHistoryMinimized[ initialTime ] = fullInitialState.block( 0, 0, 10, 1 );
    }

    // compute theta of initial spatial coordinate for reference
    double initialStateCoordinate = fullInitialState(stateIndex, 0);

    // Perform first integration step
    std::pair< Eigen::MatrixXd, double > previousState;
    std::pair< Eigen::MatrixXd, double > currentState;
    double coordinateSign;
    double coordinateSignChanges;
    Eigen::MatrixXd stateVectorInclSTM;
    currentState = propagateOrbitAugmented(fullInitialState, massParameter, initialTime, direction, 1.0E-5, 1.0E-5 );
    Eigen::MatrixXd stateVectorOnly = currentState.first;
    double currentTime = currentState.second;
    double currentStateCoordinate = stateVectorOnly(stateIndex, 0);




    coordinateSignChanges = 0;
    if (currentStateCoordinate - initialStateCoordinate > 0.0 ) {
        coordinateSign = 1.0;
    } else {
        coordinateSign = -1.0;
    }
    int stepCounter = 1;

    // Perform integration steps until end of half orbital period
    for (int i = 5; i <= 12; i++)
    {
        double initialStepSize = pow(10,(static_cast<float>(-i)));
        double maximumStepSize = initialStepSize;

        while (coordinateSignChanges <= 0.0 )
        {
            // Write every nth integration step to file.
            if ( saveFrequency > 0 && ( stepCounter % saveFrequency == 0 ) )
            {
                stateHistoryMinimized[ currentTime ] = currentState.first.block( 0, 0, 10, 1 );

                //std::cout << "current - initial: " << currentStateCoordinate - initialStateCoordinate << std::endl;
            }

                currentTime = currentState.second;
                stateVectorInclSTM = currentState.first;
                previousState = currentState;
                currentState = propagateOrbitAugmented(currentState.first, massParameter, currentTime, 1, initialStepSize, maximumStepSize);
                stateVectorOnly = currentState.first;
                currentStateCoordinate = stateVectorOnly(stateIndex, 0);

                stepCounter++;

                if ( (currentStateCoordinate - initialStateCoordinate ) * coordinateSign < 0.0 )
                {

                    coordinateSignChanges = coordinateSignChanges + 1.0;
                    coordinateSign = coordinateSign*-1.0;
                }


                if ( coordinateSignChanges > 0.0 )
                {
                    currentState = previousState;
                    currentTime = currentState.second;
                    coordinateSignChanges = coordinateSignChanges - 1.0;
                    coordinateSign = coordinateSign*-1.0;
                    break;
                }

            }
    }

    // Add final state after minimizing overshoot
    if ( saveFrequency > 0 )
    {
        stateHistoryMinimized[ currentTime ] = currentState.first.block( 0, 0, 10, 1 );
    }
    stateVectorInclSTM = currentState.first;
    std::cout << "||delta coordinate|| = " << abs( initialStateCoordinate - currentStateCoordinate ) << ", at end of iterative procedure" << std::endl;
    return currentState;

}

std::pair< Eigen::MatrixXd, double >  propagateOrbitAugmentedWithStateTransitionMatrixToFinalCondition(
        const Eigen::MatrixXd fullInitialState, const double massParameter, const double finalTime, int direction,
        std::map< double, Eigen::MatrixXd >& stateTransitionMatrixHistory, const int saveFrequency, const double initialTime )
{
    if( saveFrequency >= 0 )
    {
        stateTransitionMatrixHistory[ initialTime ] = fullInitialState;
    }

    // Perform first integration step
    std::pair< Eigen::MatrixXd, double > previousState;
    std::pair< Eigen::MatrixXd, double > currentState;
    currentState = propagateOrbitAugmented( fullInitialState, massParameter, initialTime, direction, 1.0E-5, 1.0E-5 );
    double currentTime = currentState.second;

    int stepCounter = 1;
    // Perform integration steps until end of half orbital period
    for (int i = 5; i <= 12; i++)
    {
        double initialStepSize = pow(10,(static_cast<float>(-i)));
        double maximumStepSize = initialStepSize;

        while (currentTime <= finalTime )
        {
            // Write every nth integration step to file.
            if ( saveFrequency > 0 && ( stepCounter % saveFrequency == 0 ) && i == 5 )
            {
                stateTransitionMatrixHistory[ currentTime ] = currentState.first;
            }

            currentTime = currentState.second;
            previousState = currentState;
            currentState = propagateOrbitAugmented(currentState.first, massParameter, currentTime, 1, initialStepSize, maximumStepSize);
            stepCounter++;

            if (currentState.second > finalTime )
            {
                currentState = previousState;
                currentTime = currentState.second;
                break;
            }

            if (currentState.second < finalTime )
            {
                currentState = previousState;
                currentTime = currentState.second;
                break;
            }
        }
    }

    return currentState;
}
