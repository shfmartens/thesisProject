#include "Tudat/Mathematics/NumericalIntegrators/rungeKuttaVariableStepSizeIntegrator.h"
#include "Tudat/Mathematics/NumericalIntegrators/rungeKuttaCoefficients.h"
#include "Tudat/Mathematics/RootFinders/newtonRaphson.h"
#include "Tudat/Astrodynamics/Gravitation/jacobiEnergy.h"
#include "Tudat/Astrodynamics/BasicAstrodynamics/celestialBodyConstants.h"
#include "Tudat/Astrodynamics/Gravitation/librationPoint.h"
#include "Tudat/InputOutput/basicInputOutput.h"

#include "propagateOrbitAugmented.h"
#include "stateDerivativeModelAugmented.h"

Eigen::MatrixXd getFullInitialStateAugmented( const Eigen::VectorXd& initialState )
{
    Eigen::MatrixXd fullInitialState = Eigen::MatrixXd::Zero( 10, 11 );
    fullInitialState.block( 0, 0, 10, 1 ) = initialState;
    fullInitialState.block( 0, 1, 10, 10 ).setIdentity( );
    return fullInitialState;
}

void writeStateHistoryToFileAugmented(
        const std::map< double, Eigen::VectorXd >& stateHistory, const double accelerationMagnitude, const double accelerationAngle,
        const int orbitId, const int librationPointNr,
        const int saveEveryNthIntegrationStep, const bool completeInitialConditionsHaloFamily )
{
    std::string fileNameString;
    std::string directoryString = "../data/raw/orbits/";
    // Prepare output file
    if (saveEveryNthIntegrationStep != 1000)
    {
        if (completeInitialConditionsHaloFamily == false)
        {
            fileNameString = ("L" + std::to_string(librationPointNr) + "_" + std::to_string(accelerationMagnitude) + "_" + std::to_string(accelerationAngle) + "_" + std::to_string(orbitId) + "_" + std::to_string(saveEveryNthIntegrationStep) + ".txt");
        }
        else
        {
            fileNameString = ("L" + std::to_string(librationPointNr) + "_" + std::to_string(accelerationMagnitude) + "_" + std::to_string(accelerationAngle) + "_n_" + std::to_string(orbitId) + "_" + std::to_string(saveEveryNthIntegrationStep) + ".txt");
        }
    }
    else
    {
        if (completeInitialConditionsHaloFamily == false)
        {
            fileNameString = ("L" + std::to_string(librationPointNr) + "_" + std::to_string(accelerationMagnitude) + "_" + std::to_string(accelerationAngle) + "_" + std::to_string(orbitId) + ".txt");
        }
        else
        {
            fileNameString = ("L" + std::to_string(librationPointNr) + "_" + std::to_string(accelerationMagnitude) + "_" + std::to_string(accelerationAngle) + "_n_" + std::to_string(orbitId) + ".txt");
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
    // Perform integration steps until end of half orbital period
    for (int i = 5; i <= 12; i++)
    {

        double initialStepSize = pow(10,(static_cast<float>(-i)));
        double maximumStepSize = initialStepSize;

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
                currentState = previousState;
                currentTime = currentState.second;
                break;
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
        }
    }

    return currentState;
}
