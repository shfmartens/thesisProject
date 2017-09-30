#include "Tudat/Mathematics/NumericalIntegrators/rungeKuttaVariableStepSizeIntegrator.h"
#include "Tudat/Mathematics/NumericalIntegrators/rungeKuttaCoefficients.h"
#include "Tudat/Mathematics/RootFinders/newtonRaphson.h"
#include "Tudat/Astrodynamics/Gravitation/jacobiEnergy.h"
#include "Tudat/Astrodynamics/BasicAstrodynamics/celestialBodyConstants.h"
#include "Tudat/Astrodynamics/Gravitation/librationPoint.h"
#include "Tudat/InputOutput/basicInputOutput.h"

#include "propagateOrbit.h"
#include "stateDerivativeModel.h"

Eigen::MatrixXd getFullInitialState( const Eigen::Vector6d& initialState )
{
    Eigen::MatrixXd fullInitialState = Eigen::MatrixXd::Zero( 6, 7 );
    fullInitialState.block( 0, 0, 6, 1 ) = initialState;
    fullInitialState.block( 0, 1, 6, 6 ).setIdentity( );
    return fullInitialState;
}

void writeStateHistoryToFile(
        const std::map< double, Eigen::Vector6d >& stateHistory,
        const int orbitId, const std::string orbitType, const int librationPointNr,
        const int saveEveryNthIntegrationStep, const bool completeInitialConditionsHaloFamily )
{
    std::string fileNameString;
    std::string directoryString = "../data/raw/orbits3/";
    // Prepare output file
    if (saveEveryNthIntegrationStep != 1000)
    {
        if (completeInitialConditionsHaloFamily == false)
        {
            fileNameString = ("L" + std::to_string(librationPointNr) + "_" + orbitType + "_" + std::to_string(orbitId) + "_" + std::to_string(saveEveryNthIntegrationStep) + ".txt");
        }
        else
        {
            fileNameString = ("L" + std::to_string(librationPointNr) + "_" + orbitType + "_n_" + std::to_string(orbitId) + "_" + std::to_string(saveEveryNthIntegrationStep) + ".txt");
        }
    }
    else
    {
        if (completeInitialConditionsHaloFamily == false)
        {
            fileNameString = ("L" + std::to_string(librationPointNr) + "_" + orbitType + "_" + std::to_string(orbitId) + ".txt");
        }
        else
        {
            fileNameString = ("L" + std::to_string(librationPointNr) + "_" + orbitType + "_n_" + std::to_string(orbitId) + ".txt");
        }
    }

    tudat::input_output::writeDataMapToTextFile( stateHistory, fileNameString, directoryString );
}

std::pair< Eigen::MatrixXd, double > propagateOrbit(
        Eigen::MatrixXd& stateVectorInclSTM, double massParameter, double currentTime,
        int direction, double initialStepSize, double maximumStepSize )
{
    // Declare variables
    Eigen::MatrixXd outputState = stateVectorInclSTM;
    double stepSize = initialStepSize;

    double minimumStepSize   = std::numeric_limits<double>::epsilon( ); // 2.22044604925031e-16
    const double relativeErrorTolerance = 100.0 * std::numeric_limits<double>::epsilon( ); // 2.22044604925031e-14
    const double absoluteErrorTolerance = 1.0e-24;

    // Create integrator to be used for propagating.
    tudat::numerical_integrators::RungeKuttaVariableStepSizeIntegrator< double, Eigen::MatrixXd > orbitIntegrator (
                tudat::numerical_integrators::RungeKuttaCoefficients::get( tudat::numerical_integrators::RungeKuttaCoefficients::rungeKuttaFehlberg78 ),
                &computeStateDerivative, 0.0, stateVectorInclSTM, minimumStepSize, maximumStepSize, relativeErrorTolerance, absoluteErrorTolerance);

    if (direction > 0)
    {
        Eigen::MatrixXd tempState = orbitIntegrator.performIntegrationStep(stepSize);
        stepSize                  = orbitIntegrator.getNextStepSize();
        orbitIntegrator.rollbackToPreviousState();
        outputState               = orbitIntegrator.performIntegrationStep(stepSize);
        currentTime              += orbitIntegrator.getCurrentIndependentVariable();
    }
    else
    {
        Eigen::MatrixXd tempState = orbitIntegrator.performIntegrationStep(-stepSize);
        stepSize                  = orbitIntegrator.getNextStepSize();
        orbitIntegrator.rollbackToPreviousState();
        outputState               = orbitIntegrator.performIntegrationStep(stepSize);
        currentTime              += orbitIntegrator.getCurrentIndependentVariable();
    }

    // Return the value of the state and the halfPeriod time.
    return std::make_pair( outputState, currentTime );
}

std::pair< Eigen::MatrixXd, double >  propagateOrbitToFinalCondition(
        const Eigen::MatrixXd fullInitialState, const double massParameter, const double finalTime, int direction,
        std::map< double, Eigen::Vector6d >& stateHistory, const int saveFrequency, const double initialTime )
{
    if( saveFrequency >= 0 )
    {
        stateHistory[ initialTime ] = fullInitialState.block( 0, 0, 6, 1 );
    }

    // Perform first integration step
    std::pair< Eigen::MatrixXd, double > previousState;
    std::pair< Eigen::MatrixXd, double > currentState;
    currentState = propagateOrbit( fullInitialState, massParameter, initialTime, direction );
    double currentTime = currentState.second;

    int stepCounter = 1;
    // Perform integration steps until end of half orbital period
    for (int i = 5; i <= 12; i++)
    {

        double initialStepSize = pow(10,(static_cast<float>(-i)));
        double maximumStepSize = pow(10,(static_cast<float>(-i) + 1.0));

        while (currentTime <= finalTime )
        {
            // Write every nth integration step to file.
            if ( saveFrequency > 0 && ( stepCounter % saveFrequency == 0 ) )
            {
                stateHistory[ currentTime ] = currentState.first.block( 0, 0, 6, 1 );
            }

            currentTime = currentState.second;
            previousState = currentState;
            currentState = propagateOrbit(currentState.first, massParameter, currentTime, 1, initialStepSize, maximumStepSize);

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
