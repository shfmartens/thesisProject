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
#include "propagateMassVaryingOrbitAugmented.h"
#include "stateDerivativeModelAugmented.h"
#include "stateDerivativeModelAugmentedVaryingMass.h"


std::pair< Eigen::MatrixXd, double > propagateMassVaryingOrbitAugmented(
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
    tudat::numerical_integrators::RungeKuttaVariableStepSizeIntegrator< double, Eigen::MatrixXd > orbitMassVaryingIntegratorAugmented (
                tudat::numerical_integrators::RungeKuttaCoefficients::get( tudat::numerical_integrators::RungeKuttaCoefficients::rungeKuttaFehlberg78 ),
                &computeStateDerivativeAugmentedVaryingMass, 0.0, stateVectorInclSTM, minimumStepSize, maximumStepSize, relativeErrorTolerance, absoluteErrorTolerance);


    if (direction > 0)
    {
        Eigen::MatrixXd tempState = orbitMassVaryingIntegratorAugmented.performIntegrationStep(stepSize);
        stepSize                  = orbitMassVaryingIntegratorAugmented.getNextStepSize();
        orbitMassVaryingIntegratorAugmented.rollbackToPreviousState();
        outputState               = orbitMassVaryingIntegratorAugmented.performIntegrationStep(stepSize);
        currentTime              += orbitMassVaryingIntegratorAugmented.getCurrentIndependentVariable();
    }
    else
    {
        Eigen::MatrixXd tempState = orbitMassVaryingIntegratorAugmented.performIntegrationStep(-stepSize);
        stepSize                  = orbitMassVaryingIntegratorAugmented.getNextStepSize();
        orbitMassVaryingIntegratorAugmented.rollbackToPreviousState();
        outputState               = orbitMassVaryingIntegratorAugmented.performIntegrationStep(stepSize);
        currentTime              += orbitMassVaryingIntegratorAugmented.getCurrentIndependentVariable();
    }

    // Return the value of the state and the halfPeriod time.
    return std::make_pair( outputState, currentTime );
}

std::pair< Eigen::MatrixXd, double >  propagateMassVaryingOrbitAugmentedToFinalCondition(
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
    currentState = propagateMassVaryingOrbitAugmented(fullInitialState, massParameter, initialTime, direction, 1.0E-5, 1.0E-5 );
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
                currentState = propagateMassVaryingOrbitAugmented(currentState.first, massParameter, currentTime, 1, initialStepSize, maximumStepSize);

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
