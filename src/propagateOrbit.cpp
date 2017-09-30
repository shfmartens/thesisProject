#include "Tudat/Mathematics/NumericalIntegrators/rungeKuttaVariableStepSizeIntegrator.h"
#include "Tudat/Mathematics/NumericalIntegrators/rungeKuttaCoefficients.h"
#include "Tudat/Mathematics/RootFinders/newtonRaphson.h"
#include "Tudat/Astrodynamics/Gravitation/jacobiEnergy.h"
#include "Tudat/Astrodynamics/BasicAstrodynamics/celestialBodyConstants.h"
#include "Tudat/Astrodynamics/Gravitation/librationPoint.h"

#include "propagateOrbit.h"
#include "stateDerivativeModel.h"



std::pair< Eigen::MatrixXd, double > propagateOrbit( Eigen::MatrixXd stateVectorInclSTM, double massParameter, double currentTime,
                                int direction, double initialStepSize = 1.0e-5, double maximumStepSize = 1.0e-4 )
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
        const Eigen::MatrixXd fullInitialState, const double massParameter, const double finalTime, int direction, const double initialTime )
{
    // Perform first integration step
    std::pair< Eigen::MatrixXd, double > previousHalfPeriodState;
    std::pair< Eigen::MatrixXd, double > halfPeriodState;
    halfPeriodState = propagateOrbit( fullInitialState, massParameter, initialTime, direction );
    Eigen::MatrixXd stateVectorInclSTM  = halfPeriodState.first;

    double currentTime                  = halfPeriodState.second;

    // Perform integration steps until end of half orbital period
    for (int i = 5; i <= 12; i++) {

        double initialStepSize = pow(10,(static_cast<float>(-i)));
        double maximumStepSize = pow(10,(static_cast<float>(-i) + 1.0));

        while (currentTime <= finalTime ) {
            stateVectorInclSTM      = halfPeriodState.first;
            currentTime             = halfPeriodState.second;
            previousHalfPeriodState = halfPeriodState;
            halfPeriodState         = propagateOrbit(stateVectorInclSTM, massParameter, currentTime, 1, initialStepSize, maximumStepSize);

            if (halfPeriodState.second > finalTime )
            {
                halfPeriodState = previousHalfPeriodState;
                stateVectorInclSTM      = halfPeriodState.first;
                currentTime             = halfPeriodState.second;
                break;
            }
        }
    }

    return halfPeriodState;
}
