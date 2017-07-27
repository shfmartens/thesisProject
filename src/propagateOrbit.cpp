#include "Tudat/Mathematics/NumericalIntegrators/rungeKuttaVariableStepSizeIntegrator.h"
#include "Tudat/Mathematics/NumericalIntegrators/rungeKuttaCoefficients.h"
#include "Tudat/Mathematics/RootFinders/newtonRaphson.h"
#include "Tudat/Astrodynamics/Gravitation/jacobiEnergy.h"
#include "Tudat/Astrodynamics/BasicAstrodynamics/celestialBodyConstants.h"
#include "Tudat/Astrodynamics/Gravitation/librationPoint.h"

#include "propagateOrbit.h"
#include "stateDerivativeModel.h"



Eigen::VectorXd propagateOrbit( Eigen::VectorXd stateVectorInclSTM, double massParameter, double currentTime,
                                int direction, double initialStepSize = 1.0e-4, double maximumStepSize = 1.0e-3 )
{
    // Declare variables
    Eigen::VectorXd outputVector(43);
    Eigen::VectorXd outputState = stateVectorInclSTM;
    double stepSize = initialStepSize;

    const double minimumStepSize        = 1.0e-12;
    const double relativeErrorTolerance = 1.0e-13;
    const double absoluteErrorTolerance = 1.0e-24;

    // Create integrator to be used for propagating.
    tudat::numerical_integrators::RungeKuttaVariableStepSizeIntegratorXd orbitIntegrator ( tudat::numerical_integrators::RungeKuttaCoefficients::get( tudat::numerical_integrators::RungeKuttaCoefficients::rungeKuttaFehlberg78 ), &computeStateDerivative, 0.0, stateVectorInclSTM, minimumStepSize, maximumStepSize, relativeErrorTolerance, absoluteErrorTolerance);

    if (direction > 0) {
        Eigen::VectorXd tempState = orbitIntegrator.performIntegrationStep(stepSize);
        stepSize                  = orbitIntegrator.getNextStepSize();
        orbitIntegrator.rollbackToPreviousState();
        outputState               = orbitIntegrator.performIntegrationStep(stepSize);
        currentTime              += orbitIntegrator.getCurrentIndependentVariable();
    }
    else {
        Eigen::VectorXd tempState = orbitIntegrator.performIntegrationStep(-stepSize);
        stepSize                  = orbitIntegrator.getNextStepSize();
        orbitIntegrator.rollbackToPreviousState();
        outputState               = orbitIntegrator.performIntegrationStep(stepSize);
        currentTime              += orbitIntegrator.getCurrentIndependentVariable();
    }

    // Return the value of the state and the halfPeriod time.
    outputVector.segment(0, 42) = outputState;
    outputVector(42)            = currentTime;

    return outputVector;
}
