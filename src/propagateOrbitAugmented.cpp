#include "Tudat/Mathematics/NumericalIntegrators/rungeKuttaVariableStepSizeIntegrator.h"
#include "Tudat/Mathematics/NumericalIntegrators/rungeKuttaCoefficients.h"
#include "Tudat/Mathematics/RootFinders/newtonRaphson.h"
#include "Tudat/Astrodynamics/Gravitation/jacobiEnergy.h"
#include "Tudat/Astrodynamics/BasicAstrodynamics/celestialBodyConstants.h"
#include "Tudat/Astrodynamics/Gravitation/librationPoint.h"
#include "Tudat/InputOutput/basicInputOutput.h"

#include "propagateOrbitAugmented.h"
#include "stateDerivativeModelAugmented.h"


Eigen::MatrixXd getFullAugmentedInitialState( const Eigen::Vector6d& initialState, const Eigen::Vector1d& initialMass, const Eigen::Vector1d& stableInitialMass, int integrationDirection )
{
    Eigen::MatrixXd fullInitialState = Eigen::MatrixXd::Zero( 7, 8 );
    fullInitialState.block( 0, 0, 6, 1 ) = initialState.block(0,0,6,1);

    if (integrationDirection == 1){
        fullInitialState.block( 6, 0, 1, 1)  = initialMass;
    } else {
        fullInitialState.block(6, 0, 1, 1) = stableInitialMass;
    }

    fullInitialState.block( 0, 1, 7, 7 ).setIdentity( );

    //std::cout << "THE BALLISTIC INITIAL STATE IS: " << initialState << std::endl;
    //std::cout << "THE AUGMENTED INITIAL STATE IS: " << fullInitialState << std::endl;

    return fullInitialState;
}

std::pair< Eigen::MatrixXd, double > propagateOrbitAugmented(
        const Eigen::MatrixXd& stateVectorInclSTM, double massParameter, double currentTime,
        int direction, std::string spacecraftName, std::string thrustPointing, double initialStepSize, double maximumStepSize)
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
                &computeStateDerivativeAugmented, 0.0, stateVectorInclSTM, minimumStepSize, maximumStepSize, relativeErrorTolerance, absoluteErrorTolerance);

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
