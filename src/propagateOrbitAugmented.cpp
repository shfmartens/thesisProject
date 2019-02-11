#include "Tudat/Mathematics/NumericalIntegrators/rungeKuttaVariableStepSizeIntegrator.h"
#include "Tudat/Mathematics/NumericalIntegrators/rungeKuttaCoefficients.h"
#include "Tudat/Mathematics/RootFinders/newtonRaphson.h"
#include "Tudat/Astrodynamics/Gravitation/jacobiEnergy.h"
#include "Tudat/Astrodynamics/BasicAstrodynamics/celestialBodyConstants.h"
#include "Tudat/Astrodynamics/Gravitation/librationPoint.h"
#include "Tudat/InputOutput/basicInputOutput.h"

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>
#include <Eigen/Dense>

#include "propagateOrbitAugmented.h"
#include "stateDerivativeModelAugmented.h"
#include "computeManifoldsAugmented.h"


Eigen::MatrixXd getFullAugmentedInitialState( const Eigen::Vector6d& initialState, const std::string spacecraftName, const std::string thrustPointing, int integrationDirection )
{
    Eigen::MatrixXd fullInitialState = Eigen::MatrixXd::Zero( 8, 9 );
    Eigen::Vector4d satelliteCharacteristic  = retrieveSpacecraftProperties(spacecraftName);
    fullInitialState.block( 0, 0, 6, 1 ) = initialState.block(0,0,6,1);

    if ((thrustPointing == "left" || thrustPointing == "right") && integrationDirection == 1) {
        fullInitialState( 6, 0) = satelliteCharacteristic(1);
        fullInitialState( 7, 0) = satelliteCharacteristic(0);
    }  else if ((thrustPointing == "left" || thrustPointing == "right") && integrationDirection == -1) {
        fullInitialState( 6, 0) = satelliteCharacteristic(3);
        fullInitialState( 7, 0) = satelliteCharacteristic(0);
    } else if ((thrustPointing != "left" || thrustPointing != "right") && integrationDirection == 1 ) {
        fullInitialState( 6, 0) = satelliteCharacteristic(1);
        fullInitialState( 7, 0) = satelliteCharacteristic(0) * satelliteCharacteristic(1) / 1.0;
    } else {
        fullInitialState( 6, 0) = satelliteCharacteristic(3);
        fullInitialState( 7, 0) = satelliteCharacteristic(0) * satelliteCharacteristic(3) / 1.0;
    }


    fullInitialState.block( 0, 1, 8, 8 ).setIdentity( );

//    std::cout << "THE  INITIAL THRUST  IS: " << fullInitialState.block(7, 0, 1, 1) << std::endl;
//    if ((thrustPointing == "left" || thrustPointing == "right")) {
//        std::cout << "THE EXPECTED INITIAL THRUST  IS: " << satelliteCharacteristic(0);
//    } else if ((thrustPointing != "left" || thrustPointing != "right") && integrationDirection == -1) {
//        std::cout << "THE EXPECTED INITIAL THRUST  IS: " << satelliteCharacteristic(0) * satelliteCharacteristic(3) / 1.0;
//    } else {
//        std::cout << "THE EXPECTED INITIAL THRUST  IS: " << satelliteCharacteristic(0) * satelliteCharacteristic(1) / 1.0;
//}

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
