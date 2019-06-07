#include <fstream>
#include <iomanip>
#include<iostream>
#include <boost/bind.hpp>
#include <boost/make_shared.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <boost/shared_array.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/shared_container_iterator.hpp>
#include <map>
#include <cmath>

#include "Tudat/Mathematics/BasicMathematics/mathematicalConstants.h"
#include "Tudat/Astrodynamics/BasicAstrodynamics/celestialBodyConstants.h"
#include "Tudat/Astrodynamics/Gravitation/jacobiEnergy.h"
#include "Tudat/Astrodynamics/Gravitation/librationPoint.h"
#include "Tudat/Mathematics/BasicMathematics/function.h"

#include "Tudat/Mathematics/RootFinders/newtonRaphson.h"

#include "functions/artificialLibrationPointLocationFunction.h"
#include "functions/artificialLibrationPointLocationFunction1.h"
#include "functions/artificialLibrationPointLocationFunction2.h"
//#include "functions/signfunction.h"
#include "applyMultivariateRootFinding.h"

#include "createEquilibriumLocations.h"

Eigen::MatrixXd computeEquilibriaStability(Eigen::Vector2d equilibriumLocation, const double alpha, const double accelerationMagnitude, const double massParameter) {
    Eigen::MatrixXd statePropagationMatrix = Eigen::MatrixXd::Zero(6,6);
    double xPositionScaledSquared = (equilibriumLocation(0)+massParameter) * (equilibriumLocation(0)+massParameter);
    double xPositionScaledSquared2 = (1.0-massParameter-equilibriumLocation(0)) * (1.0-massParameter-equilibriumLocation(0));
    double yPositionScaledSquared = (equilibriumLocation(1) * equilibriumLocation(1) );

    // Compute distances to primaries.
    double distanceToPrimaryBody   = sqrt(xPositionScaledSquared     + yPositionScaledSquared);
    double distanceToSecondaryBody = sqrt(xPositionScaledSquared2 + yPositionScaledSquared);

    double distanceToPrimaryCubed = distanceToPrimaryBody * distanceToPrimaryBody * distanceToPrimaryBody;
    double distanceToSecondaryCubed = distanceToSecondaryBody * distanceToSecondaryBody * distanceToSecondaryBody;

    double distanceToPrimaryToFifthPower = distanceToPrimaryCubed * distanceToPrimaryBody * distanceToPrimaryBody;
    double distanceToSecondaryToFifthPower = distanceToSecondaryCubed * distanceToSecondaryBody * distanceToSecondaryBody;

    // Compute partial derivatives of the potential.
    double Uxx = (3.0*(1.0-massParameter)*xPositionScaledSquared          )/distanceToPrimaryToFifthPower+ (3.0*massParameter*xPositionScaledSquared2           )/distanceToSecondaryToFifthPower - (1.0-massParameter)/distanceToPrimaryCubed - massParameter/distanceToSecondaryCubed + 1.0;
    double Uxy = (3.0*(1.0-massParameter)*(equilibriumLocation(0)+massParameter)*equilibriumLocation(1))/distanceToPrimaryToFifthPower- (3.0*massParameter*(1.0-massParameter-equilibriumLocation(0))*equilibriumLocation(1))/distanceToSecondaryToFifthPower;
    double Uyx = Uxy;
    double Uyy = (3.0*(1.0-massParameter)*yPositionScaledSquared                         )/distanceToPrimaryToFifthPower+ (3.0*massParameter*yPositionScaledSquared                             )/distanceToSecondaryToFifthPower - (1.0-massParameter)/distanceToPrimaryCubed - massParameter/distanceToSecondaryCubed + 1.0 ;

    // Compute partial derivatives of the low-thrust acceleration
    double xAccPartialMass = -accelerationMagnitude * std::cos(alpha * tudat::mathematical_constants::PI / 180.0);
    double xAccPartialalpha = -accelerationMagnitude * std::sin(alpha * tudat::mathematical_constants::PI / 180.0);
    double yAccPartialMass = -accelerationMagnitude * std::sin(alpha * tudat::mathematical_constants::PI / 180.0);
    double yAccPartialalpha = -accelerationMagnitude * std::cos(alpha * tudat::mathematical_constants::PI / 180.0);

    statePropagationMatrix << 0,   0,   1,   0, 0,               0,
                              0,   0,   0,   1, 0,               0,
                              Uxx, Uxy, 0,   2, xAccPartialMass, xAccPartialalpha,
                              Uyx, Uyy, -2,  0, yAccPartialMass, yAccPartialalpha,
                              0,   0,   0,   0, 0,               0,
                              0,   0,   0,   0, 0,               0;

    return statePropagationMatrix;
}

void writeResultsToFile (const int librationPointNr, const double thrustAcceleration, std::map< double, Eigen::Vector3d > equilibriaCatalog, std::map <double, Eigen::MatrixXd > stabilityCatalog ) {

    // Prepare file for initial conditions
    remove(("/Users/Sjors/Documents/thesisSoftware/tudatBundle/tudatApplications/thesisProject/data/data/raw/equilibria/L" + std::to_string(librationPointNr) + "_" + std::to_string(thrustAcceleration) + "_equilibria.txt").c_str());
    std::ofstream textFileInitialConditions;
    textFileInitialConditions.open(("/Users/Sjors/Documents/thesisSoftware/tudatBundle/tudatApplications/thesisProject/data/raw/equilibria/L" + std::to_string(librationPointNr) + "_" + std::to_string(thrustAcceleration) + "_equilibria.txt"));
    textFileInitialConditions.precision(std::numeric_limits<double>::digits10);

    for(auto ic = equilibriaCatalog.cbegin(); ic != equilibriaCatalog.cend(); ++ic) {
        textFileInitialConditions << std::left << std::scientific                                          << std::setw(25)
                                  << ic->first  << std::setw(25) << ic->second(0)  << std::setw(25) << ic->second(1) << std::setw(25) << ic->second(2) << std::endl;

    }

    remove(("/Users/Sjors/Documents/thesisSoftware/tudatBundle/tudatApplications/thesisProject/data/data/raw/equilibria/L" + std::to_string(librationPointNr) + "_" + std::to_string(thrustAcceleration) + "_equilibria_stability.txt").c_str());
    std::ofstream textFileInitialConditionsStability;
    textFileInitialConditionsStability.open(("/Users/Sjors/Documents/thesisSoftware/tudatBundle/tudatApplications/thesisProject/data/raw/equilibria/L" + std::to_string(librationPointNr) + "_" + std::to_string(thrustAcceleration) + "_equilibria_stability.txt"));
    textFileInitialConditionsStability.precision(std::numeric_limits<double>::digits10);

        for(auto ic = stabilityCatalog.cbegin(); ic != stabilityCatalog.cend(); ++ic) {
        textFileInitialConditionsStability << std::left << std::scientific                                          << std::setw(25)
                                           << ic->first  << std::setw(25) << ic->second(0)  << std::setw(25) << ic->second(1) << std::setw(25) << ic->second(2) << std::setw(25) << ic->second(3) << std::setw(25) << ic->second(4) << std::setw(25) << ic->second(5) << std::setw(25) << ic->second(6) << std::setw(25)
                                                                          << ic->second(7)  << std::setw(25) << ic->second(8) << std::setw(25) << ic->second(9) << std::setw(25) << ic->second(10) << std::setw(25) << ic->second(11) << std::setw(25) << ic->second(12) << std::setw(25)
                                                                          << ic->second(13)  << std::setw(25) << ic->second(14) << std::setw(25) << ic->second(15) << std::setw(25) << ic->second(16) << std::setw(25) << ic->second(17) << std::setw(25) << ic->second(18) << std::setw(25)
                                                                          << ic->second(19)  << std::setw(25) << ic->second(20) << std::setw(25) << ic->second(21) << std::setw(25) << ic->second(22) << std::setw(25) << ic->second(23) << std::setw(25) << ic->second(24) << std::setw(25)
                                                                          << ic->second(25)  << std::setw(25) << ic->second(26) << std::setw(25) << ic->second(27) << std::setw(25) << ic->second(28) << std::setw(25) << ic->second(29) << std::setw(25) << ic->second(30) << std::setw(25)
                                                                          << ic->second(31)  << std::setw(25) << ic->second(32) << std::setw(25) << ic->second(33) << std::setw(25) << ic->second(34) << std::setw(25) << ic->second(35) << std::endl;
    }


}




Eigen::Vector2d createEquilibriumLocations (const int librationPointNr, const double thrustAcceleration, const double accelerationAngle, const double massParameter, const double maxDeviationFromSolution, const int maxIterations) {

    //std::cout << "Compute the collinear equilibria, serving as seed solution" << std::endl;

    // Set output precision and clear screen.
    std::cout.precision(14);
    double xLocationEquilibrium;
    Eigen::Vector2d equilibriumLocation;
    Eigen::Vector2d targetEquilibrium;
    Eigen::Vector3d equilibriumLocationWithIterations;
    Eigen::MatrixXd linearizedStability;
    std::map< double, Eigen::Vector3d > equilibriaCatalog;
    std::map< double, Eigen::MatrixXd > stabilityCatalog;

    if (librationPointNr == 1)
    {
        // Create object containing the functions.
        // boost::shared_ptr< LibrationPointLocationFunction1 > LibrationPointLocationFunction = boost::make_shared< LibrationPointLocationFunction1 >( 1, massParameter );
        std::shared_ptr<ArtificialLibrationPointLocationFunction1> ArtificialLibrationPointLocationFunction = std::make_shared< ArtificialLibrationPointLocationFunction1 > (1, thrustAcceleration);

        // The termination condition.
        tudat::root_finders::NewtonRaphson::TerminationFunction terminationConditionFunction =
                boost::bind( &tudat::root_finders::termination_conditions::RootAbsoluteToleranceTerminationCondition< double >::checkTerminationCondition,
                             boost::make_shared< tudat::root_finders::termination_conditions::RootAbsoluteToleranceTerminationCondition< double > >(
                                     maxDeviationFromSolution ), _1, _2, _3, _4, _5 );

        // Test Newton-Raphson object.
        tudat::root_finders::NewtonRaphson newtonRaphson( terminationConditionFunction );

//        std::cout << "=====================" << std::endl;
//        std::cout << "True root Location: " << ArtificialLibrationPointLocationFunction->getTrueRootLocation( ) << std::endl;
//        std::cout << "True root accuracy: " << ArtificialLibrationPointLocationFunction->getTrueRootAccuracy( ) << std::endl;
//        std::cout << "Initial guess: " << ArtificialLibrationPointLocationFunction->getInitialGuess( ) << std::endl;
//        std::cout << "Evaluate initial guess:" << ArtificialLibrationPointLocationFunction->evaluate(ArtificialLibrationPointLocationFunction->getInitialGuess()) << std::endl;
//        std::cout << "Evaluate Derivative initial guess:" << ArtificialLibrationPointLocationFunction->computeDerivative(1, ArtificialLibrationPointLocationFunction->getInitialGuess()) << std::endl;
//        std::cout << "Lower Bound: " << ArtificialLibrationPointLocationFunction->getLowerBound( ) << std::endl;
//        std::cout << "Upper Bound: " << ArtificialLibrationPointLocationFunction->getUpperBound( ) << std::endl;
//        std::cout << "Test Sign function A: " << signfunction(ArtificialLibrationPointLocationFunction->getInitialGuess( )+massParameter) << std::endl;
//        std::cout << "Test Sign function B: " << signfunction(ArtificialLibrationPointLocationFunction->getInitialGuess( ) - 1.0 + massParameter) << std::endl;
//        std::cout << "=====================" << std::endl;

        // Let Newton-Raphson search for the root.
        xLocationEquilibrium = newtonRaphson.execute( ArtificialLibrationPointLocationFunction, ArtificialLibrationPointLocationFunction->getInitialGuess( ) );
        //std::cout << "The root of the seed solution is: " << xLocationEquilibrium << std::endl;
        //std::cout << "The residual of the seed solution is:" << ArtificialLibrationPointLocationFunction->evaluate(xLocationEquilibrium) << std::endl;
        equilibriumLocation(0) = xLocationEquilibrium;
        equilibriumLocation(1) = 0.0;
        int numberOfIterations = 0;
        equilibriumLocationWithIterations.block(0,0,2,1) = equilibriumLocation.block(0,0,2,1);
        equilibriumLocationWithIterations(2) = numberOfIterations;
        linearizedStability = computeEquilibriaStability(equilibriumLocation, 0.0, thrustAcceleration, massParameter);
        //std::cout << "linearized stability is at natural L point: " << linearizedStability << std::endl;
    } else {
        // Create object containing the functions.
        //boost::shared_ptr< LibrationPointLocationFunction2 > LibrationPointLocationFunction = boost::make_shared< LibrationPointLocationFunction2 >( 1, massParameter );
        std::shared_ptr<ArtificialLibrationPointLocationFunction2> ArtificialLibrationPointLocationFunction = std::make_shared< ArtificialLibrationPointLocationFunction2 > (2, thrustAcceleration);

        // The termination condition.
        tudat::root_finders::NewtonRaphson::TerminationFunction terminationConditionFunction =
                boost::bind( &tudat::root_finders::termination_conditions::RootAbsoluteToleranceTerminationCondition< double >::checkTerminationCondition,
                             boost::make_shared< tudat::root_finders::termination_conditions::RootAbsoluteToleranceTerminationCondition< double > >(
                                     maxDeviationFromSolution ), _1, _2, _3, _4, _5 );
        // Test Newton-Raphson object.
        tudat::root_finders::NewtonRaphson newtonRaphson( terminationConditionFunction );

//        std::cout << "=====================" << std::endl;
//        std::cout << "True root Location: " << ArtificialLibrationPointLocationFunction->getTrueRootLocation( ) << std::endl;
//        std::cout << "True root accuracy: " << ArtificialLibrationPointLocationFunction->getTrueRootAccuracy( ) << std::endl;
//        std::cout << "Initial guess: " << ArtificialLibrationPointLocationFunction->getInitialGuess( ) << std::endl;
//        std::cout << "Evaluate initial guess:" << ArtificialLibrationPointLocationFunction->evaluate(ArtificialLibrationPointLocationFunction->getInitialGuess()) << std::endl;
//        std::cout << "Evaluate Derivative initial guess:" << ArtificialLibrationPointLocationFunction->computeDerivative(1, ArtificialLibrationPointLocationFunction->getInitialGuess()) << std::endl;
//        std::cout << "Lower Bound: " << ArtificialLibrationPointLocationFunction->getLowerBound( ) << std::endl;
//        std::cout << "Upper Bound: " << ArtificialLibrationPointLocationFunction->getUpperBound( ) << std::endl;
//        std::cout << "Test Sign function A: " << signfunction(ArtificialLibrationPointLocationFunction->getInitialGuess( )+massParameter) << std::endl;
//        std::cout << "Test Sign function B: " << signfunction(ArtificialLibrationPointLocationFunction->getInitialGuess( ) - 1.0 + massParameter) << std::endl;
//        std::cout << "=====================" << std::endl;

        // Let Newton-Raphson search for the root.
        xLocationEquilibrium = newtonRaphson.execute( ArtificialLibrationPointLocationFunction, ArtificialLibrationPointLocationFunction->getInitialGuess( ) );
        //std::cout << "The root of the seed solution is: " << xLocationEquilibrium << std::endl;
        //std::cout << "The residual of the seed solution is:" << ArtificialLibrationPointLocationFunction->evaluate(xLocationEquilibrium) << std::endl;

        equilibriumLocation(0) = xLocationEquilibrium;
        equilibriumLocation(1) = 0.0;
        int numberOfIterations = 0;
        equilibriumLocationWithIterations.block(0,0,2,1) = equilibriumLocation.block(0,0,2,1);
        equilibriumLocationWithIterations(2) = numberOfIterations;
        linearizedStability = computeEquilibriaStability(equilibriumLocation, 0.0, thrustAcceleration, massParameter);
        //std::cout << "linearized stability is at natural L point: " << linearizedStability << std::endl;

    }

    //Store the first Equilibrium location
    equilibriaCatalog[ 0.0 * tudat::mathematical_constants::PI / 180.0 ] = equilibriumLocationWithIterations;
    stabilityCatalog[ 0.0 * tudat::mathematical_constants::PI / 180.0 ] = linearizedStability;

    if ( accelerationAngle == 0.0){

        targetEquilibrium = equilibriumLocation;

    }

    for (int i = 1; i <= 3600; i++ ) {
        auto alpha = static_cast< double > (i * 0.1);
        equilibriumLocationWithIterations = applyMultivariateRootFinding(librationPointNr, equilibriumLocation, alpha, thrustAcceleration ,massParameter, 5.0e-15, maxIterations );
        equilibriumLocation = equilibriumLocationWithIterations.block(0,0,2,1);
        linearizedStability = computeEquilibriaStability(equilibriumLocation, 0.0, thrustAcceleration, massParameter);

        equilibriaCatalog[ alpha * tudat::mathematical_constants::PI / 180.0 ] = equilibriumLocationWithIterations;
        stabilityCatalog [alpha * tudat::mathematical_constants::PI / 180.0 ] = linearizedStability;

        if (alpha == accelerationAngle ){

            targetEquilibrium = equilibriumLocation;

        }
    }

    writeResultsToFile(librationPointNr, thrustAcceleration, equilibriaCatalog, stabilityCatalog);

    return targetEquilibrium;

}
