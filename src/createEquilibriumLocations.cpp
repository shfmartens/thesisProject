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

#include "Tudat/Mathematics/BasicMathematics/mathematicalConstants.h"
#include "Tudat/Astrodynamics/BasicAstrodynamics/celestialBodyConstants.h"
#include "Tudat/Astrodynamics/Gravitation/jacobiEnergy.h"
#include "Tudat/Astrodynamics/Gravitation/librationPoint.h"
#include "Tudat/Mathematics/BasicMathematics/function.h"

#include "Tudat/Mathematics/RootFinders/newtonRaphson.h"

#include "functions/artificialLibrationPointLocationFunction.h"
#include "functions/artificialLibrationPointLocationFunction1.h"
#include "functions/artificialLibrationPointLocationFunction2.h"
#include "functions/signfunction.h"
#include "applyMultivariateRootFinding.h"

#include "createEquilibriumLocations.h"

void writeResultsToFile (const int librationPointNr, const double thrustAcceleration, std::map< double, Eigen::Vector2d > equilibriaCatalog ) {

    // Prepare file for initial conditions
    remove(("/Users/Sjors/Documents/thesisSoftware/tudatBundle/tudatApplications/thesisProject/data/data/raw/equilibria/L" + std::to_string(librationPointNr) + "_" + std::to_string(thrustAcceleration) + "_equilibria.txt").c_str());
    std::ofstream textFileInitialConditions;
    textFileInitialConditions.open(("/Users/Sjors/Documents/thesisSoftware/tudatBundle/tudatApplications/thesisProject/data/raw/equilibria/L" + std::to_string(librationPointNr) + "_" + std::to_string(thrustAcceleration) + "_equilibria.txt"));
    textFileInitialConditions.precision(std::numeric_limits<double>::digits10);

    for(auto ic = equilibriaCatalog.cbegin(); ic != equilibriaCatalog.cend(); ++ic) {
        textFileInitialConditions << std::left << std::scientific                                          << std::setw(25)
                                  << ic->first  << std::setw(25) << ic->second(0)  << std::setw(25) << ic->second(1)  << std::endl;

    }


}


void createEquilibriumLocations (const int librationPointNr, const double thrustAcceleration, const double massParameter, const double maxDeviationFromSolution, const int maxIterations) {

    std::cout << "Compute the collinear equilibria, serving as seed solution" << std::endl;

    // Set output precision and clear screen.
    std::cout.precision(14);
    double xLocationEquilibrium;
    Eigen::Vector2d equilibiumLocation;
    std::map< double, Eigen::Vector2d > equilibriaCatalog;

    if (librationPointNr == 1)
    {
        // Create object containing the functions.
        // boost::shared_ptr< LibrationPointLocationFunction1 > LibrationPointLocationFunction = boost::make_shared< LibrationPointLocationFunction1 >( 1, massParameter );
        std::shared_ptr<ArtificialLibrationPointLocationFunction1> ArtificialLibrationPointLocationFunction = std::make_shared< ArtificialLibrationPointLocationFunction1 > (1, massParameter);

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
        std::cout << "The root of the seed solution is: " << xLocationEquilibrium << std::endl;
        std::cout << "The residual of the seed solution is:" << ArtificialLibrationPointLocationFunction->evaluate(xLocationEquilibrium) << std::endl;
        equilibiumLocation(0) = xLocationEquilibrium;
        equilibiumLocation(1) = 0.0;

    } else {
        // Create object containing the functions.
        //boost::shared_ptr< LibrationPointLocationFunction2 > LibrationPointLocationFunction = boost::make_shared< LibrationPointLocationFunction2 >( 1, massParameter );
        std::shared_ptr<ArtificialLibrationPointLocationFunction2> ArtificialLibrationPointLocationFunction = std::make_shared< ArtificialLibrationPointLocationFunction2 > (2, massParameter);

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
        std::cout << "The root of the seed solution is: " << xLocationEquilibrium << std::endl;
        std::cout << "The residual of the seed solution is:" << ArtificialLibrationPointLocationFunction->evaluate(xLocationEquilibrium) << std::endl;

        equilibiumLocation(0) = xLocationEquilibrium;
        equilibiumLocation(1) = 0.0;
    }

    //Store the first Equilibrium location
    equilibriaCatalog[ 0.0 * tudat::mathematical_constants::PI / 180.0 ] = equilibiumLocation;

    for (int i = 1; i < 3600; i++ ) {
        auto alpha = static_cast< double > (i * 0.1);
        equilibiumLocation = applyMultivariateRootFinding(librationPointNr, equilibiumLocation, alpha, thrustAcceleration ,massParameter, 5.0e-15, maxIterations );

        equilibriaCatalog[ alpha * tudat::mathematical_constants::PI / 180.0 ] = equilibiumLocation;
    }

    writeResultsToFile(librationPointNr, thrustAcceleration, equilibriaCatalog);

//    for(auto it = equilibriaCatalog.cbegin(); it != equilibriaCatalog.cend(); ++it)
//    {
//        std::cout << it->first << " " << it->second << "\n";
//    }


}
