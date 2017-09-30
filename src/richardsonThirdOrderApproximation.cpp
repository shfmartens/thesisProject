#include <boost/bind.hpp>
#include <boost/make_shared.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include "Tudat/Mathematics/BasicMathematics/mathematicalConstants.h"
#include "Tudat/Astrodynamics/BasicAstrodynamics/celestialBodyConstants.h"
#include "Tudat/Astrodynamics/Gravitation/jacobiEnergy.h"
#include "Tudat/Astrodynamics/Gravitation/librationPoint.h"

#include "Tudat/Mathematics/RootFinders/newtonRaphson.h"

#include "functions/librationPointLocationFunction.h"
#include "functions/librationPointLocationFunction1.h"
#include "functions/librationPointLocationFunction2.h"
#include "propagateOrbit.h"

Eigen::VectorXd richardsonThirdOrderApproximation(std::string orbitType, int librationPointNr,
                                                  double amplitude, double n )
{
    std::cout << "\nCreate initial conditions:\n" << std::endl;

    // Set output precision and clear screen.
    std::cout.precision(14);

    const double primaryGravitationalParameter = tudat::celestial_body_constants::EARTH_GRAVITATIONAL_PARAMETER;
    const double secondaryGravitationalParameter = tudat::celestial_body_constants::MOON_GRAVITATIONAL_PARAMETER;
    const double massParameter = tudat::gravitation::circular_restricted_three_body_problem::computeMassParameter( primaryGravitationalParameter, secondaryGravitationalParameter );
    Eigen::VectorXd initialStateVectorInclPeriod = Eigen::VectorXd::Zero(7);

    double gammaL;
    double c2;
    double c3;
    double c4;
    double Ax = 0.0;
    double Az = 0.0;

    if (librationPointNr == 1)
    {
        // Create object containing the functions.
        boost::shared_ptr< LibrationPointLocationFunction1 > LibrationPointLocationFunction = boost::make_shared< LibrationPointLocationFunction1 >( 1, massParameter );

        // The termination condition.
        tudat::root_finders::NewtonRaphson::TerminationFunction terminationConditionFunction =
                boost::bind( &tudat::root_finders::termination_conditions::RootAbsoluteToleranceTerminationCondition< double >::checkTerminationCondition,
                             boost::make_shared< tudat::root_finders::termination_conditions::RootAbsoluteToleranceTerminationCondition< double > >(
                                     LibrationPointLocationFunction->getTrueRootAccuracy( ) ), _1, _2, _3, _4, _5 );

        // Test Newton-Raphson object.
        tudat::root_finders::NewtonRaphson newtonRaphson( terminationConditionFunction );

        // Let Newton-Raphson search for the root.
        gammaL = newtonRaphson.execute( LibrationPointLocationFunction, LibrationPointLocationFunction->getInitialGuess( ) );

        c2 = 1.0 / pow(gammaL, 3.0) * (pow(1.0,2.0) * massParameter + pow(-1.0,2.0) * (1.0 - massParameter) * pow(gammaL, 2.0+1.0) / pow((1.0 - gammaL), (2.0+1.0)));
        c3 = 1.0 / pow(gammaL, 3.0) * (pow(1.0,3.0) * massParameter + pow(-1.0,3.0) * (1.0 - massParameter) * pow(gammaL, 3.0+1.0) / pow((1.0 - gammaL), (3.0+1.0)));
        c4 = 1.0 / pow(gammaL, 3.0) * (pow(1.0,4.0) * massParameter + pow(-1.0,4.0) * (1.0 - massParameter) * pow(gammaL, 4.0+1.0) / pow((1.0 - gammaL), (4.0+1.0)));
    } else {
        // Create object containing the functions.
        boost::shared_ptr< LibrationPointLocationFunction2 > LibrationPointLocationFunction = boost::make_shared< LibrationPointLocationFunction2 >( 1, massParameter );

        // The termination condition.
        tudat::root_finders::NewtonRaphson::TerminationFunction terminationConditionFunction =
                boost::bind( &tudat::root_finders::termination_conditions::RootAbsoluteToleranceTerminationCondition< double >::checkTerminationCondition,
                             boost::make_shared< tudat::root_finders::termination_conditions::RootAbsoluteToleranceTerminationCondition< double > >(
                                     LibrationPointLocationFunction->getTrueRootAccuracy( ) ), _1, _2, _3, _4, _5 );
        // Test Newton-Raphson object.
        tudat::root_finders::NewtonRaphson newtonRaphson( terminationConditionFunction );

        // Let Newton-Raphson search for the root.
        gammaL = newtonRaphson.execute( LibrationPointLocationFunction, LibrationPointLocationFunction->getInitialGuess( ) );

        c2 = 1 / pow(gammaL, 3.0) * (pow(-1.0, 2.0) * massParameter + pow(-1.0, 2.0) * (1.0 - massParameter) * pow(gammaL, 2.0+1.0) / pow((1.0 + gammaL), (2.0 + 1.0)));
        c3 = 1 / pow(gammaL, 3.0) * (pow(-1.0, 3.0) * massParameter + pow(-1.0, 3.0) * (1.0 - massParameter) * pow(gammaL, 3.0+1.0) / pow((1.0 + gammaL), (3.0 + 1.0)));
        c4 = 1 / pow(gammaL, 3.0) * (pow(-1.0, 4.0) * massParameter + pow(-1.0, 4.0) * (1.0 - massParameter) * pow(gammaL, 4.0+1.0) / pow((1.0 + gammaL), (4.0 + 1.0)));
    }

    double lambda = pow(1.0 - c2/2.0 + 1.0/2.0*pow((pow((c2-2.0), 2.0) + 4.0*(c2-1.0)*(1.0+2.0*c2)), 0.5), 0.5);
    double k      = 2.0 * lambda / (pow(lambda,2.0) + 1.0 - c2);
    double delta  = pow(lambda,2.0) - c2;

    double d1  = 3.0 * pow(lambda, 2.0) / k * (k * (6.0 * pow(lambda, 2.0) - 1.0) - 2.0 * lambda);
    double d2  = 8.0 * pow(lambda, 2.0) / k * (k * (11.0 * pow(lambda, 2.0) - 1.0) - 2.0 * lambda);

    double a21 = 3.0 * c3 * (pow(k,2.0) - 2.0) / (4.0 * (1.0 + 2.0 * c2));
    double a22 = 3.0 * c3 / (4.0 * (1.0 + 2.0 * c2));
    double a23 = -3.0 * c3 * lambda / (4.0 * k * d1) * (3.0 * pow(k,3.0) * lambda - 6.0 * k * (k - lambda) + 4.0);
    double a24 = -3.0 * c3 * lambda / (4.0 * k * d1) * (2.0 + 3.0 * k * lambda);

    double b21 = -3.0 * c3 * lambda / (2.0 * d1) * (3.0 * k * lambda - 4.0);
    double b22 = 3.0 * c3 * lambda / d1;

    double d21 = -c3 / (2.0 * pow(lambda,2.0));

    double a31 = -9.0 * lambda / (4.0 * d2) * (4.0 * c3 * (k * a23 - b21) + k * c4 * (4.0 + pow(k, 2.0)))
          + (9.0 * pow(lambda, 2.0) + 1.0 - c2) / (2.0 * d2) * (3.0 * c3 * (2.0 * a23 - k * b21) + c4 * (2.0 + 3.0 * pow(k,2.0)));
    double a32 = -1.0 / d2 * (9.0 * lambda / 4.0 * (4.0 * c3 * (k * a24 - b22) + k * c4)
                      + 3.0 / 2.0 * (9.0 * pow(lambda, 2.0) + 1.0 - c2) * (c3 * (k * b22 + d21 - 2.0 * a24) - c4));

    double b31 = 3.0 / (8.0 * d2) * (8.0 * lambda * (3.0 * c3 * (k * b21 - 2.0 * a23) - c4 * (2.0 + 3.0 * pow(k, 2.0)))
                          + (9.0 * pow(lambda, 2.0) + 1.0 + 2.0 * c2) * (4.0 * c3 * (k * a23 - b21) + k * c4 * (4.0 + pow(k, 2.0))));
    double b32 = 1.0 / d2 * (9.0 * lambda * (c3 * (k * b22 + d21 - 2 * a24) - c4)
                    + 3.0/8.0 * (9.0 * pow(lambda, 2.0) + 1.0 + 2.0 * c2) * (4.0 * c3 * (k * a24 - b22) + k * c4));

    double d31 = 3.0 / (64.0 * pow(lambda, 2.0)) * (4.0 * c3 * a24 + c4);
    double d32 = 3.0 / (64.0 * pow(lambda, 2.0)) * (4.0 * c3 * (a23 - d21) + c4 * (4.0 + pow(k,2.0)));

    double a1  = -3.0/2.0 * c3 * (2.0 * a21 + a23 + 5.0 * d21) - 3.0/8.0 * c4 * (12.0 - pow(k,2.0));
    double a2  = 3.0/2.0 * c3 * (a24 - 2.0 * a22) + 9.0/8.0 * c4;

    double s1  = 1.0 / (2.0 * lambda * (lambda * (1.0 + pow(k, 2.0)) - 2.0 * k)) * (
            3.0 / 2.0 * c3 * (2.0 * a21 * (pow(k, 2.0) - 2.0) - a23 * (pow(k, 2.0) + 2.0) - 2.0 * k * b21) - 3.0/8.0 * c4 * (
                    3.0 * pow(k, 4.0) - 8.0 * pow(k, 2.0) + 8.0));
    double s2  = 1.0 / (2.0 * lambda * (lambda * (1.0 + pow(k, 2.0)) - 2.0 * k)) * (
            3.0/2.0 * c3 * (2.0 * a22 * (pow(k, 2.0) - 2.0) + a24 * (pow(k, 2.0) + 2.0) + 2.0 * k * b22 + 5.0 * d21) + 3.0/8.0 * c4 * (
                    12.0 - pow(k, 2.0)));

    double l1  = a1 + 2.0 * pow(lambda, 2.0) * s1;
    double l2  = a2 + 2.0 * pow(lambda, 2.0) * s2;

    if (orbitType == "horizontal"){
        Ax = amplitude;
        Az = 0.0;
    } if (orbitType == "vertical"){
        Ax = 0.0;
        Az = amplitude;
    } if (orbitType == "halo"){
        Az = amplitude;
        Ax = pow(((-delta - l2 * pow(Az, 2.0)) / l1), 0.5);
    }

    std::cout << "Ax = " << Ax << ", Az =  = " << Az << std::endl;

    double omega1 = 0.0;
    double omega2 = s1 * pow(Ax, 2.0) + s2 * pow(Az, 2.0);
    double omega  = 1.0 + omega1 + omega2;

    double tau1   = 0.0;
    double deltan = 2.0 - n;

    double x             = a21 * pow(Ax, 2.0) + a22 * pow(Az, 2.0) - Ax * std::cos(tau1) + (a23 * pow(Ax, 2.0) - a24 * pow(Az, 2.0)) * std::cos(2.0 * tau1) + (a31 * pow(Ax, 3.0) - a32 * Ax * pow(Az, 2.0)) * std::cos(3.0 * tau1);
    double y             = k * Ax * std::sin(tau1) + (b21 * pow(Ax, 2.0) - b22 * pow(Az, 2.0)) * std::sin(2 * tau1) + (b31 * pow(Ax, 3.0) - b32 * Ax * pow(Az, 2.0)) * std::sin(3.0 * tau1);
    double z             = deltan * Az * std::cos(tau1) + deltan * d21 * Ax * Az * (std::cos(2.0 * tau1) - 3.0) + deltan * (d32 * Az * pow(Ax, 2.0) - d31 * pow(Az, 3.0)) * std::cos(3.0 * tau1);
    double xdot          = omega * lambda * Ax * std::sin(tau1) - 2.0 * omega * lambda * (a23 * pow(Ax, 2.0) - a24 * pow(Az, 2.0)) * std::sin(2.0 * tau1) - 3.0 * omega * lambda * (a31 * pow(Ax, 3.0) - a32 * Ax * pow(Az, 2.0)) * std::sin(3.0 * tau1);
    double ydot          = omega * lambda * (k * Ax * std::cos(tau1) + 2.0 * (b21 * pow(Ax, 2.0) - b22 * pow(Az, 2.0)) * std::cos(2.0 * tau1) + 3.0 * (b31 * pow(Ax, 3.0) - b32 * Ax * pow(Az, 2.0)) * std::cos(3.0 * tau1));
    double zdot          = -1.0 * omega * lambda * deltan * Az * std::sin(tau1) - 2.0 * omega * lambda * deltan * d21 * Ax * Az * std::sin(2.0 * tau1) - 3.0 * omega * lambda * deltan * (d32 * Az * pow(Ax, 2.0) - d31 * pow(Az, 3.0)) * std::sin(3.0 * tau1);
    double orbitalPeriod = 2.0 * tudat::mathematical_constants::PI / (lambda * omega);

    if (librationPointNr == 1){
        initialStateVectorInclPeriod(0) = (x - 1.0) * gammaL + 1.0 - massParameter;
    } if (librationPointNr == 2){
        initialStateVectorInclPeriod(0) = (x + 1.0) * gammaL + 1.0 - massParameter;
    }

    initialStateVectorInclPeriod(1) = y * gammaL;
    initialStateVectorInclPeriod(2) = z * gammaL;
    initialStateVectorInclPeriod(3) = xdot * gammaL;
    initialStateVectorInclPeriod(4) = ydot * gammaL;
    initialStateVectorInclPeriod(5) = zdot * gammaL;
    initialStateVectorInclPeriod(6) = orbitalPeriod;

    std::cout << initialStateVectorInclPeriod << std::endl;
    
//    std::cout << "gammaL = " << gammaL << endl
//              << "lambda = " << lambda << endl
//              << "k = "      << k      << endl
//              << "delta = "  << delta  << endl
//              << "c2 = "     << c2     << endl
//              << "c3 = "     << c3     << endl
//              << "c4 = "     << c4     << endl
//              << "s1 = "     << s1     << endl
//              << "s2 = "     << s2     << endl
//              << "l1 = "     << l1     << endl
//              << "l2 = "     << l2     << endl
//              << "a1 = "     << a1     << endl
//              << "a2 = "     << a2     << endl
//              << "d1 = "     << d1     << endl
//              << "d2 = "     << d2     << endl
//              << "a21 = "    << a21    << endl
//              << "a22 = "    << a22    << endl
//              << "a23 = "    << a23    << endl
//              << "a24 = "    << a24    << endl
//              << "a31 = "    << a31    << endl
//              << "a32 = "    << a32    << endl
//              << "b21 = "    << b21    << endl
//              << "b22 = "    << b22    << endl
//              << "b31 = "    << b31    << endl
//              << "b32 = "    << b32    << endl
//              << "d21 = "    << d21    << endl
//              << "d31 = "    << d31    << endl
//              << "d32 = "    << d32    << endl;
    return initialStateVectorInclPeriod;
}
