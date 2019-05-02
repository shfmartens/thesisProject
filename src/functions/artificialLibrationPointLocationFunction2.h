/*    Copyright (c) 2010-2017, Delft University of Technology
 *    All rigths reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 *
 */

#ifndef TUDATBUNDLE_ARTIFICIALLIBRATIONPOINTLOCATIONFUNCTION2_H
#define TUDATBUNDLE_ARTIFICIALLIBRATIONPOINTLOCATIONFUNCTION2_H

#include <cmath>
#include <stdexcept>
#include "Tudat/Mathematics/BasicMathematics/basicFunction.h"

#include "artificialLibrationPointLocationFunction.h"


//! Test function for the root-finders.
/*!
 * This struct contains functions, necessary to test the various root-finding methods.
 * The test function implemented in this case is:
 * \f[
 *      f(x) = x^{2} - 3
 * \f]
 */
struct ArtificialLibrationPointLocationFunction2 : public ArtificialLibrationPointLocationFunction,
        public tudat::basic_mathematics::BasicFunction< double, double >
{
    //! Maximum order of the derivative before throwing an exception.
    unsigned int maximumDerivativeOrder;

    //! Create a function, where aMaximumDerivativeOrder is the maximum order of the derivative.
    ArtificialLibrationPointLocationFunction2( unsigned int aMaximumDerivativeOrder, const double thrustAcceleration )
        : maximumDerivativeOrder( aMaximumDerivativeOrder ), thrustAcceleration_( thrustAcceleration )
    { }

    //! Mathematical test function.
    double evaluate( const double inputValue )
    {
        extern double massParameter;
        // Define Mathematical function: f(x) =
        return inputValue - (1.0 - massParameter) * (inputValue + massParameter) / (signfunction(inputValue + massParameter) * pow(inputValue+massParameter,3.0))
                -1.0 * massParameter * (inputValue - 1.0 + massParameter)/(signfunction(inputValue -1.0 + massParameter)*pow(inputValue-1.0+massParameter,3.0)) + thrustAcceleration_;

    }

    //! Derivatives of mathematical test function.
    double computeDerivative( const unsigned int order, const double inputValue )
    { 
        extern double massParameter;
        // Sanity check.
        if ( order > maximumDerivativeOrder )
        {
            throw std::runtime_error( "The root-finder should not evaluate higher derivatives!" );
        }

        // Return the analytical expression for the derivatives.
        if ( order == 0 )
        {
            // Return the function value: y =
            return evaluate( inputValue );
        }

        else if ( order == 1 )
        {
            // Return the first derivative function value: y =
            return 1.0 + (1.0 - massParameter) * 2.0/(signfunction(inputValue+massParameter)*pow(inputValue+massParameter, 3.0))
                    + massParameter * 2.0/(signfunction(inputValue-1.0+massParameter)*pow(inputValue-1.0+massParameter, 3.0));
        }

        else if ( order == 2 )
        {
            // Return the second derivative function value: y = .
            return (1.0 - massParameter) * -6.0/(signfunction(inputValue+massParameter)*pow(inputValue+massParameter, 4.0))
                    + massParameter * -6.0/(signfunction(inputValue-1.0+massParameter)*pow(inputValue-1.0+massParameter, 4.0));
        }

        else
        {
            throw std::runtime_error(
                        "An error occured when evaluating the order of the derivative." );
        }
    }

    //! Crash on integration as root_finders should not execute these.
    double computeDefiniteIntegral( const unsigned int order, const double lowerBound,
                                    const double upperbound )
    {
        throw std::runtime_error( "The root-finder should not evaluate integrals!" );
    }

    //! Get the expected true location of the root.
    /*!
     * Returns the expected true location of the function root, here \f$1\f$.
     *
     * \return True location of the root.
     */
    double getTrueRootLocation( )
    {
        return 1.1556821477825;
    }
    
    //! Get the accuracy of the true location of the root.
    /*!
     * Returns the accuracy of the true location of the function root, here 1e-15.
     *
     * \return Accuracy of the true location of the root.
     */
    double getTrueRootAccuracy( ) { return 0.1; }

    //! Get a reasonable initial guess of the root location.
    /*!
     * Returns a reasonable initial guess for the true location of the function root, here 4.
     *
     * \return Initial guess for the true location of the function root.
     */
    double getInitialGuess( ) { return 1.1556821477825; }

    //! Get a reasonable lower boundary for the root location.
    /*!
     * Returns a reasonable lower bound for the true location of the function root, here -1.
     *
     * \return Lower bound for the true location of the function root.
     */
    double getLowerBound( ) {
        extern double massParameter;
        return 1.001-massParameter; }

    //! Get a reasonable upper boundary for the root location.
    /*!
     * Returns a reasonable upper bound for the true location of the function root, here 4.
     *
     * \return Upper bound for the true location of the function root.
     */
    double getUpperBound( ) { return 1.4; }

protected:

    double thrustAcceleration_;

private:
};


#endif // TUDATBUNDLE_ARTIFICIALLIBRATIONPOINTLOCATIONFUNCTION2_H
