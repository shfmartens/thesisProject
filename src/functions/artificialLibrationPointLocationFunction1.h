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

#ifndef TUDATBUNDLE_ARTIFICIALLIBRATIONPOINTLOCATIONFUNCTION1_H
#define TUDATBUNDLE_ARTIFICIALLIBRATIONPOINTLOCATIONFUNCTION1_H

#include <cmath>
#include <iostream>
#include <stdexcept>
#include "Tudat/Mathematics/BasicMathematics/basicFunction.h"
#include "signfunction.h"

#include "artificialLibrationPointLocationFunction.h"


//! Test function for the root-finders.
/*!
 * This struct contains functions, necessary to test the various root-finding methods.
 * The test function implemented in this case is:
 * \f[
 *      f(x) = x^{2} - 3
 * \f]
 */
struct ArtificialLibrationPointLocationFunction1 : public ArtificialLibrationPointLocationFunction,
        public tudat::basic_mathematics::BasicFunction< double, double >
{
    //! Maximum order of the derivative before throwing an exception.
    unsigned int maximumDerivativeOrder;

    //! Create a function, where aMaximumDerivativeOrder is the maximum order of the derivative.
    ArtificialLibrationPointLocationFunction1( unsigned int aMaximumDerivativeOrder, const double massParameter )
        : maximumDerivativeOrder( aMaximumDerivativeOrder ), massParameter_( massParameter )
    { }

    //! Mathematical test function.
    double evaluate( const double inputValue )
    {
        extern double thrustAcceleration;
        // Define Mathematical function: f(x) =

        return inputValue - (1.0 - massParameter_) * (inputValue + massParameter_) / (signfunction(inputValue + massParameter_) * pow(inputValue+massParameter_,3.0))
                -1.0 * massParameter_ * (inputValue - 1.0 + massParameter_)/(signfunction(inputValue -1.0 + massParameter_)*pow(inputValue-1.0+massParameter_,3.0)) + thrustAcceleration;

        //        return pow(inputValue, 2.0) -3.0;
//        return pow(inputValue, 5.0) - (3.0 - massParameter_) * pow(inputValue, 4.0)
//               + (3.0 - 2.0 * massParameter_) * pow(inputValue, 3.0) - massParameter_ * pow(inputValue, 2.0)
//               + 2.0 * massParameter_ * inputValue - massParameter_;
    }

    //! Derivatives of mathematical test function.
    double computeDerivative( const unsigned int order, const double inputValue )
    { 
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
            return 1.0 + (1.0 - massParameter_) * 2.0/(signfunction(inputValue+massParameter_)*pow(inputValue+massParameter_, 3.0))
                    + massParameter_ * 2.0/(signfunction(inputValue-1.0+massParameter_)*pow(inputValue-1.0+massParameter_, 3.0));
//            return 2.0 * pow(inputValue, 1.0);
//            return 5.0*pow(inputValue, 4.0) - 4.0*(3.0 - massParameter_)*pow(inputValue, 3.0)
//                   + 3.0*(3.0 - 2.0 * massParameter_) * pow(inputValue, 2.0) - 2.0*massParameter_ * inputValue
//                   + 2.0 * massParameter_;
        }

        else if ( order == 2 )
        {
            // Return the second derivative function value: y = .
            return (1.0 - massParameter_) * -6.0/(signfunction(inputValue+massParameter_)*pow(inputValue+massParameter_, 4.0))
                    + massParameter_ * -6.0/(signfunction(inputValue-1.0+massParameter_)*pow(inputValue-1.0+massParameter_, 4.0));
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
        return 0.8369151483688;
        //return 1.5;
    }
    
    //! Get the accuracy of the true location of the root.
    /*!
     * Returns the accuracy of the true location of the function root, here 1e-15.
     *
     * \return Accuracy of the true location of the root.
     */
    double getTrueRootAccuracy( ) { return 1.0e-3; }

    //! Get a reasonable initial guess of the root location.
    /*!
     * Returns a reasonable initial guess for the true location of the function root, here 4.
     *
     * \return Initial guess for the true location of the function root.
     */
    double getInitialGuess( ) { return 0.8369151483688; }

    //! Get a reasonable lower boundary for the root location.
    /*!
     * Returns a reasonable lower bound for the true location of the function root, here -1.
     *
     * \return Lower bound for the true location of the function root.
     */
    double getLowerBound( ) { return 0.75; }

    //! Get a reasonable upper boundary for the root location.
    /*!
     * Returns a reasonable upper bound for the true location of the function root, here 4.
     *
     * \return Upper bound for the true location of the function root.
     */
    double getUpperBound( ) { return 0.999-massParameter_; }

protected:

    double massParameter_;
private:
};


#endif // TUDATBUNDLE_ARTIFICIALLIBRATIONPOINTLOCATIONFUNCTION1_H
