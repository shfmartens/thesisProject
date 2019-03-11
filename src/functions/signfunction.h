/*    Copyright (c) 2010-2017, Delft University of Technology
 *    All rigths reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 *
 *    References
 *      Milea, A. Virtual Inheritance in C++, and solving the diamond problem,
 *          http://www.cprogramming.com/tutorial/virtual_inheritance.html, 1997-2011,
 *          last accessed: 9th September, 2012.
 *
 *    Notes
 *      You need to define what function implementation you use yourself to avoid the Diamond
 *      Problem (Milea, 2011).
 *
 */

#ifndef TUDATBUNDLE_SIGNFUNCTION_H
#define TUDATBUNDLE_SIGNFUNCTION_H

#include "Tudat/Mathematics/BasicMathematics/basicFunction.h"


//! Simple definition of a test function, so that it can be used by all root-finder unit tests.
double signfunction (const double inputValue) {

    double outputValue;

   if (inputValue < 0.0 ){
      outputValue = -1.0;
    } else {
      outputValue = 1.0;
  }
   return outputValue;
};

#endif // TUDATBUNDLE_SIGNFUNCTION_H
