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

#ifndef TUDATBUNDLE_lEGENDREGAUSSLOBATTOFUNCTION_H
#define TUDATBUNDLE_lEGENDREGAUSSLOBATTOFUNCTION_H

#include <cmath>
#include <iostream>
#include <stdexcept>
#include "Tudat/Mathematics/BasicMathematics/basicFunction.h"
#include "Eigen/Core"
#include "legendreGaussLobatto.h"



struct LegendreGaussLobattoFunction : public LegendreGaussLobatto,
        public tudat::basic_mathematics::BasicFunction< double, double >
{

    Eigen::VectorXd getNormalizedNodeTimes( )
    {
        Eigen::VectorXd normalizedNodeTimes = Eigen::VectorXd::Zero(7);
        normalizedNodeTimes(0) = 0.0;
        normalizedNodeTimes(3) = 0.5;
        normalizedNodeTimes(6) = 1.0;

        return normalizedNodeTimes;

    }



protected:

private:
};


#endif // LegendreGaussLobattoFunction
