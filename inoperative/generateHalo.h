#ifndef GENERATEHALO_H
#define GENERATEHALO_H

// Include-statements
#include <iostream>
#include <iomanip>
#include <fstream>
#include <Eigen/Core>
#include <Eigen/LU>
#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>

#include "Tudat/Mathematics/NumericalIntegrators/rungeKuttaVariableStepSizeIntegrator.h"
#include "Tudat/Mathematics/NumericalIntegrators/rungeKuttaCoefficients.h"
#include "Tudat/Mathematics/RootFinders/newtonRaphson.h"
#include "Tudat/Astrodynamics/Gravitation/jacobiEnergy.h"
#include "Tudat/Astrodynamics/BasicAstrodynamics/celestialBodyConstants.h"
#include "Tudat/Astrodynamics/Gravitation/librationPoint.h"

#include "thesisProject/createStateVector.h"
#include "stateDerivativeModel.h" //NB: mass parameter needs to be changed here as well.
#include "thesisProject/computeDifferentialCorrection.h"
#include "generateHalo.cpp"





#endif // GENERATEHALO_H
