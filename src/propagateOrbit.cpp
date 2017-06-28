/*    Copyright (c) 2010-2014, Delft University of Technology
 *    All rights reserved.
 *
 *    Redistribution and use in source and binary forms, with or without modification, are
 *    permitted provided that the following conditions are met:
 *      - Redistributions of source code must retain the above copyright notice, this list of
 *        conditions and the following disclaimer.
 *      - Redistributions in binary form must reproduce the above copyright notice, this list of
 *        conditions and the following disclaimer in the documentation and/or other materials
 *        provided with the distribution.
 *      - Neither the name of the Delft University of Technology nor the names of its contributors
 *        may be used to endorse or promote products derived from this software without specific
 *        prior written permission.
 *
 *    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS
 *    OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 *    MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *    COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 *    EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
 *    GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 *    AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 *    NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
 *    OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *    Changelog
 *      YYMMDD    Author            Comment
 *      140401    J. R�hner         First creation of code.
 *
 *    References
 *    - None.
 *
 *    Notes
 *    - None.
 *
 */

// Include-statements.
#include "Tudat/Mathematics/NumericalIntegrators/rungeKuttaVariableStepSizeIntegrator.h"
#include "Tudat/Mathematics/NumericalIntegrators/rungeKuttaCoefficients.h"
#include "Tudat/Mathematics/RootFinders/newtonRaphson.h"
#include "Tudat/Astrodynamics/Gravitation/jacobiEnergy.h"
#include "Tudat/Astrodynamics/BasicAstrodynamics/celestialBodyConstants.h"
#include "Tudat/Astrodynamics/Gravitation/librationPoint.h"

#include "propagateOrbit.h"
#include "stateDerivativeModel.h"


// Begin function.
Eigen::VectorXd propagateOrbit(Eigen::VectorXd inputState, double massParameter, double halfPeriodFlag,
                               double direction, std::string orbit_type) {

// Declare namespaces.
using namespace tudat;
using namespace std;
using numerical_integrators::RungeKuttaVariableStepSizeIntegratorXd;
using numerical_integrators::RungeKuttaCoefficients;
namespace crtbp = tudat::gravitation::circular_restricted_three_body_problem;
using namespace root_finders;

    // Declare state vector to be returned, a vector to store the previous state vector and a stepsize parameter.
    Eigen::VectorXd outputVector(43);
    Eigen::VectorXd outputState = inputState;
    Eigen::VectorXd previousOutputState = outputState;
    double stepSize = 1.0e-4;
    double currentTime = 0.0;
    double previousTime = currentTime;
    double stateIdx;

    if (orbit_type == "halo"){
        stateIdx = 1;
    }
    if(orbit_type == "near_vertical"){
        stateIdx = 2;
    }

    // Create integrator to be used for propagating.
    RungeKuttaVariableStepSizeIntegratorXd orbitIntegrator ( RungeKuttaCoefficients::get( RungeKuttaCoefficients::rungeKuttaFehlberg78 ), &computeStateDerivative, 0.0, inputState, 1.0e-12, 1.0, 1.0e-13, 1.0e-13);

    // Perform integration until either the half-period point is reached, or a full period has passed.
    if (halfPeriodFlag == 0.5) {
        while (true) {

            // Perform integration step and get next stepSize.
            previousOutputState = outputState;
            outputState = orbitIntegrator.performIntegrationStep(stepSize);
            stepSize = orbitIntegrator.getNextStepSize();
            previousTime = currentTime;
            currentTime = orbitIntegrator.getCurrentIndependentVariable();

//            cout << outputState.segment( 0, 6 ) << "\n" << endl;

            // Check if half-period is reached. This is the case when the sign of the y-location has changed.
            if ( outputState(stateIdx) / fabs( outputState(stateIdx) ) == - inputState(stateIdx + 3) / fabs( inputState(stateIdx + 3) ) ) {

                // Linearly approximate all six states at the exact half period point.
                if (outputState(stateIdx) > previousOutputState(stateIdx)) {

                    // Linearly interpolate to y=0.
                    outputState = previousOutputState +
                            ( outputState - previousOutputState ) * ( -previousOutputState(stateIdx) /
                                    ( outputState(stateIdx) - previousOutputState(stateIdx) ) );
                    currentTime = previousTime +
                            ( currentTime - previousTime ) * ( -previousOutputState(stateIdx) /
                                    ( outputState(stateIdx) - previousOutputState(stateIdx) ) );
                    break;
                }

                else {

                    // Linearly interpolate to y=0.
                    outputState = outputState + ( outputState - previousOutputState ) * ( -outputState(stateIdx) /
                            ( outputState(stateIdx) - previousOutputState(stateIdx) ) );
                    currentTime = previousTime + ( currentTime - previousTime ) * ( -outputState(stateIdx) /
                            ( outputState(stateIdx) - previousOutputState(stateIdx) ) );
                    break;
                }
            }
        }
    }

    else {
        if (direction > 0.0) {
            Eigen::VectorXd tempState = orbitIntegrator.performIntegrationStep(stepSize);
            stepSize = orbitIntegrator.getNextStepSize();
            orbitIntegrator.rollbackToPreviousState();
            outputState = orbitIntegrator.performIntegrationStep(stepSize);
            currentTime = halfPeriodFlag + orbitIntegrator.getCurrentIndependentVariable();
        }
        else {
            Eigen::VectorXd tempState = orbitIntegrator.performIntegrationStep(-stepSize);
            stepSize = orbitIntegrator.getNextStepSize();
            orbitIntegrator.rollbackToPreviousState();
            outputState = orbitIntegrator.performIntegrationStep(stepSize);
            currentTime = halfPeriodFlag + orbitIntegrator.getCurrentIndependentVariable();
        }
    }

    // Return the value of the state and the halfPeriod time.
    outputVector.segment(0,42) = outputState;
    outputVector(42) = currentTime;
    //cout << "Time: " << outputVector(42) << endl;
    return outputVector;
}
