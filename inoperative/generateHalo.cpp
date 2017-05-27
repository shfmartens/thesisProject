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
#include "thesisProject/stateDerivativeModel.h" //NB: mass parameter needs to be changed here as well.
#include "thesisProject/computeDifferentialCorrection.h"



Eigen::VectorXd generateHalo(Eigen::VectorXd inputVector)

{

using namespace tudat;
using namespace std;
using numerical_integrators::RungeKuttaVariableStepSizeIntegratorXd;
using numerical_integrators::RungeKuttaCoefficients;
namespace crtbp = tudat::gravitation::circular_restricted_three_body_problem;
using namespace root_finders;

    // Declare mass parameter.
    const double GMEarth = tudat::basic_astrodynamics::celestial_body_constants::EARTH_GRAVITATIONAL_PARAMETER;
    const double GMMoon = tudat::basic_astrodynamics::celestial_body_constants::MOON_GRAVITATIONAL_PARAMETER;
    const double massParameter = crtbp::computeMassParameter(GMEarth, GMMoon); // NB: also change in state derivative!;
    cout << "Mass parameter is: " << massParameter << ", based on Earth: " << GMEarth << " and on Moon: " << GMMoon << endl;

    // Declare L2 libration point object with Earth-Moon mass parameter and Newton-Raphson method
    // with 1000 iterations as maximum and 1.0e-14 relative X-tolerance.
    crtbp::LibrationPoint librationPointL2( massParameter, boost::make_shared< NewtonRaphson >( 1.0e-14, 1000 ) );

    //librationPointL2.computeLocationOfLibrationPoint( librationPointL2 );
    librationPointL2.computeLocationOfLibrationPoint( crtbp::LibrationPoint::L2 );

    // Determine location of libration point in Earth-Moon system.
    const Eigen::Vector3d positionOfLibrationPointL2
            = librationPointL2.getLocationOfLagrangeLibrationPoint( );

    // Open a file to write to.
    remove("finalOrbit.txt");
    ofstream textFileTwo("finalOrbit.txt");
    textFileTwo.precision(14);

    // Input x- and z-location and y-velocity.
    double xLocation = inputVector(0);
    double zLocation = inputVector(1);
    double jacobiEnergy = inputVector(2);

    // Create an initial state vector, including the STM.
    const Eigen::VectorXd initialState = createStateVector(xLocation, zLocation, massParameter, jacobiEnergy);
    double xPosition = initialState(0);
    double zPosition = initialState(2);
    double yVelocity = initialState(4);
    Eigen::VectorXd currentState = initialState;
    Eigen::VectorXd previousState = currentState;
    Eigen::VectorXd differenceBetweenStates = previousState;

    // Instantiate an integrator.
    RungeKuttaVariableStepSizeIntegratorXd initialIntegrator ( RungeKuttaCoefficients::get( RungeKuttaCoefficients::rungeKuttaFehlberg78 ), &computeStateDerivative, 0.0, initialState, 1.0e-12, 1.0, 1.0e-13, 1.0e-13);

    // Declare current time, step size and sign of the y-position.
    double currentTime = 0.0;
    double stepSize = 1.0e-4;
    double initialSignYPosition = initialState(4)/fabs(initialState(4));
    double signYPosition = initialSignYPosition;
    double deviation = 1.0;



    // Run simulation loop untill x-axis is crossed again.
    while ((signYPosition < 0.0 && initialSignYPosition < 0.0) || (signYPosition > 0.0 && initialSignYPosition > 0.0)) {

        // Perform integration step.
        currentState = initialIntegrator.performIntegrationStep(stepSize);

        // Compute current time.
        currentTime = initialIntegrator.getCurrentIndependentVariable();

        // Compute the current sign of the y-position.
        signYPosition = currentState(1)/fabs(currentState(1));

        // Compute new stepsize.
        stepSize = initialIntegrator.getNextStepSize();

        // Check if the Y-axis was crossed and if so interpolate to find the values at the crossing.
        if ((signYPosition > 0.0 && initialSignYPosition < 0.0) || (signYPosition < 0.0 && initialSignYPosition > 0.0)) {

            // Linearly approximate all six states.
            if (currentState(1) > previousState(1)) {

                // Compute difference between states.
                differenceBetweenStates = currentState - previousState;

                // Linearly interpolate to y=0.
                currentState = previousState + differenceBetweenStates * (-previousState(1) / differenceBetweenStates(1));

            }

            else {

                // Compute difference between states.
                differenceBetweenStates = previousState - currentState;

                // Linearly interpolate to y=0.
                currentState = currentState + differenceBetweenStates * (-currentState(1) / differenceBetweenStates(1));


            }
        }

        // Save this state to be used as the previous state in the next iteration.
        previousState = currentState;

    }


    // Start correction iteration.
    int iCount = 0;
    while (deviation > 1.0e-9 && iCount < 1000) {

        // Increment counter.
        iCount++;

        // Create differential correction vector.
        Eigen::VectorXd differentialCorrection = computeDifferentialCorrection(currentState);
        xPosition = xPosition + differentialCorrection(0);
        zPosition = zPosition + differentialCorrection(2);
        yVelocity = yVelocity + differentialCorrection(4);
        Eigen::VectorXd tempStateVector(6);
        tempStateVector << xPosition,
                0,
                zPosition,
                0,
                yVelocity,
                0;
        jacobiEnergy = gravitation::circular_restricted_three_body_problem::computeJacobiEnergy(massParameter, tempStateVector);
        Eigen::VectorXd betterInitialState = createStateVector(xPosition, zPosition, massParameter, jacobiEnergy);
        initialSignYPosition = yVelocity/fabs(yVelocity);
        signYPosition = initialSignYPosition;
        betterInitialState(4) = yVelocity;

        // Instantiate an integrator.
        RungeKuttaVariableStepSizeIntegratorXd betterIntegrator ( RungeKuttaCoefficients::get( RungeKuttaCoefficients::rungeKuttaFehlberg78 ), &computeStateDerivative, 0.0, betterInitialState, 1.0e-12, 1.0, 1.0e-12, 1.0e-12);

        // Simulate the orbit untill the y-axis is crossed again.
        while ((signYPosition < 0.0 && initialSignYPosition < 0.0) || (signYPosition > 0.0 && initialSignYPosition > 0.0)) {

            // Perform integration step.
            currentState = betterIntegrator.performIntegrationStep(stepSize);

            // Compute current time.
            currentTime = betterIntegrator.getCurrentIndependentVariable();

            // Compute the current sign of the y-position.
            signYPosition = currentState(1)/fabs(currentState(1));

            // Compute new stepsize.
            stepSize = betterIntegrator.getNextStepSize();

            // Check if the Y-axis was crossed and if so interpolate to find the values at the crossing.
            if ((signYPosition > 0.0 && initialSignYPosition < 0.0) || (signYPosition < 0.0 && initialSignYPosition > 0.0)) {

                // Linearly approximate all six states.
                if (currentState(1) > previousState(1)) {

                    // Compute difference between states.
                    differenceBetweenStates = currentState - previousState;

                    // Linearly interpolate to y=0.
                    currentState = previousState + differenceBetweenStates * (-previousState(1) / differenceBetweenStates(1));

                    // Compute exact time of crossing (interpolated).
                    double previousTime = currentTime - stepSize;
                    currentTime = previousTime + (currentTime-previousTime) * (-previousState(1) / differenceBetweenStates(1));

                }

                else {

                    // Compute difference between states.
                    differenceBetweenStates = previousState - currentState;

                    // Linearly interpolate to y=0.
                    currentState = currentState + differenceBetweenStates * (-currentState(1) / differenceBetweenStates(1));

                    // Compute exact time of crossing (interpolated).
                    double previousTime = currentTime - stepSize;
                    currentTime = previousTime + (currentTime-previousTime) * (-currentState(1) / differenceBetweenStates(1));

                }

                // Compute deviation from perfect periodic orbit.
                deviation = fabs(currentState(3)) + fabs(currentState(5));

            }


            // Save this state to be used as the previous state in the next iteration.
            previousState = currentState;

        }
    }


    // Output period to screen.
    double period = 2.0*currentTime;
    cout << "Period approximately: " << period << endl;

    // Create final initial state.
    Eigen::VectorXd finalInitialState = createStateVector(xPosition, zPosition, massParameter, jacobiEnergy);
    finalInitialState(4) = yVelocity;

    // Write header
    //textFileTwo << "Original orbit 3D with initial x: " << xLocation << ", y: " << initialState(1) << ", z: " << zLocation << " and initial C: " << jacobiEnergy << endl;
    //textFileTwo << "Final C: ; Period: ;" << endl;
    //textFileTwo << left << setw(25) << "X" << setw(25) << "Y" << setw(25) << "Z" << setw(25) << "Xdot" << setw(25) << "Ydot" << setw(25) << "Zdot" << '\n';

    // Instantiate an integrator.
    RungeKuttaVariableStepSizeIntegratorXd finalIntegrator ( RungeKuttaCoefficients::get( RungeKuttaCoefficients::rungeKuttaFehlberg78 ), &computeStateDerivative, 0.0, finalInitialState, 1.0e-12, 1.0, 1.0e-13, 1.0e-13);
    currentTime = 0.0;

    // Simulate the orbit untill the y-axis is crossed again.
    while (currentTime < 1.1 * period) {

        // Perform integration step.
        currentState = finalIntegrator.performIntegrationStep(stepSize);

        // Compute current time.
        currentTime = finalIntegrator.getCurrentIndependentVariable();

        // Compute the current sign of the y-position.
        signYPosition = currentState(1)/fabs(currentState(1));

        // Compute new stepsize.
        stepSize = finalIntegrator.getNextStepSize();

        // Write to file.
        textFileTwo << left << fixed << setw(20) << currentState(0) << setw(20) << currentState(1) << setw(20) << currentState(2) << setw(20) << currentState(3) << setw(20) << currentState(4) << setw(20) << currentState(5) << setw(20) << currentState(6) << setw(20) << currentState(7) << setw(20) << currentState(8) << setw(20) << currentState(9) << setw(20) << currentState(10) << setw(20) << currentState(11) << setw(20) << currentState(12) << setw(20) << currentState(13) << setw(20) << currentState(14) << setw(20) << currentState(15) << setw(20) << currentState(16) << setw(20) << currentState(17) << setw(20) <<  currentState(18) << setw(20) << currentState(19) << setw(20) << currentState(20) << setw(20) << currentState(21) << setw(20) << currentState(22) << setw(20) << currentState(23) << setw(20) << currentState(24) << setw(20) << currentState(25) << setw(20) << currentState(26) << setw(20) << currentState(27) << setw(20) << currentState(28) << setw(20) << currentState(29) << setw(20) << currentState(30) << setw(20) << currentState(31) << setw(20) << currentState(32) << setw(20) << currentState(33) << setw(20) << currentState(34) << setw(20) << currentState(35) << setw(20) << currentState(36) << setw(20) << currentState(37) << setw(20) << currentState(38) << setw(20) << currentState(39) << setw(20) << currentState(40) << setw(20) << currentState(41) << endl;

    }

    // Compute Jacobi energy.
    cout << "Final initial vector " << endl << finalInitialState.segment(0,6) << endl;
    jacobiEnergy = gravitation::circular_restricted_three_body_problem::computeJacobiEnergy(massParameter, finalInitialState.segment(0,6));
    cout << "Jacobi energy C = " << jacobiEnergy << endl;

    // Close file.
    textFileTwo.close();

    // Return latest state.
    return currentState;

} // End of File.
