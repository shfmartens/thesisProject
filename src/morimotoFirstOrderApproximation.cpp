#include <iostream>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>
#include <Eigen/Dense>

#include <cmath>

#include <boost/bind.hpp>
#include <boost/make_shared.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <boost/shared_array.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/shared_container_iterator.hpp>

#include "Tudat/Mathematics/BasicMathematics/mathematicalConstants.h"
#include "Tudat/Astrodynamics/BasicAstrodynamics/celestialBodyConstants.h"
#include "Tudat/Astrodynamics/Gravitation/jacobiEnergy.h"
#include "Tudat/Astrodynamics/Gravitation/librationPoint.h"
#include "Tudat/Mathematics/BasicMathematics/function.h"

#include "Tudat/Mathematics/RootFinders/newtonRaphson.h"
#include "stateDerivativeModelAugmented.h"
#include "morimotoFirstOrderApproximation.h"

#include "createEquilibriumLocations.h"

Eigen::VectorXd computeOffsets( const Eigen::Vector2d equilibriumLocation, const double minimumCenterEigenValue, const int stabilityType, const double amplitude,  const double massParameter, const double currentTime, const int timeParameter )
{
    // initialize variables
    double omegaBeta;
    double correctionBeta;
    double phaseAngleInPlane;
    double timeatPatchPoint;
    Eigen::VectorXd initialOffsets = Eigen::VectorXd::Zero(7);


    // compute the correction for amplitude of the other in-plane component
    double xDistanceSquared = pow(equilibriumLocation(0) + massParameter, 2);
    double xDistanceSquared2 = pow(equilibriumLocation(0) + massParameter - 1.0, 2);
    double yDistanceSquared = pow(equilibriumLocation(1), 2);

    double distanceToPrimary = sqrt(xDistanceSquared + yDistanceSquared);
    double distanceToSecondary = sqrt(xDistanceSquared2 + yDistanceSquared);

    double distanceToPrimaryToTheThirthPower = distanceToPrimary * distanceToPrimary * distanceToPrimary;
    double distanceToySecondaryToTheThirthPower = distanceToSecondary * distanceToSecondary * distanceToSecondary;

    double distanceParameter = massParameter * distanceToySecondaryToTheThirthPower + (1.0 - massParameter ) * distanceToPrimaryToTheThirthPower;


    // For H-L and Halo Type orbits
    if (stabilityType == 1 || stabilityType == 2) {

        // calculate frequency term
        omegaBeta = std::abs(minimumCenterEigenValue);


        // calculate correction term on amplitude
        correctionBeta = (2.0 * omegaBeta) / (omegaBeta * omegaBeta + 2.0 * distanceParameter + 1.0);

        // compute phase angle of the solution via arctan
        //phaseAngleInPlane = atan2(equilibriumLocation(1), (1.0-massParameter)-equilibriumLocation(0));
        phaseAngleInPlane = 0.0;
        //std::cout << "phaseAngleInPlane: " << phaseAngleInPlane << std::endl;

    }

    // Set time to 0.0 at final state
    if (timeParameter == 1 ) {

        timeatPatchPoint = 0.0;

    } else {

        timeatPatchPoint = currentTime;
    }

    // BRAIG & MCINNES Artificial Halo Orbits
    initialOffsets(0) = -correctionBeta * amplitude * std::cos(omegaBeta * timeatPatchPoint + phaseAngleInPlane);
    initialOffsets(1) = amplitude * std::sin(omegaBeta * timeatPatchPoint + phaseAngleInPlane);
    initialOffsets(2) = 0;
    initialOffsets(3) = -1.0 * omegaBeta * correctionBeta * amplitude * std::sin(omegaBeta * timeatPatchPoint + phaseAngleInPlane);
    initialOffsets(4) = omegaBeta * amplitude * std::cos(omegaBeta * timeatPatchPoint + phaseAngleInPlane);
    initialOffsets(5) = 0;
    initialOffsets(6) = 2.0 * tudat::mathematical_constants::PI / ( omegaBeta * omegaBeta );

    return initialOffsets;


}

void computeCenterEigenValues( const Eigen::MatrixXd statePropagationMatrix, double& minimumCenterEigenvalue, int& stabilityType, const double maxEigenvalueDeviation )
{


    // Compute eigenvectors of the monodromy matrix
    Eigen::EigenSolver< Eigen::MatrixXd > eig( statePropagationMatrix );

    //std::cout << "eigenvalues: \n "<<eig.eigenvalues() << std::endl;
    //std::cout << "eigenvectors: \n "<<eig.eigenvectors() << std::endl;

    // initialize counters and vectors for storing the variables
    int numberOfSaddleEigenvalues = 0;
    int numberOfCenterEigenvalues = 0;
    int numberOfCenterInPlaneEigenvalues = 0;
    int numberOfMixedEigenvalues = 0;
    int numberOfZeroEigenvalues = 0;
    double temporaryMinimumCenterEigenvalue = 0;

    // Select the in-plane mixed eigenvalues

    //std::cout << "TEST TEST: \n"<< eig.eigenvectors()(5,5) << std::endl;

    for ( int i = 0; i <= 5; i++ ) {

        // check if the eigenvalue is purely imaginary (Center mode)
        if ( std::abs(eig.eigenvalues().real()(i)) < maxEigenvalueDeviation && std::abs(eig.eigenvalues().imag()(i)) > maxEigenvalueDeviation ) {

            //std::cout << " center mode condition achieved "<< std::endl;
            // check if the eigen Vector is in plane
            if ((std::abs(eig.eigenvectors().imag()(2,i)) + std::abs(eig.eigenvectors().imag()(5,i))) < maxEigenvalueDeviation  ) {

                //std::cout << " in plane condition "<< std::endl;
                //std::cout << " eig.eigenvalues().imag()(i)" << eig.eigenvalues().imag()(i) << std::endl;

                if (numberOfCenterEigenvalues == 0 ) {

                    temporaryMinimumCenterEigenvalue = eig.eigenvalues().imag()(i);

                } else if ( eig.eigenvalues().imag()(i) < temporaryMinimumCenterEigenvalue ){
                    temporaryMinimumCenterEigenvalue = eig.eigenvalues().imag()(i);
                }

                numberOfCenterInPlaneEigenvalues++;
            }

            numberOfCenterEigenvalues++;
        }
        // check if the eigenvalue is purely real (Saddle mode)
        if ( std::abs(eig.eigenvalues().real()(i)) > maxEigenvalueDeviation && std::abs(eig.eigenvalues().imag()(i)) < maxEigenvalueDeviation) {

            numberOfSaddleEigenvalues++;
        }

        // check if the eigenvalue is purely zero (No mode)
        if ( std::abs(eig.eigenvalues().real()(i)) < maxEigenvalueDeviation && std::abs(eig.eigenvalues().imag()(i)) < maxEigenvalueDeviation) {

            numberOfZeroEigenvalues++;
        }

        // check if the eigenvalue is Mixed (Mixed mode)
        if ( std::abs(eig.eigenvalues().real()(i)) > maxEigenvalueDeviation && std::abs(eig.eigenvalues().imag()(i)) > maxEigenvalueDeviation) {

            numberOfMixedEigenvalues++;
        }

    }

    minimumCenterEigenvalue = temporaryMinimumCenterEigenvalue;

//            std::cout << "===" <<std::endl
//                      << "numberOfMixedEigenValues: " << numberOfMixedEigenvalues << std::endl
//                      << "numberOfSaddleEigenValues: " << numberOfSaddleEigenvalues << std::endl
//                      << "numberOfCenterEigenValues: " << numberOfCenterEigenvalues << std::endl
//                      << "numberOfCenterInPlaneEigenValues: " << numberOfCenterInPlaneEigenvalues << std::endl
//                      << "temporaryMinimunCenterValue: " << temporaryMinimumCenterEigenvalue << std::endl
//                      << "===" <<std::endl;

    if ( numberOfCenterEigenvalues == 4 ){

        // alpha is saddle (in plane)
        // beta is center (in plane )
        // out-of plane is center
        stabilityType = 1;


    } else if (numberOfCenterEigenvalues == 6 ){
        // alpha is center (in plane)
        // beta is center (in plane )
        // out-of plane is center
        stabilityType = 2;
    } else if ( numberOfSaddleEigenvalues == 2 ){
        // alpha is saddle (in plane)
        // beta is saddle (in plane )
        // out-of plane is center

        stabilityType = 3;
    }


}

Eigen::VectorXd morimotoFirstOrderApproximation(int librationPointNr,
                                                  double amplitude, double thrustMagnitude, double accelerationAngle, double accelerationAngle2, const double initialMass, const int numberOfPatchPoints )
{
    std::cout << "\nCreate initial conditions:" << std::endl
              << "Ax: " << amplitude << std::endl;

    // Set output precision and clear screen.
    std::cout.precision(14);

    // Compute the mass parameter of the Earth-Moon system
    const double primaryGravitationalParameter = tudat::celestial_body_constants::EARTH_GRAVITATIONAL_PARAMETER;
    const double secondaryGravitationalParameter = tudat::celestial_body_constants::MOON_GRAVITATIONAL_PARAMETER;
    const double massParameter = tudat::gravitation::circular_restricted_three_body_problem::computeMassParameter( primaryGravitationalParameter, secondaryGravitationalParameter );

    // Compute location of the artificial equilibrium point
    Eigen::Vector2d equilibriumLocation = createEquilibriumLocations( librationPointNr, thrustMagnitude, accelerationAngle );
    //Eigen::Vector2d equilibriumLocation = Eigen::Vector2d::Zero(2);
    std::cout << "equilibriumLocation: \n" << equilibriumLocation << std::endl;

    // Compute the SPM at the equilibrium location
    Eigen::MatrixXd fullStateVectorEquilibrium = Eigen::MatrixXd::Zero( 10, 11 );
    Eigen::MatrixXd statePropagationMatrix        = Eigen::MatrixXd::Zero( 10, 11 );

    fullStateVectorEquilibrium.block(0,0,2,1) = equilibriumLocation;
    fullStateVectorEquilibrium.block(0,1,10,10).setIdentity();
    statePropagationMatrix = computeStateDerivativeAugmented(0.0, fullStateVectorEquilibrium);

    //std::cout << "SPM: \n" << statePropagationMatrix << std::endl;

    // Compute the eigenvalues of the upper 6x6 block of the matrix.
    int stabilityType;
    double minimumCenterEigenValue;

    computeCenterEigenValues( statePropagationMatrix.block(0,1,6,6), minimumCenterEigenValue, stabilityType);

    std::cout << "stabilityType:  " << stabilityType << std::endl;
    std::cout << "minimumCenterEigenvalue:  " << minimumCenterEigenValue << std::endl;

    // Compute the offset in position and velocity w.r.t to the equilibrium point for all patch points.

    Eigen::VectorXd initialOffsets = Eigen::VectorXd::Zero(6);
    Eigen::VectorXd stateAtCurrentPatchPoint = Eigen::VectorXd::Zero(11);
    Eigen::VectorXd lowThrustInitialStateVectorGuess = Eigen::VectorXd::Zero(11*numberOfPatchPoints);

    double linearizedOrbitalPeriod = 2.0 * tudat::mathematical_constants::PI / (abs(minimumCenterEigenValue));

    std::cout << "2 * PI: " << 2.0 * tudat::mathematical_constants::PI << std::endl;
    std::cout << "linearizedOrbitalPeriod: " << linearizedOrbitalPeriod << std::endl;
    std::cout << "absMINIMUMCenterEigenValue: " <<  (abs(minimumCenterEigenValue)) << std::endl;


    for (int i = 0; i < numberOfPatchPoints; i++){

        initialOffsets.setZero();
        stateAtCurrentPatchPoint.setZero();

        // compute current time at patch point
        auto periodVariable = static_cast<double>(i);
        auto periodVariable2 = static_cast<double>(numberOfPatchPoints);
        double currentTime = periodVariable * linearizedOrbitalPeriod / (periodVariable2 - 1.0);
        int timeParameter;

        //std::cout << "periodVariable: " << periodVariable << std::endl;
        //std::cout << "periodVariable2: " << periodVariable2 << std::endl;
        //std::cout << "currentTime: " << currentTime << std::endl;
        //std::cout << "HALF PERIOD CHECK: " << tudat::mathematical_constants::PI - (linearizedOrbitalPeriod / 2.0 * abs(minimumCenterEigenValue) )<< std::endl;
        //std::cout << "HALF PERIOD CHECK: " << cos(tudat::mathematical_constants::PI - (linearizedOrbitalPeriod / 2.0 * abs(minimumCenterEigenValue) ))<< std::endl;


        // Set final time to 0.0 instead of orbital period to achieve identical end and beginning state
        // thereby avoiding numerical errors
        if (i == numberOfPatchPoints - 1){

            timeParameter = 1;
        } else {

            timeParameter = 0;
        }

        //std::cout << "currentTime: "<< currentTime << std::endl;
        //std::cout << "linearizedOrbitalPeriod: " << linearizedOrbitalPeriod << std::endl;

        // compute offsets w.r.t equilibirum point at current time
        initialOffsets = computeOffsets( equilibriumLocation, minimumCenterEigenValue, stabilityType, amplitude, massParameter, currentTime, timeParameter );

        // Fill stateAtCurrentPatchPoint vector
        stateAtCurrentPatchPoint(0) = equilibriumLocation(0) + initialOffsets(0);
        stateAtCurrentPatchPoint(1) = equilibriumLocation(1) + initialOffsets(1);
        stateAtCurrentPatchPoint(2) = initialOffsets(2);
        stateAtCurrentPatchPoint(3) =  initialOffsets(3);
        stateAtCurrentPatchPoint(4) = initialOffsets(4);
        stateAtCurrentPatchPoint(5) = initialOffsets(5);
        stateAtCurrentPatchPoint(6) = thrustMagnitude;
        stateAtCurrentPatchPoint(7) = accelerationAngle;
        stateAtCurrentPatchPoint(8) = accelerationAngle2;
        stateAtCurrentPatchPoint(9) = initialMass;
        stateAtCurrentPatchPoint(10) = currentTime;

        //Add current state to complete initial guess

        //std::cout << "stateAtCurrentPatchPoint: \n"<<stateAtCurrentPatchPoint << std::endl;

        lowThrustInitialStateVectorGuess.segment(i * 11,11) = stateAtCurrentPatchPoint;

    }



    return lowThrustInitialStateVectorGuess;
}

