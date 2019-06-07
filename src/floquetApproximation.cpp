#include <iostream>
#include <typeinfo>


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
#include "floquetApproximation.h"
#include "applyPredictionCorrection.h"


#include "createEquilibriumLocations.h"
#include "propagateOrbitAugmented.h"


Eigen::VectorXd floquetApproximation(int librationPointNr, std::string orbitType,
                                                  double amplitude, double thrustMagnitude, double accelerationAngle, double accelerationAngle2, const double initialMass, const int numberOfPatchPoints, const double maxEigenValueDeviation )
{
    std::cout << "\nCreate initial conditions:" << std::endl
              << "Ax: " << amplitude << std::endl;

    Eigen::VectorXd lowThrustInitialStateVectorGuess(11*numberOfPatchPoints);
    lowThrustInitialStateVectorGuess.setZero();

//    std::cout << "\n===check inputs ===" << std::endl
//              << "libPointNr: " << librationPointNr << std::endl
//              << "orbitType: " << orbitType << std::endl
//              << "amplitude: " << amplitude << std::endl
//              << "thrustMagnitude: " << thrustMagnitude << std::endl
//              << "alpha: " << accelerationAngle << std::endl
//              << "beta: " << accelerationAngle2 << std::endl
//              << "initialMass: " << initialMass << std::endl
//              << "patchpoints: " << numberOfPatchPoints << std::endl
//              << "maxEigenValueDeviation: " << maxEigenValueDeviation << std::endl
//              << " === check inputs completed === \n" << std::endl;

    // Set output precision and clear screen.
    std::cout.precision(14);

    // Compute the mass parameter of the Earth-Moon system
    const double primaryGravitationalParameter = tudat::celestial_body_constants::EARTH_GRAVITATIONAL_PARAMETER;
    const double secondaryGravitationalParameter = tudat::celestial_body_constants::MOON_GRAVITATIONAL_PARAMETER;
    const double massParameter = tudat::gravitation::circular_restricted_three_body_problem::computeMassParameter( primaryGravitationalParameter, secondaryGravitationalParameter );

    // Compute location of the artificial equilibrium point
    Eigen::Vector2d equilibriumLocation = createEquilibriumLocations( librationPointNr, thrustMagnitude, accelerationAngle );
    Eigen::VectorXd equilibriumStateVector =   Eigen::VectorXd::Zero( 10 );
    equilibriumStateVector.segment(0,2) = equilibriumLocation;
    equilibriumStateVector(6) = thrustMagnitude;
    equilibriumStateVector(7) = accelerationAngle;
    equilibriumStateVector(8) = accelerationAngle2;
    equilibriumStateVector(9) = initialMass;

    std::cout << "fullStateVectorEquilibirum: \n" << equilibriumStateVector << std::endl;

    // Provide an offset in the direction of the minimum in-plane center eigenvalue of the state propagation matrix
    Eigen::MatrixXd stateDerivativeInclSPM = Eigen::MatrixXd::Zero(10,11);
    Eigen::MatrixXd statePropagationMatrix = Eigen::MatrixXd::Zero(6,6);

    stateDerivativeInclSPM = computeStateDerivativeAugmented( 0.0, getFullInitialStateAugmented( equilibriumStateVector) );
    statePropagationMatrix = stateDerivativeInclSPM.block(0,1,6,6);

    Eigen::EigenSolver< Eigen::MatrixXd > eigSPM( statePropagationMatrix );
    //std::cout << "\neigenvalues SPM: \n" << eigSPM.eigenvalues() << std::endl;
    //std::cout << "eigenvectors SPM: \n" << eigSPM.eigenvectors() << std::endl;

    int indexEigenValue;
    std::complex<double> centerEigenValue;
    Eigen::VectorXcd centerEigenVector = Eigen::VectorXd::Zero(6);
    Eigen::VectorXd centerEigenVectorModulus = Eigen::VectorXd::Zero(6);
    Eigen::VectorXd centerEigenVectorReal = Eigen::VectorXd::Zero(6);


    for (int i = 0; i < 6; i++ )
    {
        // check if eigen value is imaginary
        if (std::abs(eigSPM.eigenvalues()(i).real()) < maxEigenValueDeviation and std::abs(eigSPM.eigenvalues()(i).imag()) > maxEigenValueDeviation )
        {

            // depending on orbitType, select the in-plane or out-of-plane eigenVector corresponding to the positive eigenvalue
            if (orbitType == "horizontal" && std::abs( eigSPM.eigenvectors()(2,i) + eigSPM.eigenvectors()(5,i) ) < maxEigenValueDeviation && eigSPM.eigenvalues()(i).imag() > 0)
            {
                indexEigenValue = i;
                centerEigenValue = eigSPM.eigenvalues()(i);
                centerEigenVector= eigSPM.eigenvectors().block(0,i,6,1);

                for (int k = 0; k < 6; k++)
                {
                     centerEigenVectorModulus(k) = std::abs( eigSPM.eigenvectors()(k,i))   ;
                     centerEigenVectorReal(k) = eigSPM.eigenvectors()(k,i).real()   ;
                     if (k ==2 or k==5)
                     {
                        centerEigenVectorReal(k) = 0.0;
                     }


                }
            }

            if (orbitType == "vertical" && std::abs( eigSPM.eigenvectors()(2,i) + eigSPM.eigenvectors()(5,i) ) > maxEigenValueDeviation && eigSPM.eigenvalues()(i).imag() > 0)
            {
                indexEigenValue = i;
                centerEigenValue = eigSPM.eigenvalues()(i);
                centerEigenVector= eigSPM.eigenvectors().block(0,i,6,1);


                for (int k = 0; k < 6; k++)
                {
                     centerEigenVectorReal(k) = eigSPM.eigenvectors()(k,i).real();

                }

            }


        }

    }

    Eigen::VectorXd initialStateAfterOffset = Eigen::VectorXd::Zero(10);
    double normalizationFactor;

    if (orbitType == "horizontal")
    {

        normalizationFactor = std::abs( 1.0 /(centerEigenVectorReal.segment(0,3).norm()) );


    } else
    {
        normalizationFactor = std::abs( 1.0 /(centerEigenVectorReal.segment(0,3).norm()) );


    }

    initialStateAfterOffset.segment(0,6) = equilibriumStateVector.segment(0,6) + normalizationFactor * amplitude * centerEigenVectorReal;
    initialStateAfterOffset.segment(6,4) = equilibriumStateVector.segment(6,4);

    std::cout << "\ninitialStateAfterOffset: \n"<<  initialStateAfterOffset << std::endl;

    double linearizedOrbitalPeriod = 2.0 * tudat::mathematical_constants::PI / (std::abs(centerEigenValue));

    std::cout << "linearizedOrbitalPeriod: " << linearizedOrbitalPeriod << std::endl;

    double initialTime = 0.0;
    double finalTime = 0.0;
    double currentTime = 0.0;
    Eigen::VectorXd initialStateVector = initialStateAfterOffset;

    std::map< double, Eigen::VectorXd > stateHistoryInitialGuess;
    for (int i = 0; i <= (numberOfPatchPoints -2); i++){

        // compute current time at patch point
        auto periodVariable = static_cast<double>(i);
        auto periodVariableFinal = static_cast<double>(i+1);
        auto periodVariable2 = static_cast<double>(numberOfPatchPoints);
        double initialTime = periodVariable * linearizedOrbitalPeriod / (periodVariable2 - 1.0);
        double currentfinalTime =   periodVariableFinal * linearizedOrbitalPeriod / (periodVariable2 - 1.0);

        // create final Time
        finalTime = currentfinalTime;

        std::pair< Eigen::MatrixXd, double > finalTimeState = propagateOrbitAugmentedToFinalCondition( getFullInitialStateAugmented( initialStateVector),
                                                                 massParameter, finalTime, 1, stateHistoryInitialGuess, 1000, initialTime);

        Eigen::MatrixXd stateVectorInclSTM      = finalTimeState.first;
        double currentTime             = finalTimeState.second;
        Eigen::VectorXd stateVectorOnly = stateVectorInclSTM.block( 0, 0, 10, 1 );

        initialTime = currentTime;


        initialStateVector = stateVectorOnly;

        if (i == 0  )
        {

            lowThrustInitialStateVectorGuess.segment(0,10) = initialStateAfterOffset;
            lowThrustInitialStateVectorGuess(10) = 0.0;

        } else if (i == ( numberOfPatchPoints - 2 ) )
        {
            lowThrustInitialStateVectorGuess.segment(11*(i+1),10) = initialStateAfterOffset;
            lowThrustInitialStateVectorGuess((11*(i+1))+10) = currentTime;

        }
        {

            lowThrustInitialStateVectorGuess.segment(11*(i+1),10) = stateVectorOnly;
            lowThrustInitialStateVectorGuess((11*(i+1))+10) = currentTime;

        }

    }

    std::map< double, Eigen::VectorXd > stateHistoryCorrectedGuess;

    if ( (amplitude  < 1.01E-5) or (amplitude > 2.7E-5 and amplitude < 2.9E-5) or  (amplitude > 4.5E-5 and amplitude < 4.7E-5)
         or (amplitude > 6.3E-5 and amplitude < 6.5E-5) or (amplitude > 8.1E-5 and amplitude < 8.3E-5) or amplitude > 9.99E-5)
    {

        Eigen::VectorXd differentialCorrectionResults = applyPredictionCorrection(librationPointNr, lowThrustInitialStateVectorGuess, 0.0, massParameter, numberOfPatchPoints,
                                                                                  false, 1.0E-12, 1.0E-12, 1.0E-12);

        std::pair< Eigen::MatrixXd, double > finalTimeState = propagateOrbitAugmentedToFinalCondition( getFullInitialStateAugmented( differentialCorrectionResults.segment(0,10)),
                                                                 massParameter, differentialCorrectionResults(10), 1, stateHistoryCorrectedGuess, 1000, initialTime);

    }

    writeFloquetDataToFile( stateHistoryInitialGuess, stateHistoryCorrectedGuess, librationPointNr, orbitType, equilibriumStateVector, numberOfPatchPoints, amplitude);


    return lowThrustInitialStateVectorGuess;
}

