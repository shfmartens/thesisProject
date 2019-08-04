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

void computeMotionDecomposition(const int librationPointNr, const std::string orbitType, Eigen::MatrixXd statePropagationMatrix, Eigen::MatrixXd stateTransitionMatrix, Eigen::VectorXd initialPerturbationVector, const double perturbationTime, const double numericalThreshold )
{
    Eigen::VectorXd computedCorrection = Eigen::VectorXd::Zero(3);

    //  ==== Decompose the motion into the six modes ==== //

    // Compute the modal matrix
    Eigen::MatrixXcd modalMatrix = Eigen::MatrixXcd::Zero(6,6);
    Eigen::EigenSolver< Eigen::MatrixXd > eigSPM( statePropagationMatrix );
    Eigen::MatrixXcd eigenVectorsMatrix = eigSPM.eigenvectors();
    Eigen::VectorXcd  eigenValues = eigSPM.eigenvalues();
    Eigen::MatrixXcd floquetExponentMatrix = Eigen::MatrixXcd::Zero(6,6);

    for (int i = 0; i < 6; i++)
    {
        floquetExponentMatrix(i,i) = std::exp(eigenValues(i) * perturbationTime);
    }

    modalMatrix = stateTransitionMatrix * eigenVectorsMatrix * floquetExponentMatrix;

    // Compute current perturbations
    Eigen::VectorXd perturbationVector = stateTransitionMatrix*initialPerturbationVector;

    for (int i = 0; i < 6; i++)
    {
        for (int j = 0; j < 6;  j++)
        {
            if (std::abs(modalMatrix(i,j).real()) < numericalThreshold)
            {
                std::complex<double> replacement(0,modalMatrix(i,j).imag());
                modalMatrix(i,j) = replacement;
            }

            if (std::abs(modalMatrix(i,j).imag()) < numericalThreshold)
            {
                std::complex<double> replacement(modalMatrix(i,j).real(),0);
            modalMatrix(i,j) = replacement;
            }
        }
    }

    Eigen::VectorXcd perturbationCoefficients = modalMatrix.inverse() * perturbationVector;

    for (int i = 0; i < 6; i++)
    {
        for (int j = 0; j < 6;  j++)
        {
            if (std::abs(perturbationCoefficients(i,j).real()) < numericalThreshold)
            {
                std::complex<double> replacement(0,perturbationCoefficients(i,j).imag());
                perturbationCoefficients(i,j) = replacement;
            }

            if (std::abs(perturbationCoefficients(i,j).imag()) < numericalThreshold)
            {
                std::complex<double> replacement(perturbationCoefficients(i,j).real(),0);
            perturbationCoefficients(i,j) = replacement;
            }
        }
    }

    Eigen::MatrixXcd perturbationDecomposition(6,6);

    for (int i = 0; i < 6; i ++)
    {
        perturbationDecomposition.block(0,i,6,1) = perturbationCoefficients(i) * modalMatrix.block(0,i,6,1);
    }

    //std::cout << "Motion Decomposition: \n" << perturbationDecomposition << std::endl;
}

Eigen::VectorXd computeVelocityCorrection(const int librationPointNr, const std::string orbitType, Eigen::MatrixXd statePropagationMatrix, Eigen::MatrixXd stateTransitionMatrix, Eigen::VectorXd initialPerturbationVector, const double perturbationTime, const double numericalThreshold )
{
    Eigen::VectorXd computedCorrection = Eigen::VectorXd::Zero(3);

    //  ==== Decompose the motion into the six modes ==== //

    // Compute the modal matrix
    Eigen::MatrixXcd modalMatrix = Eigen::MatrixXcd::Zero(6,6);
    Eigen::EigenSolver< Eigen::MatrixXd > eigSPM( statePropagationMatrix );
    Eigen::MatrixXcd eigenVectorsMatrix = eigSPM.eigenvectors();
    Eigen::VectorXcd  eigenValues = eigSPM.eigenvalues();
    Eigen::MatrixXcd floquetExponentMatrix = Eigen::MatrixXcd::Zero(6,6);

    for (int i = 0; i < 6; i++)
    {
        floquetExponentMatrix(i,i) = std::exp(eigenValues(i) * perturbationTime);
    }

    modalMatrix = stateTransitionMatrix * eigenVectorsMatrix * floquetExponentMatrix;

    // Compute current perturbations
    Eigen::VectorXd perturbationVector = stateTransitionMatrix*initialPerturbationVector;

    for (int i = 0; i < 6; i++)
    {
        for (int j = 0; j < 6;  j++)
        {
            if (std::abs(modalMatrix(i,j).real()) < numericalThreshold)
            {
                std::complex<double> replacement(0,modalMatrix(i,j).imag());
                modalMatrix(i,j) = replacement;
            }

            if (std::abs(modalMatrix(i,j).imag()) < numericalThreshold)
            {
                std::complex<double> replacement(modalMatrix(i,j).real(),0);
            modalMatrix(i,j) = replacement;
            }
        }
    }

    Eigen::VectorXcd perturbationCoefficients = modalMatrix.inverse() * perturbationVector;

    for (int i = 0; i < 6; i++)
    {
        for (int j = 0; j < 6;  j++)
        {
            if (std::abs(perturbationCoefficients(i,j).real()) < numericalThreshold)
            {
                std::complex<double> replacement(0,perturbationCoefficients(i,j).imag());
                perturbationCoefficients(i,j) = replacement;
            }

            if (std::abs(perturbationCoefficients(i,j).imag()) < numericalThreshold)
            {
                std::complex<double> replacement(perturbationCoefficients(i,j).real(),0);
            perturbationCoefficients(i,j) = replacement;
            }
        }
    }

    Eigen::MatrixXcd perturbationDecomposition(6,6);

    for (int i = 0; i < 6; i ++)
    {
        perturbationDecomposition.block(0,i,6,1) = perturbationCoefficients(i) * modalMatrix.block(0,i,6,1);
    }

    if (orbitType == "horizontal")
    {

        // Compute the required corrections
        Eigen::MatrixXcd updateMatrix(4,4);
        Eigen::VectorXcd deviationVector(4,1);
        Eigen::MatrixXcd correctionVector(4,1);

        updateMatrix.setZero();
        deviationVector.setZero();
        correctionVector.setZero();


        updateMatrix.block(0,0,2,1) = perturbationDecomposition.block(0,2,2,1);
        updateMatrix.block(2,0,2,1) = perturbationDecomposition.block(3,2,2,1);
        updateMatrix.block(0,1,2,1) = perturbationDecomposition.block(0,3,2,1);
        updateMatrix.block(2,1,2,1) = perturbationDecomposition.block(3,3,2,1);
        updateMatrix(2,2) = -1.0;
        updateMatrix(3,3) = -1.0;

        deviationVector(0) = perturbationDecomposition(0,0) + perturbationDecomposition(0,1);
        deviationVector(1) = perturbationDecomposition(1,0) + perturbationDecomposition(1,1);
        deviationVector(2) = perturbationDecomposition(3,0) + perturbationDecomposition(3,1);
        deviationVector(3) = perturbationDecomposition(4,0) + perturbationDecomposition(4,1);

        correctionVector = updateMatrix.inverse()*deviationVector;

        for (int i = 0; i < 4; i++)
        {

           if (std::abs(correctionVector(i).real()) < numericalThreshold)
              {
                 std::complex<double> replacement(0,correctionVector(i).imag());
                 correctionVector(i) = replacement;
              }

           if (std::abs(correctionVector(i).imag()) < numericalThreshold)
              {
                  std::complex<double> replacement(correctionVector(i).real(),0);
                  correctionVector(i) = replacement;
              }
        }

        // Fill the output vector
        computedCorrection(0) = correctionVector(2).real();
        computedCorrection(1) = correctionVector(3).real();

    }
    if (orbitType == "vertical")
    {

        // Compute the required corrections
        Eigen::MatrixXcd updateMatrix(2,3);
        Eigen::VectorXcd deviationVector(2,1);
        Eigen::MatrixXcd correctionVector(3,1);

        updateMatrix.setZero();
        deviationVector.setZero();
        correctionVector.setZero();

        updateMatrix(0,0) = perturbationDecomposition(2,4);
        updateMatrix(1,0) = perturbationDecomposition(5,4);
        updateMatrix(0,1) = perturbationDecomposition(2,4);
        updateMatrix(1,1) = perturbationDecomposition(5,4);

        updateMatrix(1,2) = -1.0;

        std::cout << "\nvertical update MATRIX: \n" << updateMatrix <<std::endl;

        correctionVector = (updateMatrix.transpose() ) * ( updateMatrix*updateMatrix.transpose() )*deviationVector;

        for (int i = 0; i < 6; i++)
        {

           if (std::abs(correctionVector(i).real()) < numericalThreshold)
              {
                 std::complex<double> replacement(0,correctionVector(i).imag());
                 correctionVector(i) = replacement;
              }

           if (std::abs(correctionVector(i).imag()) < numericalThreshold)
              {
                  std::complex<double> replacement(correctionVector(i).real(),0);
                  correctionVector(i) = replacement;
              }
        }

        // Fill the output vector
        computedCorrection(0) = correctionVector(2).real();
        computedCorrection(1) = correctionVector(3).real();
        computedCorrection(1) = correctionVector(4).real();

    }


    return computedCorrection;
}


Eigen::VectorXd floquetApproximation(int librationPointNr, std::string orbitType,
                                                  double amplitude, double thrustMagnitude, double accelerationAngle, double accelerationAngle2, const double initialMass, const int numberOfPatchPoints, const int numberOfCorrections, const double maxEigenValueDeviation )
{
    Eigen::VectorXd lowThrustInitialStateVectorGuess(11*numberOfPatchPoints);
    lowThrustInitialStateVectorGuess.setZero();

    std::cout.precision(6);

    std::cout << "\n Generate the initial Guess via Floquet control using: " << numberOfCorrections << " corrections and " << numberOfPatchPoints << " numberOfPatchPoints. " << std::endl;

    // ====  1. Compute the full equilibrium State Vector ==== //

    // Compute the mass parameter of the Earth-Moon system
    const double primaryGravitationalParameter = tudat::celestial_body_constants::EARTH_GRAVITATIONAL_PARAMETER;
    const double secondaryGravitationalParameter = tudat::celestial_body_constants::MOON_GRAVITATIONAL_PARAMETER;
    const double massParameter = tudat::gravitation::circular_restricted_three_body_problem::computeMassParameter( primaryGravitationalParameter, secondaryGravitationalParameter );

    // Compute location of the artificial equilibrium point
    Eigen::Vector2d equilibriumLocation = createEquilibriumLocations( librationPointNr, thrustMagnitude, accelerationAngle, "acceleration", massParameter );

    Eigen::VectorXd equilibriumStateVector =   Eigen::VectorXd::Zero( 10 );
    equilibriumStateVector.segment(0,2) = equilibriumLocation;
    equilibriumStateVector(6) = thrustMagnitude;
    equilibriumStateVector(7) = accelerationAngle;
    equilibriumStateVector(8) = accelerationAngle2;
    equilibriumStateVector(9) = initialMass;

    //std::cout << "fullStateVectorEquilibirum: \n" << equilibriumStateVector << std::endl;

    // ====  1. Compute the initial uncorrected state ==== //
    Eigen::VectorXd initialStateVectorUncorrected =   Eigen::VectorXd::Zero( 10 );
    Eigen::VectorXd initialStateVectorCorrected = Eigen::VectorXd::Zero(10);
    Eigen::VectorXd initialPerturbationVector = Eigen::VectorXd::Zero( 10 );
    Eigen::VectorXd initialPerturbationVectorAfterCorrection = Eigen::VectorXd::Zero( 10 );

    initialStateVectorUncorrected = equilibriumStateVector;
    double xArgument = 0.0;
    double yArgument = 0.0;
    double offsetAngle = 0.0;

    if(orbitType == "horizontal")
    {
        if (librationPointNr == 1 or librationPointNr == 2)
           {
                xArgument = equilibriumStateVector(0) - (1.0 - massParameter);

            } else
            {
                xArgument = equilibriumStateVector(0) - ( - massParameter);
            }

        yArgument = equilibriumStateVector(1);
        offsetAngle = atan2(yArgument, xArgument);

        initialPerturbationVector(0) = amplitude * cos(offsetAngle);
        initialPerturbationVector(1) = amplitude * sin(offsetAngle);
    } else
    {
        initialPerturbationVector(2) = amplitude;
    }

    initialPerturbationVector(2) = amplitude;

    initialStateVectorUncorrected = equilibriumStateVector + initialPerturbationVector;


    //  3. Compute the State Propagation Matrix

    // Provide an offset in the direction of the minimum in-plane center eigenvalue of the state propagation matrix
    Eigen::MatrixXd stateDerivativeInclSPM = Eigen::MatrixXd::Zero(10,11);
    Eigen::MatrixXd statePropagationMatrix = Eigen::MatrixXd::Zero(6,6);

    stateDerivativeInclSPM = computeStateDerivativeAugmented( 0.0, getFullInitialStateAugmented( equilibriumStateVector) );
    statePropagationMatrix = stateDerivativeInclSPM.block(0,1,6,6);

    //std::cout << "statePropagationMatrix: \n" << statePropagationMatrix << std::endl;

    computeMotionDecomposition(librationPointNr, orbitType, statePropagationMatrix, Eigen::MatrixXd::Identity(6,6), initialPerturbationVector.segment(0,6), 0.0, 1.0E-13);

    Eigen::VectorXd correctionVelocity = computeVelocityCorrection(librationPointNr, orbitType, statePropagationMatrix, Eigen::MatrixXd::Identity(6,6), initialPerturbationVector.segment(0,6), 0.0);

    initialStateVectorCorrected = initialStateVectorUncorrected;
    initialStateVectorCorrected.segment(3,3) = initialStateVectorCorrected.segment(3,3) + correctionVelocity;

    initialPerturbationVectorAfterCorrection = equilibriumStateVector - initialStateVectorCorrected;

    computeMotionDecomposition(librationPointNr, orbitType, statePropagationMatrix, Eigen::MatrixXd::Identity(6,6), initialPerturbationVectorAfterCorrection.segment(0,6), 0.0, 1.0E-13);


    //std::cout << "initialStateVector Corrected: \n" << initialStateVectorCorrected << std::endl;

    //std::cout << "initialPerturbationVector: \n" << initialPerturbationVector << std::endl;
    //std::cout << "initialPerturbationVectorAfterCorrection: \n" << initialPerturbationVectorAfterCorrection << std::endl;


    // 5. Estimate the approximate period via propagataToFinalThetaCorrection

    std::map< double, Eigen::VectorXd > stateHistoryPeriodGuess;

    std::pair< Eigen::MatrixXd, double > finalTimeStateTheta = propagateOrbitAugmentedToFinalThetaCondition(getFullInitialStateAugmented( initialStateVectorCorrected), massParameter,
                                                                                                            1, stateHistoryPeriodGuess, -1, 0.0);
    Eigen::MatrixXd stateVectorInclSTMTheta      = finalTimeStateTheta.first;
    double currentTimeTheta             = finalTimeStateTheta.second;
    Eigen::VectorXd stateVectorOnly = stateVectorInclSTMTheta.block( 0, 0, 10, 1 );

    // 6. Discretize the trajectory in specified number of patch points

    double finalTime = 0.0;
    double finalPeriodicTime = currentTimeTheta;
   Eigen::VectorXd initialStateVector = initialStateVectorCorrected;

   Eigen::VectorXd interiorManeuverVector = Eigen::VectorXd::Zero(3*numberOfPatchPoints);
   interiorManeuverVector.setZero();

    std::map< double, Eigen::VectorXd > stateHistoryInitialGuess;
    for (int i = 0; i < (numberOfPatchPoints - 1); i++){

        // compute current time at patch point
        auto periodVariable = static_cast<double>(i);
        auto periodVariableFinal = static_cast<double>(i+1);
        auto periodVariable2 = static_cast<double>(numberOfPatchPoints);
        double initialTime = periodVariable * finalPeriodicTime / (periodVariable2 - 1.0);
        double currentfinalTime =   periodVariableFinal * finalPeriodicTime / (periodVariable2 - 1.0);

        // create state at segment end-time
        finalTime = currentfinalTime;

        std::pair< Eigen::MatrixXd, double > finalTimeState;
        if ( i < ( numberOfPatchPoints - 2) or numberOfCorrections == 0 )
        {
            finalTimeState = propagateOrbitAugmentedToFinalCondition( getFullInitialStateAugmented( initialStateVector),
                                                                             massParameter, finalTime, 1, stateHistoryInitialGuess, 1000, initialTime);

        } else
        {
            finalTimeState = propagateOrbitAugmentedToFullRevolutionCondition( getFullInitialStateAugmented( initialStateVector),
                                                                               librationPointNr, massParameter, offsetAngle, 1, stateHistoryInitialGuess, 1000, initialTime);
        }
        Eigen::MatrixXd stateVectorInclSTM      = finalTimeState.first;
        double currentTime             = finalTimeState.second;
        Eigen::VectorXd stateVectorOnly = stateVectorInclSTM.block( 0, 0, 10, 1 );

        Eigen::VectorXd intermediateVelocityCorrection = Eigen::VectorXd::Zero(3);

        if (numberOfPatchPoints > 2 and numberOfCorrections > 0 and i < (numberOfPatchPoints - 2)){

            Eigen::VectorXd intermediatePerturbationVector = Eigen::VectorXd::Zero(10);
            intermediatePerturbationVector.segment(0,6) = equilibriumStateVector.segment(0,6) - stateVectorOnly.segment(0,6);
            //intermediatePerturbationVector.segment(0,6) = stateVectorInclSTM.block(0,1,6,6)*initialPerturbationVectorAfterCorrection;

            std::cout << "patch point: " << i+1 << " check motion decomposition before correction" << std::endl;

            computeMotionDecomposition(librationPointNr, orbitType, statePropagationMatrix, Eigen::MatrixXd::Identity(6,6), intermediatePerturbationVector.segment(0,6), 0.0, 1.0E-13);

            intermediateVelocityCorrection = computeVelocityCorrection(librationPointNr, orbitType, statePropagationMatrix, Eigen::MatrixXd::Identity(6,6), intermediatePerturbationVector.segment(0,6), 0.0 );

            interiorManeuverVector.segment((i+1)*3,3) = intermediateVelocityCorrection;
        }



        initialTime = currentTime;


        initialStateVector = stateVectorOnly;
        initialStateVector.segment(3,3) = initialStateVector.segment(3,3) - intermediateVelocityCorrection;

        initialPerturbationVectorAfterCorrection.segment(0,6) = equilibriumStateVector.segment(0,6) - initialStateVector.segment(0,6);

        //std::cout << "patch point: " << i+1 << " check motion decomposition after correction" << std::endl;

        computeMotionDecomposition(librationPointNr, orbitType, statePropagationMatrix, Eigen::MatrixXd::Identity(6,6), initialPerturbationVectorAfterCorrection.segment(0,6), 0.0, 1.0E-13);


        if (i == 0  )
       {

            lowThrustInitialStateVectorGuess.segment(0,10) = initialStateVectorCorrected;
            lowThrustInitialStateVectorGuess(10) = 0.0;

            lowThrustInitialStateVectorGuess.segment(11*(i+1),10) = stateVectorOnly;
            lowThrustInitialStateVectorGuess((11*(i+1))+10) = currentTime;

        } else if (i == ( numberOfPatchPoints - 2 ) )
        {
            lowThrustInitialStateVectorGuess.segment(11*(i+1),10) = initialStateVectorCorrected;
            lowThrustInitialStateVectorGuess((11*(i+1))+10) = currentTime;

        } else
        {

            lowThrustInitialStateVectorGuess.segment(11*(i+1),10) = stateVectorOnly;
            lowThrustInitialStateVectorGuess((11*(i+1))+10) = currentTime;

        }

    }

    std::map< double, Eigen::VectorXd > stateHistoryCorrectedGuess;
    stateHistoryCorrectedGuess.clear();


      writeFloquetDataToFile( stateHistoryInitialGuess, stateHistoryCorrectedGuess, librationPointNr, orbitType, equilibriumStateVector, numberOfCorrections, amplitude, interiorManeuverVector);


//    //std::cout << "lowThrustInitialStateVectorGuess: \n" << lowThrustInitialStateVectorGuess << std::endl;

    return lowThrustInitialStateVectorGuess;
}

