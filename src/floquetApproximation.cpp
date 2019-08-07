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
        Eigen::MatrixXcd updateMatrix(6,5);
        Eigen::VectorXcd deviationVector(6,1);
        Eigen::MatrixXcd correctionVector(5,1);

        updateMatrix.setZero();
        deviationVector.setZero();
        correctionVector.setZero();

        updateMatrix.block(0,0,3,1) = perturbationDecomposition.block(0,4,3,1);
        updateMatrix.block(3,0,3,1) = perturbationDecomposition.block(3,4,3,1);
        updateMatrix.block(0,1,3,1) = perturbationDecomposition.block(0,5,3,1);
        updateMatrix.block(3,1,3,1) = perturbationDecomposition.block(3,5,3,1);

        updateMatrix(3,2) = -1.0;
        updateMatrix(4,3) = -1.0;
        updateMatrix(5,4) = -1.0;

        deviationVector = perturbationDecomposition.block(0,0,6,1) + perturbationDecomposition.block(0,1,6,1) +
                perturbationDecomposition.block(0,2,6,1) + perturbationDecomposition.block(0,3,6,1);

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
        computedCorrection(2) = correctionVector(4).real();

    }


    return computedCorrection;
}


Eigen::VectorXd floquetApproximation(int librationPointNr, std::string orbitType,
                                                  double amplitude, double thrustMagnitude, double accelerationAngle, double accelerationAngle2, const double initialMass, const int numberOfPatchPoints, const double correctionTime, const double maxEigenValueDeviation )
{
    Eigen::VectorXd lowThrustInitialStateVectorGuess(11*numberOfPatchPoints);
    lowThrustInitialStateVectorGuess.setZero();

    std::cout << "\n ====== Generate the initial guess with floquet controller ====== " << std::endl
              << "Amplitude: " << amplitude << std::endl
              << "numberOfPathPoints: " << numberOfPatchPoints << std::endl
              << "correctionTime: " << correctionTime << std::endl;


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

    initialStateVectorUncorrected = equilibriumStateVector + initialPerturbationVector;

    //  3. Compute the State Propagation Matrix

    // Provide an offset in the direction of the minimum in-plane center eigenvalue of the state propagation matrix
    Eigen::MatrixXd stateDerivativeInclSPM = Eigen::MatrixXd::Zero(10,11);
    Eigen::MatrixXd statePropagationMatrix = Eigen::MatrixXd::Zero(6,6);

    stateDerivativeInclSPM = computeStateDerivativeAugmented( 0.0, getFullInitialStateAugmented( equilibriumStateVector) );
    statePropagationMatrix = stateDerivativeInclSPM.block(0,1,6,6);

    computeMotionDecomposition(librationPointNr, orbitType, statePropagationMatrix, Eigen::MatrixXd::Identity(6,6), initialPerturbationVector.segment(0,6), 0.0, 1.0E-13);

    Eigen::VectorXd correctionVelocity = computeVelocityCorrection(librationPointNr, orbitType, statePropagationMatrix, Eigen::MatrixXd::Identity(6,6), initialPerturbationVector.segment(0,6), 0.0);

    initialStateVectorCorrected = initialStateVectorUncorrected;
    initialStateVectorCorrected.segment(3,3) = initialStateVectorCorrected.segment(3,3) + correctionVelocity;

    initialPerturbationVectorAfterCorrection = equilibriumStateVector - initialStateVectorCorrected;

    computeMotionDecomposition(librationPointNr, orbitType, statePropagationMatrix, Eigen::MatrixXd::Identity(6,6), initialPerturbationVectorAfterCorrection.segment(0,6), 0.0, 1.0E-13);

    // 5. Estimate the approximate period via propagataToFinalThetaCorrection

    std::map< double, Eigen::VectorXd > stateHistoryPeriodGuess;

    std::pair< Eigen::MatrixXd, double > finalTimeStateRev;
    double currentTimeRev = 0.0;

    bool fullRevolutionCompleted = false;
    double initialTime = 0.0;
    int thetaSignChanges = 0;

    Eigen::VectorXd initialStateVector = Eigen::VectorXd::Zero(10);
    initialStateVector = initialStateVectorCorrected;
    int numberOfCorrections = 0;
    double thetaSign = 0.0;

    if (orbitType == "horizontal")
    {
        while (fullRevolutionCompleted == false )
        {
            double finalTime = initialTime + correctionTime;

            std::pair< Eigen::MatrixXd, double > finalTimeStateRev = propagateOrbitAugmentedToFullRevolutionOrFinalTime( getFullInitialStateAugmented(initialStateVector), librationPointNr, massParameter, offsetAngle,
                                                                                                                  finalTime, 1, thetaSignChanges,  thetaSign, fullRevolutionCompleted, stateHistoryPeriodGuess, 1000, initialTime);

            Eigen::MatrixXd stateVectorInclSTMRev     = finalTimeStateRev.first;
            currentTimeRev            = finalTimeStateRev.second;
            Eigen::VectorXd stateVectorOnly = stateVectorInclSTMRev.block( 0, 0, 10, 1 );

            // Set the initial Time to the next correction point
            initialTime = currentTimeRev;

            // Correct the stateVector
            Eigen::VectorXd intermediateVelocityCorrection = Eigen::VectorXd::Zero(3);

            Eigen::VectorXd intermediatePerturbationVector = Eigen::VectorXd::Zero(10);
            intermediatePerturbationVector.segment(0,6) = equilibriumStateVector.segment(0,6) - stateVectorOnly.segment(0,6);

            intermediateVelocityCorrection = computeVelocityCorrection(librationPointNr, orbitType, statePropagationMatrix, Eigen::MatrixXd::Identity(6,6), intermediatePerturbationVector.segment(0,6), 0.0 );

            initialStateVector = stateVectorOnly;
            initialStateVector.segment(3,3) = initialStateVector.segment(3,3)-intermediateVelocityCorrection;

            numberOfCorrections++;

        }

    } else
    {
        Eigen::EigenSolver< Eigen::MatrixXd > eigSPM( statePropagationMatrix );
        currentTimeRev = 2.0 * tudat::mathematical_constants::PI / ( eigSPM.eigenvalues()(4).imag() );
        //std::cout << "estimated period  center eigenvalue: " << currentTimeTheta << std::endl;
    }

    std::cout << "estimated period: " << currentTimeRev << std::endl;

//    // 6. Discretize the trajectory in specified number of patch points
//        // determine the patch point spacing interval in time

//    fullRevolutionCompleted = false;
//    initialTime = 0.0;
//    thetaSignChanges = 0;

//    initialStateVector = Eigen::VectorXd::Zero(10);
//    initialStateVector = initialStateVectorCorrected;
//    numberOfCorrections = 0;
//    thetaSign = 0.0;

//    double RevolutionTime = currentTimeRev;
//    auto patchPointVariable = static_cast<double>(numberOfPatchPoints);
//    double patchPointInterval = RevolutionTime / (patchPointVariable -1.0);

//    //std::cout << "patchPointInterval: " << patchPointInterval << std::endl;
//    Eigen::VectorXd interiorManeuverVector;
//    if (numberOfCorrections > 0)
//    {
//     interiorManeuverVector = Eigen::VectorXd::Zero(3*numberOfCorrections);
//    } else
//    {
//        interiorManeuverVector = Eigen::VectorXd::Zero(3);
//    }
//    interiorManeuverVector.setZero();

//   std::map< double, Eigen::VectorXd > stateHistoryInitialGuess;

//   numberOfCorrections = 0;

//   // Store the initial patch point and define patch point variables
//   lowThrustInitialStateVectorGuess.segment(0,10) = initialStateVectorCorrected;
//   lowThrustInitialStateVectorGuess(10) = 0.0;

//   int numberOfPatchPointsStored = 1;
//   double patchPointTime = patchPointInterval;

//   while (fullRevolutionCompleted == false )
//   {
//       double finalTime = initialTime + correctionTime;

//       if ( (patchPointTime > initialTime) and (patchPointTime < finalTime) )
//       {
//           std::cout << "=== TESTING PATCH POINT CONDITIONS === " << std::endl
//                      << "initialTime: " << initialTime << std::endl
//                      << "patchPointTime: " << patchPointTime << std::endl
//                      << "finalTime: " << finalTime << std::endl;


//           std::pair< Eigen::MatrixXd, double >finalTimeStatePatchPoint = propagateOrbitAugmentedToFinalCondition( getFullInitialStateAugmented( initialStateVector),
//                                                                                        massParameter, patchPointTime, 1, stateHistoryInitialGuess, 1000, initialTime);

//           Eigen::MatrixXd stateVectorInclSTMPatchPoint    = finalTimeStatePatchPoint.first;
//           double currentTimePatchPoint         = finalTimeStatePatchPoint.second;
//           Eigen::VectorXd stateVectorOnlyPatchPoint = stateVectorInclSTMPatchPoint.block( 0, 0, 10, 1 );

//           lowThrustInitialStateVectorGuess.segment(numberOfPatchPointsStored*11,10) = stateVectorOnlyPatchPoint;
//           lowThrustInitialStateVectorGuess(numberOfPatchPointsStored*11+10) = currentTimePatchPoint;

//           patchPointTime = patchPointTime + patchPointInterval;
//           numberOfPatchPointsStored++;

//           if(numberOfPatchPointsStored == (numberOfPatchPoints - 1))
//           {
//               lowThrustInitialStateVectorGuess.segment(numberOfPatchPointsStored*11,10) = initialStateVectorCorrected;
//               lowThrustInitialStateVectorGuess(numberOfPatchPointsStored*11+10) = RevolutionTime;


//               numberOfPatchPointsStored++;

//               fullRevolutionCompleted = true;
//           }

//       }

////      std::cout << " \n=== propagationCheck SECOND ROUND ===" << std::endl
////                << "referenceAngle: " << offsetAngle * 180.0 / tudat::mathematical_constants::PI << std::endl
////                  << "initialTime: " << initialTime << std::endl
////                << "finalTime: " << finalTime << std::endl
////                 << "thetaSign: " << thetaSignChanges << std::endl
////                  << "thetaSignChanges: " << thetaSignChanges << std::endl
////                  << "numberOfCorrections: " << numberOfCorrections << std::endl;


//       std::pair< Eigen::MatrixXd, double > finalTimeStateRev = propagateOrbitAugmentedToFullRevolutionOrFinalTime( getFullInitialStateAugmented(initialStateVector), librationPointNr, massParameter, offsetAngle,
//                                                                                                             finalTime, 1, thetaSignChanges,  thetaSign, fullRevolutionCompleted, stateHistoryPeriodGuess, -1, initialTime);

//       Eigen::MatrixXd stateVectorInclSTMRev     = finalTimeStateRev.first;
//       currentTimeRev            = finalTimeStateRev.second;
//       Eigen::VectorXd stateVectorOnly = stateVectorInclSTMRev.block( 0, 0, 10, 1 );

//       // Set the initial Time to the next correction point
//       initialTime = currentTimeRev;

//       // Correct the stateVector
//       Eigen::VectorXd intermediateVelocityCorrection = Eigen::VectorXd::Zero(3);

//       Eigen::VectorXd intermediatePerturbationVector = Eigen::VectorXd::Zero(10);
//       intermediatePerturbationVector.segment(0,6) = equilibriumStateVector.segment(0,6) - stateVectorOnly.segment(0,6);

//       intermediateVelocityCorrection = computeVelocityCorrection(librationPointNr, orbitType, statePropagationMatrix, Eigen::MatrixXd::Identity(6,6), intermediatePerturbationVector.segment(0,6), 0.0 );

//       initialStateVector = stateVectorOnly;
//       initialStateVector.segment(3,3) = initialStateVector.segment(3,3)-intermediateVelocityCorrection;

//       if (numberOfCorrections > 0)
//       {
//           interiorManeuverVector.segment((numberOfCorrections)*3,3) = intermediateVelocityCorrection;
//       }


//       numberOfCorrections++;

//   }

      //writeFloquetDataToFile( stateHistoryPeriodGuess, librationPointNr, orbitType, equilibriumStateVector, correctionTime, amplitude, interiorManeuverVector);


     //std::cout << "lowThrustInitialStateVectorGuess: \n" << lowThrustInitialStateVectorGuess << std::endl;

     return lowThrustInitialStateVectorGuess;
}


//    for (int i = 0; i < (numberOfPatchPoints - 1); i++){

//        // compute current time at patch point
//        auto periodVariable = static_cast<double>(i);
//        auto periodVariableFinal = static_cast<double>(i+1);
//        auto periodVariable2 = static_cast<double>(numberOfPatchPoints);
//        double initialTime = periodVariable * RevolutionTime / (periodVariable2 - 1.0);
//        double currentfinalTime =   periodVariableFinal * RevolutionTime / (periodVariable2 - 1.0);

//        // create state at segment end-time
//        finalTime = currentfinalTime;

//        std::pair< Eigen::MatrixXd, double > finalTimeState;
//        if ( i < ( numberOfPatchPoints - 2) or numberOfCorrections == 0 )
//        {
//            finalTimeState = propagateOrbitAugmentedToFinalCondition( getFullInitialStateAugmented( initialStateVector),
//                                                                             massParameter, finalTime, 1, stateHistoryInitialGuess, 1000, initialTime);

//            //std::cout << "\n===Patch Point: " << i << " ==="<< std::endl
//            //          << "initialStateVector: \n" << initialStateVector << std::endl;

//        } else
//        {
//            finalTimeState = propagateOrbitAugmentedToFullRevolutionCondition( getFullInitialStateAugmented( initialStateVector),
//                                                                               librationPointNr, massParameter, offsetAngle, 1, stateHistoryInitialGuess, 1000, initialTime);

//            //std::cout << "\n===Patch Point: " << i << " ==="<< std::endl
//            //          << "initialStateVector: \n" << initialStateVector << std::endl;
//        }
//        Eigen::MatrixXd stateVectorInclSTM      = finalTimeState.first;
//        double currentTime             = finalTimeState.second;
//        Eigen::VectorXd stateVectorOnly = stateVectorInclSTM.block( 0, 0, 10, 1 );

//        //std::cout<< "finalStateVector: \n" << stateVectorOnly << std::endl;


//        Eigen::VectorXd intermediateVelocityCorrection = Eigen::VectorXd::Zero(3);

//        if (numberOfPatchPoints > 2 and numberOfCorrections > 0 and i < (numberOfPatchPoints - 2)){

//            Eigen::VectorXd intermediatePerturbationVector = Eigen::VectorXd::Zero(10);
//            intermediatePerturbationVector.segment(0,6) = equilibriumStateVector.segment(0,6) - stateVectorOnly.segment(0,6);
//            //intermediatePerturbationVector.segment(0,6) = stateVectorInclSTM.block(0,1,6,6)*initialPerturbationVectorAfterCorrection;

//            //std::cout << "patch point: " << i+1 << " check motion decomposition before correction" << std::endl;

//            //computeMotionDecomposition(librationPointNr, orbitType, statePropagationMatrix, Eigen::MatrixXd::Identity(6,6), intermediatePerturbationVector.segment(0,6), 0.0, 1.0E-13);

//            intermediateVelocityCorrection = computeVelocityCorrection(librationPointNr, orbitType, statePropagationMatrix, Eigen::MatrixXd::Identity(6,6), intermediatePerturbationVector.segment(0,6), 0.0 );

//            interiorManeuverVector.segment((i+1)*3,3) = intermediateVelocityCorrection;
//        }



//        initialTime = currentTime;


//        initialStateVector = stateVectorOnly;
//        initialStateVector.segment(3,3) = initialStateVector.segment(3,3) - intermediateVelocityCorrection;

//        initialPerturbationVectorAfterCorrection.segment(0,6) = equilibriumStateVector.segment(0,6) - initialStateVector.segment(0,6);

//        //std::cout << "patch point: " << i+1 << " check motion decomposition after correction" << std::endl;

//        //computeMotionDecomposition(librationPointNr, orbitType, statePropagationMatrix, Eigen::MatrixXd::Identity(6,6), initialPerturbationVectorAfterCorrection.segment(0,6), 0.0, 1.0E-13);


//        if (i == 0  )
//       {

//            lowThrustInitialStateVectorGuess.segment(0,10) = initialStateVectorCorrected;
//            lowThrustInitialStateVectorGuess(10) = 0.0;

//            lowThrustInitialStateVectorGuess.segment(11*(i+1),10) = initialStateVector;
//            lowThrustInitialStateVectorGuess((11*(i+1))+10) = currentTime;

//        } else if (i == ( numberOfPatchPoints - 2 ) )
//        {
//            lowThrustInitialStateVectorGuess.segment(11*(i+1),10) = initialStateVectorCorrected;
//            lowThrustInitialStateVectorGuess((11*(i+1))+10) = currentTime;

//        } else
//        {

//            lowThrustInitialStateVectorGuess.segment(11*(i+1),10) = initialStateVector;
//            lowThrustInitialStateVectorGuess((11*(i+1))+10) = currentTime;

//        }

//    }

//    std::map< double, Eigen::VectorXd > stateHistoryCorrectedGuess;
//    stateHistoryCorrectedGuess.clear();


//      writeFloquetDataToFile( stateHistoryInitialGuess, stateHistoryCorrectedGuess, librationPointNr, orbitType, equilibriumStateVector, numberOfCorrections, amplitude, interiorManeuverVector);


//    //std::cout << "\nResulting initial guess: \n" << lowThrustInitialStateVectorGuess << std::endl;



