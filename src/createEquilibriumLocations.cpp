#include <fstream>
#include <iomanip>
#include<iostream>
#include <boost/bind.hpp>
#include <boost/make_shared.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <boost/shared_array.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/shared_container_iterator.hpp>
#include <map>
#include <cmath>

#include "Tudat/Mathematics/BasicMathematics/mathematicalConstants.h"
#include "Tudat/Astrodynamics/BasicAstrodynamics/celestialBodyConstants.h"
#include "Tudat/Astrodynamics/Gravitation/jacobiEnergy.h"
#include "Tudat/Astrodynamics/Gravitation/librationPoint.h"
#include "Tudat/Mathematics/BasicMathematics/function.h"

#include "Tudat/Mathematics/RootFinders/newtonRaphson.h"

#include "functions/artificialLibrationPointLocationFunction.h"
#include "functions/artificialLibrationPointLocationFunction1.h"
#include "functions/artificialLibrationPointLocationFunction2.h"
#include "functions/artificialLibrationPointLocationFunction3.h"
#include "propagateOrbitAugmented.h"

//#include "functions/signfunction.h"
#include "applyMultivariateRootFinding.h"

#include "createEquilibriumLocations.h"

Eigen::Vector6d computeDeviationAfterPropagation(const Eigen::Vector3d equilibriumLocationWithIterations, const double accelerationMagnitude, const double accelerationAngle, const double massParameter, const double finalTime)
{
    Eigen::Vector6d outputVector; outputVector.setZero();

    // compute full initial state
    Eigen::MatrixXd initialStateInclSTM(10,11); initialStateInclSTM.setZero();
    initialStateInclSTM.block(0,0,2,1) = equilibriumLocationWithIterations.segment(0,2);
    initialStateInclSTM(6,0) = accelerationMagnitude;
    initialStateInclSTM(7,0) = accelerationAngle;
    initialStateInclSTM(9,0) = 1.0;
    initialStateInclSTM.block(0,1,10,10).setIdentity();

    std::map<double, Eigen::VectorXd > stateHistory;
    std::pair< Eigen::MatrixXd, double > finalStateInclSTMAndTime = propagateOrbitAugmentedToFinalCondition(initialStateInclSTM, massParameter, finalTime, 1, stateHistory, -1, 0.0);

    Eigen::MatrixXd finalStateInclSTM = finalStateInclSTMAndTime.first;
    outputVector = initialStateInclSTM.block(0,0,6,1) - finalStateInclSTM.block(0,0,6,1);

    return outputVector;
}

void writeResultsToFile (const int librationPointNr, const double parameterOfInterest, const std::string parameterSpecification, const double seedAngle, const double continuationDirection, std::map< double, Eigen::Vector3d > equilibriaCatalog, std::map <double, Eigen::MatrixXd > stabilityCatalog, std::map< double, Eigen::Vector6d > deviationCatalog ) {

    std::string direction;
    if (continuationDirection > 0.0)
    {
        direction = "forward";
    } else
    {
        direction = "backward";
    }
    // Prepare file for initial conditions
    remove(("../data/data/raw/equilibria/L" + std::to_string(librationPointNr) + "_" + parameterSpecification + "_" + std::to_string(parameterOfInterest) + "_" + std::to_string(seedAngle) + "_" + direction +"_equilibria.txt").c_str());
    std::ofstream textFileInitialConditions;
    textFileInitialConditions.open(("../data/raw/equilibria/L" + std::to_string(librationPointNr) + "_" + parameterSpecification + "_" + std::to_string(parameterOfInterest) + "_" + std::to_string(seedAngle) + "_" + direction + "_equilibria.txt"));
    textFileInitialConditions.precision(std::numeric_limits<double>::digits10);

    for(auto ic = equilibriaCatalog.cbegin(); ic != equilibriaCatalog.cend(); ++ic) {
        textFileInitialConditions << std::left << std::scientific                                          << std::setw(25)
                                  << ic->first  << std::setw(25) << ic->second(0)  << std::setw(25) << ic->second(1) << std::setw(25) << ic->second(2) << std::endl;

    }

    remove(("../data/data/raw/equilibria/L" + std::to_string(librationPointNr) + "_" + parameterSpecification + "_" + std::to_string(parameterOfInterest) + "_" + std::to_string(seedAngle) + "_" + direction + "_equilibria_stability.txt").c_str());
    std::ofstream textFileInitialConditionsStability;
    textFileInitialConditionsStability.open(("../data/raw/equilibria/L" + std::to_string(librationPointNr) + "_" + parameterSpecification + "_" + std::to_string(parameterOfInterest) + "_" + std::to_string(seedAngle) + "_" + direction + "_equilibria_stability.txt"));

    textFileInitialConditionsStability.precision(std::numeric_limits<double>::digits10);

        for(auto ic = stabilityCatalog.cbegin(); ic != stabilityCatalog.cend(); ++ic) {
        textFileInitialConditionsStability << std::left << std::scientific                                          << std::setw(25)
                                           << ic->first  << std::setw(25) << ic->second(0)  << std::setw(25) << ic->second(1) << std::setw(25) << ic->second(2) << std::setw(25) << ic->second(3) << std::setw(25) << ic->second(4) << std::setw(25) << ic->second(5) << std::setw(25) << ic->second(6) << std::setw(25)
                                                                          << ic->second(7)  << std::setw(25) << ic->second(8) << std::setw(25) << ic->second(9) << std::setw(25) << ic->second(10) << std::setw(25) << ic->second(11) << std::setw(25) << ic->second(12) << std::setw(25)
                                                                          << ic->second(13)  << std::setw(25) << ic->second(14) << std::setw(25) << ic->second(15) << std::setw(25) << ic->second(16) << std::setw(25) << ic->second(17) << std::setw(25) << ic->second(18) << std::setw(25)
                                                                          << ic->second(19)  << std::setw(25) << ic->second(20) << std::setw(25) << ic->second(21) << std::setw(25) << ic->second(22) << std::setw(25) << ic->second(23) << std::setw(25) << ic->second(24) << std::setw(25)
                                                                          << ic->second(25)  << std::setw(25) << ic->second(26) << std::setw(25) << ic->second(27) << std::setw(25) << ic->second(28) << std::setw(25) << ic->second(29) << std::setw(25) << ic->second(30) << std::setw(25)
                                                                          << ic->second(31)  << std::setw(25) << ic->second(32) << std::setw(25) << ic->second(33) << std::setw(25) << ic->second(34) << std::setw(25) << ic->second(35) << std::endl;
    }

    remove(("../data/data/raw/equilibria/L" + std::to_string(librationPointNr) + "_" + parameterSpecification + "_" + std::to_string(parameterOfInterest) + "_" + std::to_string(seedAngle) + "_" + direction + "_equilibria_deviation.txt").c_str());
    std::ofstream textFileInitialConditionsDeviation;
   textFileInitialConditionsDeviation.open(("../data/raw/equilibria/L" + std::to_string(librationPointNr) + "_" + parameterSpecification + "_" + std::to_string(parameterOfInterest) + "_" + std::to_string(seedAngle) + "_" + direction + "_equilibria_deviation.txt"));

    textFileInitialConditionsDeviation.precision(std::numeric_limits<double>::digits10);

    for(auto ic = deviationCatalog.cbegin(); ic != deviationCatalog.cend(); ++ic) {
        textFileInitialConditionsDeviation << std::left << std::scientific                                          << std::setw(25)
                                  << ic->first  << std::setw(25) << ic->second(0)  << std::setw(25) << ic->second(1) << std::setw(25) << ic->second(2) << std::setw(25) << ic->second(3) << std::setw(25) << ic->second(4) << std::setw(25) << ic->second(5) << std::endl;

    }




}


Eigen::MatrixXd computeEquilibriaStability(Eigen::Vector2d equilibriumLocation, const double alpha, const double accelerationMagnitude, const double massParameter) {
    Eigen::MatrixXd statePropagationMatrix = Eigen::MatrixXd::Zero(6,6);
    double xPositionScaledSquared = (equilibriumLocation(0)+massParameter) * (equilibriumLocation(0)+massParameter);
    double xPositionScaledSquared2 = (1.0-massParameter-equilibriumLocation(0)) * (1.0-massParameter-equilibriumLocation(0));
    double yPositionScaledSquared = (equilibriumLocation(1) * equilibriumLocation(1) );

    // Compute distances to primaries.
    double distanceToPrimaryBody   = sqrt(xPositionScaledSquared     + yPositionScaledSquared);
    double distanceToSecondaryBody = sqrt(xPositionScaledSquared2 + yPositionScaledSquared);

    double distanceToPrimaryCubed = distanceToPrimaryBody * distanceToPrimaryBody * distanceToPrimaryBody;
    double distanceToSecondaryCubed = distanceToSecondaryBody * distanceToSecondaryBody * distanceToSecondaryBody;

    double distanceToPrimaryToFifthPower = distanceToPrimaryCubed * distanceToPrimaryBody * distanceToPrimaryBody;
    double distanceToSecondaryToFifthPower = distanceToSecondaryCubed * distanceToSecondaryBody * distanceToSecondaryBody;

    // Compute partial derivatives of the potential.
    double Uxx = (3.0*(1.0-massParameter)*xPositionScaledSquared          )/distanceToPrimaryToFifthPower+ (3.0*massParameter*xPositionScaledSquared2           )/distanceToSecondaryToFifthPower - (1.0-massParameter)/distanceToPrimaryCubed - massParameter/distanceToSecondaryCubed + 1.0;
    double Uxy = (3.0*(1.0-massParameter)*(equilibriumLocation(0)+massParameter)*equilibriumLocation(1))/distanceToPrimaryToFifthPower- (3.0*massParameter*(1.0-massParameter-equilibriumLocation(0))*equilibriumLocation(1))/distanceToSecondaryToFifthPower;
    double Uyx = Uxy;
    double Uyy = (3.0*(1.0-massParameter)*yPositionScaledSquared                         )/distanceToPrimaryToFifthPower+ (3.0*massParameter*yPositionScaledSquared                             )/distanceToSecondaryToFifthPower - (1.0-massParameter)/distanceToPrimaryCubed - massParameter/distanceToSecondaryCubed + 1.0 ;

    // Compute partial derivatives of the low-thrust acceleration
    double xAccPartialMass = -accelerationMagnitude * std::cos(alpha * tudat::mathematical_constants::PI / 180.0);
    double xAccPartialalpha = -accelerationMagnitude * std::sin(alpha * tudat::mathematical_constants::PI / 180.0);
    double yAccPartialMass = -accelerationMagnitude * std::sin(alpha * tudat::mathematical_constants::PI / 180.0);
    double yAccPartialalpha = -accelerationMagnitude * std::cos(alpha * tudat::mathematical_constants::PI / 180.0);

    statePropagationMatrix << 0,   0,   1,   0, 0,               0,
                              0,   0,   0,   1, 0,               0,
                              Uxx, Uxy, 0,   2, xAccPartialMass, xAccPartialalpha,
                              Uyx, Uyy, -2,  0, yAccPartialMass, yAccPartialalpha,
                              0,   0,   0,   0, 0,               0,
                              0,   0,   0,   0, 0,               0;

    return statePropagationMatrix;
}


double newtonRapshonRootFinding(const int librationPointNr, const double accelerationMagnitude, const double massParameter, const double maxDeviationFromSolution, const double maxNumberOfIterations, const double relaxationParameter )
{

double convergedGuess;
double currentGuess;
double correctedGuess;
double functionValue;
double derivativeValue;
int numberOfIterations;

if (librationPointNr == 1)
{
    currentGuess = 0.8369151483688;
} else if (librationPointNr == 2)
{
    currentGuess = 1.1556821477825;
} else
{
    currentGuess = -1.0050626438969;

}

// Compute initial functionValue

double DistanceToPrimaryNorm = sqrt( (currentGuess + massParameter) * (currentGuess + massParameter) );
double DistanceToSecondaryNorm = sqrt( (currentGuess - 1.0 + massParameter) * (currentGuess - 1.0 + massParameter) );

double r13Cubed = DistanceToPrimaryNorm * DistanceToPrimaryNorm * DistanceToPrimaryNorm;
double r23Cubed = DistanceToSecondaryNorm * DistanceToSecondaryNorm * DistanceToSecondaryNorm;

double termRelatedToPrimary = ( (1.0 - massParameter) / r13Cubed) ;
double termRelatedToSecondary = (massParameter / r23Cubed);

functionValue = currentGuess * (1.0 - termRelatedToPrimary - termRelatedToSecondary)
                        + massParameter * (-termRelatedToPrimary - termRelatedToSecondary) + termRelatedToSecondary + accelerationMagnitude;


numberOfIterations = 1;
while ( std::abs(functionValue) > maxDeviationFromSolution)
{
  if ( numberOfIterations > maxNumberOfIterations )
  {
      std::cout << "Newton Raphson did not converge within maxNumberOfIterations " << std::endl;
      std::cout << "FunctionValue: " << functionValue << std::endl;

          return 0.0;
  }


  double derivativeTermRelatedToPrimary = ( (currentGuess + massParameter) *(currentGuess + massParameter) )/ ( DistanceToPrimaryNorm * DistanceToPrimaryNorm );
  double derivativeTermRelatedToSecondary = ( (currentGuess -1.0 + massParameter) *(currentGuess -1.0 + massParameter) )/ ( DistanceToSecondaryNorm * DistanceToSecondaryNorm );

  derivativeValue = 1.0 - termRelatedToPrimary * (1.0 - 3.0 * derivativeTermRelatedToPrimary ) - termRelatedToSecondary * (1.0 - 3.0 * derivativeTermRelatedToSecondary);

  correctedGuess = currentGuess - relaxationParameter * functionValue/derivativeValue;

  // compute new deviationFromSolution
  double DistanceToPrimaryNorm = sqrt( (correctedGuess + massParameter) * (correctedGuess + massParameter) );
  double DistanceToSecondaryNorm = sqrt( (correctedGuess - 1.0 + massParameter) * (correctedGuess - 1.0 + massParameter) );

  double r13Cubed = DistanceToPrimaryNorm * DistanceToPrimaryNorm * DistanceToPrimaryNorm;
  double r23Cubed = DistanceToSecondaryNorm * DistanceToSecondaryNorm * DistanceToSecondaryNorm;

  double termRelatedToPrimary = ( (1.0 - massParameter) / r13Cubed) ;
  double termRelatedToSecondary = (massParameter / r23Cubed);

  functionValue = currentGuess * (1.0 - termRelatedToPrimary - termRelatedToSecondary)
                          + massParameter * (-termRelatedToPrimary - termRelatedToSecondary) + termRelatedToSecondary + accelerationMagnitude;

  currentGuess = correctedGuess;

  numberOfIterations++;
}


convergedGuess = currentGuess;
return convergedGuess;

}

void equilibriaValidation(Eigen::Vector2d equilibriumLocation, double acceleration, double alpha, double massParameter)
{
double xDistancePrimary = ( equilibriumLocation(0) + massParameter );
double xDistanceSecondary = ( equilibriumLocation(0) - 1.0 + massParameter );
double yDistance = equilibriumLocation(1);

double distanceToPrimary = sqrt( (xDistancePrimary*xDistancePrimary + yDistance*yDistance) );
double distanceToSecondary = sqrt( (xDistanceSecondary*xDistanceSecondary + yDistance*yDistance) );

double distanceToPrimaryCubed = distanceToPrimary * distanceToPrimary * distanceToPrimary;
double distanceToSecondaryCubed = distanceToSecondary * distanceToSecondary * distanceToSecondary;

double termRelatedToPrimary = (1.0 - massParameter)/distanceToPrimaryCubed;
double termRelatedToSecondary = (massParameter)/distanceToSecondaryCubed;

double xEquation = equilibriumLocation(0)*(1.0 - termRelatedToPrimary - termRelatedToSecondary)
                    + massParameter * ( -termRelatedToPrimary -termRelatedToSecondary )
                    + termRelatedToSecondary + acceleration*std::cos(alpha * tudat::mathematical_constants::PI / 180.0);
double yEquation = equilibriumLocation(1)*(1.0 - termRelatedToPrimary - termRelatedToSecondary) + acceleration*std::sin(alpha * tudat::mathematical_constants::PI / 180.0);

//std::cout<< "deviationX: " << xEquation << std::endl;
//std::cout<< "deviationY: " << yEquation << std::endl;

}

Eigen::Vector2d computeSeedSolution(const int librationPointNr, const double thrustAcceleration, const double seedAngle, const double maxDeviationFromSolution, const double massParameter, bool& seedExistence )
{

    Eigen::Vector2d seedSolution;
    seedSolution.setZero();

    double accelerationSign;
    if (seedAngle < 90.0)
    {
        accelerationSign = 1.0;
    } else
    {
        accelerationSign = -1.0;
    }

    // Create object containing the functions.
    if (librationPointNr < 4)
    {

      seedSolution(0) = newtonRapshonRootFinding(librationPointNr, thrustAcceleration * accelerationSign, massParameter);
      seedSolution(1) = 0.0;

    }  else
    {
        double distanceToPrimary = std::cbrt ( (1.0 - massParameter) / (1.0 - massParameter + ( accelerationSign * thrustAcceleration) ) );
        double distanceToSecondary =  1.0 / ( std::cbrt( (1.0 - ( accelerationSign * thrustAcceleration ) / massParameter ) ) );


        double c1 = distanceToPrimary - distanceToSecondary - 1.0;
        double c2 = distanceToPrimary - distanceToSecondary + 1.0;
        double c3 = distanceToPrimary + distanceToSecondary - 1.0;


        if (distanceToPrimary <= 0 or distanceToSecondary <= 0)
        {
            std::cout << "distance to primary or secondary zero or negative" << std::endl;
            seedExistence = false;

        }

        if (c1 >= 0)
        {
           std::cout << "c1 triangle constraint violated" << std::endl;
           seedExistence = false;
        }
        if ( c2 <= 0)
        {
            std::cout << "c2 triangle constraint violated" << std::endl;
            seedExistence = false;

        }
        if ( c3 <= 0)
        {
            std::cout << "c3 triangle constraint violated" << std::endl;
            seedExistence = false;

        }


        double theta = std::acos ( ( 1.0 + std::pow(distanceToPrimary,2.0) - std::pow(distanceToSecondary,2.0) ) / (2.0 * distanceToPrimary ) );


        if (librationPointNr == 4)
        {
            seedSolution(0) = -massParameter + distanceToPrimary*std::cos(theta);
            seedSolution(1) = distanceToPrimary*std::sin(theta);
        }
        if (librationPointNr == 5)
        {
            seedSolution(0) = -massParameter + distanceToPrimary*std::cos(theta);
            seedSolution(1) = -distanceToPrimary*std::sin(theta);
        }
    }

//    std::cout << "\n== Seed result =="<< std::endl
//              << "librationPointNr: " << librationPointNr << std::endl
//              << "alt: " << thrustAcceleration << std::endl
//              << "seedAngle: " << seedAngle << std::endl
//              << "seedExistence: " << seedExistence << std::endl
//              << "equilibriumLocation: \n" << seedSolution << std::endl;
//              equilibriaValidation(seedSolution, thrustAcceleration, seedAngle, massParameter);

    return seedSolution;
}


Eigen::Vector2d createEquilibriumLocations (const int librationPointNr, const double thrustAcceleration, const double accelerationAngle, const std::string parameterSpecification, const double ySign, const double massParameter, const double maxDeviationFromSolution, const int maxIterations, const int saveFrequency, const double stepSize, const double relaxationParameter)
{

    // Set output precision and clear screen.
    std::cout.precision(14);
    Eigen::Vector2d equilibriumLocation;
    Eigen::Vector2d seedSolution;
    Eigen::Vector3d equilibriumLocationWithIterations;
    Eigen::Vector2d targetEquilibrium;
    Eigen::MatrixXd linearizedStability;
    Eigen::VectorXd fullEquilibriumVector(10); fullEquilibriumVector.setZero();
    //double angleModCondition = 180;
    //double propagationTime = 0.01;

    fullEquilibriumVector(6) = thrustAcceleration;
    fullEquilibriumVector(9) = 1.0;

    if ( parameterSpecification == "acceleration")
    {

        Eigen::Array2d alphaArray = Eigen::Array2d::Zero();
        alphaArray << 0.0, 180.0;
        bool seedStored = false;

        for (int i = 0; i < 2; i++)
        {
            bool seedExistence = true;

            double seedAngle = alphaArray(i);

            // Check if a seed solution exists, if not, return a zero vector, otherwise computw the pat of the equilibrium contour.
            seedSolution = computeSeedSolution(librationPointNr, thrustAcceleration, seedAngle, maxDeviationFromSolution, massParameter, seedExistence );
//            std::cout   << "==== targetEquilibriumCheck Start =====" << std::endl
//                        << "seedAngle: " << seedAngle << std::endl
//                        << "accelerationAngle (Target): " << accelerationAngle << std::endl
//                        << "eqLocation: \n" <<  seedSolution << std::endl;

            if ( seedExistence == true )
            {

                // continuate the seed solutions in both directions!
                double continuationDirection;
                for (int i = 1; i < 3; i++)
                     {
                        std::map< double, Eigen::Vector3d > equilibriaCatalog;
                        std::map< double, Eigen::Vector6d > deviationCatalog;
                        std::map< double, Eigen::MatrixXd > stabilityCatalog;
                        equilibriaCatalog.clear();
                        deviationCatalog.clear();
                        stabilityCatalog.clear();

                        // determine continuation direction
                         if (i == 1)
                        {
                            continuationDirection = -1.0;
                        } else
                        {
                            continuationDirection = 1.0;
                        }

                         // store the desired equilibrium in the outputVector!
                         if (  seedAngle > (accelerationAngle - stepSize/10.0) && seedAngle < (accelerationAngle + stepSize /10.0) && seedStored == false )
                         {


                          targetEquilibrium = seedSolution;
                          seedStored =  true;

//                          std::cout   << "==== seed solution is stored =====" << std::endl
//                                      << "seedAngle: " << seedAngle << std::endl
//                                      << "lower bound: " << accelerationAngle - stepSize/10.0 << std::endl
//                                      << "upper bound: " << accelerationAngle + stepSize/10.0 << std::endl
//                                      << "targetEq: " << targetEquilibrium << std::endl;


                         }

                        //Store seed solution and its linearizedStability in the output vectors
                         equilibriumLocationWithIterations.segment(0,2) = seedSolution;
                         equilibriumLocationWithIterations(3) = 1;
                         linearizedStability = computeEquilibriaStability(seedSolution, seedAngle, thrustAcceleration, massParameter);

                         std::cout << "store first solution " << std::endl;
                         equilibriaCatalog[ seedAngle * tudat::mathematical_constants::PI / 180.0] = equilibriumLocationWithIterations;
                         stabilityCatalog[  seedAngle * tudat::mathematical_constants::PI / 180.0] = linearizedStability;
                         //deviationCatalog[  seedAngle * tudat::mathematical_constants::PI/  180.0] = computeDeviationAfterPropagation(equilibriumLocationWithIterations, thrustAcceleration, seedAngle, massParameter, propagationTime );

                         std::cout << "first solution stored" << std::endl;


                         // Initialize rootfinding procedure parameters
                         int stepCounter = 1;
                         double alpha = seedAngle;
                         equilibriumLocation = seedSolution;

                         while ( (alpha < (seedAngle + 360.0) && continuationDirection > 0.0) or (alpha > (seedAngle - 360.0) && continuationDirection < 0.0) )
                         {

                             alpha = alpha + continuationDirection * stepSize;

                             equilibriumLocationWithIterations = applyMultivariateRootFinding(equilibriumLocation, thrustAcceleration, alpha, massParameter, relaxationParameter, maxDeviationFromSolution, maxIterations);
                             equilibriumLocation = equilibriumLocationWithIterations.block(0,0,2,1);
                             linearizedStability = computeEquilibriaStability(equilibriumLocation, alpha, thrustAcceleration, massParameter);


                             if ( stepCounter % saveFrequency == 0)
                             {

                                 double alphaLog;

                                 if (alpha < 0.0)
                                 {
                                     alphaLog = alpha + 360.0;
                                 } else if (alpha > 360.0)
                                 {
                                     alphaLog = alpha - 360.0;
                                 } else
                                 {
                                     alphaLog = alpha;
                                 }

                                 equilibriaCatalog[ alphaLog * tudat::mathematical_constants::PI / 180.0 ] = equilibriumLocationWithIterations;
                                 stabilityCatalog [alphaLog * tudat::mathematical_constants::PI / 180.0 ] = linearizedStability;

                             }

                             double alphaLog;

                             if (alpha < 0.0)
                             {
                                 alphaLog = alpha + 360.0;
                             } else if (alpha > 360.0)
                             {
                                 alphaLog = alpha - 360.0;
                             } else
                             {
                                 alphaLog = alpha;
                             }

                             //if (std::abs(std::fmod(alphaLog,angleModCondition)) < stepSize/10.0 or alphaLog > 357.0)
                             //{

                               //    deviationCatalog[  alphaLog * tudat::mathematical_constants::PI/  180.0] = computeDeviationAfterPropagation(equilibriumLocationWithIterations, thrustAcceleration, alphaLog, massParameter, propagationTime );

                             //}

                             stepCounter++;

                             double alphaCondition;
                             alphaCondition = alpha;
                               if (alpha < 0.0)
                                   {
                                      alphaCondition = alphaCondition + 360.0;
                                   } else if (alpha > 360.0)
                                   {
                                      alphaCondition = alphaCondition - 360.0;
                                   } else
                                   {
                                      alphaCondition = alphaCondition;
                                   }

                             if ( alphaCondition > (accelerationAngle - stepSize/10.0) && alphaCondition < (accelerationAngle + stepSize /10.0) && seedStored == false && (equilibriumLocation(1)*ySign > 0.0 or librationPointNr < 3 ) )
                             {

                                 targetEquilibrium = equilibriumLocation;

                             }

                         }

                         writeResultsToFile(librationPointNr, thrustAcceleration, "acceleration", seedAngle, continuationDirection,  equilibriaCatalog, stabilityCatalog, deviationCatalog);


                }

            } else
            {
                double continuationDirection;
                for (int i = 1; i < 3; i++)
                     {
                     if (i == 1)
                     {
                        continuationDirection = -1.0;
                     } else
                     {
                        continuationDirection = 1.0;
                     }

                     std::map< double, Eigen::Vector3d > equilibriaCatalog;
                     std::map< double, Eigen::Vector6d > deviationCatalog;
                     std::map< double, Eigen::MatrixXd > stabilityCatalog;
                     equilibriaCatalog.clear();
                     stabilityCatalog.clear();

                     equilibriaCatalog[ 0.0 ] = Eigen::VectorXd::Zero(3);
                     stabilityCatalog[ 0.0 ] = Eigen::MatrixXd::Zero(6,6);
                     deviationCatalog[0.0] = Eigen::Vector6d::Zero();


                     writeResultsToFile(librationPointNr, thrustAcceleration, "acceleration", seedAngle, continuationDirection, equilibriaCatalog, stabilityCatalog, deviationCatalog);

                }

            }



        }

        return targetEquilibrium;

    } else
    {
        bool seedExistence = true;
        std::map< double, Eigen::Vector3d > equilibriaCatalog;
        std::map< double, Eigen::Vector6d > deviationCatalog;

        std::map< double, Eigen::MatrixXd > stabilityCatalog;
        equilibriaCatalog.clear();
        stabilityCatalog.clear();
        deviationCatalog.clear();

        double accModCondition = 0.2;

        seedSolution = computeSeedSolution(librationPointNr, 0.0, 0.0, maxDeviationFromSolution, massParameter, seedExistence );
        std::cout   << "==== targetEquilibriumCheck Start =====" << std::endl
                    << "targetAngle: " << accelerationAngle << std::endl
                    << "accelerationAngle (Target): " << accelerationAngle << std::endl
                    << "eqLocation: \n" <<  seedSolution << std::endl;

        //Store seed solution and its linearizedStability in the output vectors
         equilibriumLocationWithIterations.segment(0,2) = seedSolution;
         equilibriumLocationWithIterations(3) = 1;
         linearizedStability = computeEquilibriaStability(seedSolution, 0.0, 0.0, massParameter);

         equilibriaCatalog[ 0.0] = equilibriumLocationWithIterations;
         stabilityCatalog[  0.0] = linearizedStability;
         //deviationCatalog[  0.0] = computeDeviationAfterPropagation(equilibriumLocationWithIterations, 0.0, 0.0, massParameter, propagationTime);

         double accelerationStepSize = stepSize / 1000.0;
         double accelerationVariable = 0.0;
         equilibriumLocation = equilibriumLocationWithIterations.segment(0,2);
         int stepCounter = 1;

         while (accelerationVariable <= 0.1)
         {

             accelerationVariable = accelerationVariable + accelerationStepSize;

             equilibriumLocationWithIterations = applyMultivariateRootFinding(equilibriumLocation, accelerationVariable, accelerationAngle, massParameter, relaxationParameter, maxDeviationFromSolution, maxIterations);
             equilibriumLocation = equilibriumLocationWithIterations.block(0,0,2,1);
             linearizedStability = computeEquilibriaStability(equilibriumLocation, accelerationAngle, accelerationVariable, massParameter);

             if ( stepCounter % saveFrequency == 0)
             {


                 equilibriaCatalog[ accelerationVariable ] = equilibriumLocationWithIterations;
                 stabilityCatalog [ accelerationVariable ] = linearizedStability;



             }

             if (std::abs(std::fmod(accelerationVariable,accModCondition)) < accelerationStepSize/10.0  )
             {

                 deviationCatalog[  accelerationVariable * tudat::mathematical_constants::PI/  180.0] = computeDeviationAfterPropagation(equilibriumLocationWithIterations, accelerationVariable, accelerationAngle, massParameter, 1.0);

             }

             stepCounter++;

         }

         double continuationDirection = 1.0;
         double seedAngle = 0.0;

         writeResultsToFile(librationPointNr, accelerationAngle, "angle", seedAngle, continuationDirection,  equilibriaCatalog, stabilityCatalog, deviationCatalog);


    }


}




