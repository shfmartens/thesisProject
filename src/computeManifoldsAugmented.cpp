#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>
#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <math.h>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <exception>

#include "Tudat/Astrodynamics/Gravitation/jacobiEnergy.h"
#include "Tudat/Astrodynamics/Gravitation/librationPoint.h"
#include "Tudat/Astrodynamics/BasicAstrodynamics/celestialBodyConstants.h"
#include "Tudat/Astrodynamics/BasicAstrodynamics/astrodynamicsFunctions.h"

#include "propagateOrbit.h"
#include "propagateOrbitAugmented.h"
#include "computeManifolds.h"
#include "computeManifoldsAugmented.h"
#include "stateDerivativeModelAugmented.h"

#include "Tudat/Mathematics/BasicMathematics/function.h"

#include "Tudat/Mathematics/RootFinders/newtonRaphson.h"

#include "functions/contourLocationFunction.h"
#include "functions/contourLocationFunction1.h"

Eigen::MatrixXd retrieveSpacecraftProperties( const std::string spacecraftName)
{
    //Declare output matrix
    Eigen::VectorXd spacecraftProperties = Eigen::VectorXd( 4 );

    // Declare and initialize Earth-Moon system properties

    // double moonMass = 0.07346E24;               // [kg]   Obtained from NASA Earth Moon Fact Sheet [km];
    double moonMass = tudat::celestial_body_constants::MOON_GRAVITATIONAL_PARAMETER / tudat::physical_constants::GRAVITATIONAL_CONSTANT;
    double length_asterix   = 0.3844E6;      // [km]   Obtained from NASA Earth Moon Fact Sheet [km] and same as Koen!
    double time_asterix     = tudat::basic_astrodynamics::computeKeplerOrbitalPeriod(length_asterix * 1000 , tudat::celestial_body_constants::EARTH_GRAVITATIONAL_PARAMETER, moonMass  ) / (2.0 * tudat::mathematical_constants::PI);  // [s]
    //double gravNul        = 9.8065E-3; // [ km/s^2 ] Obtained from Source A. Paper A. Cox, K. Howell D. Folta
    double gravNul          = tudat::celestial_body_constants::EARTH_GRAVITATIONAL_PARAMETER / pow(tudat::celestial_body_constants::EARTH_EQUATORIAL_RADIUS,2);

    // Declare spacecraft variables

    double thrustMagnitude;
    double specificImpulse;
    double initialMass;

    if (spacecraftName == "DeepSpace") {
        thrustMagnitude = 1.0E-2;            // [N]
        initialMass = 486.0;                 // [kg]
        specificImpulse = 3200.0;            // [s] from Results from Deep Space 1 Technology validation mission max 3200, min 1900
    }
    if (spacecraftName == "Hayabasu") {
        thrustMagnitude = 24.0E-3;           // [N]
        initialMass = 510.0;                 // [kg]
        specificImpulse = 3000.0;            // [s], Hayabusa Asteroid Explorer Pow- ered by Ion Engines on the way to Earth,
    }
    if (spacecraftName == "Dawn") {
        thrustMagnitude = 92.7E-3;           // [N]
        initialMass = 1218.0;                // [kg]
        specificImpulse = 3200.0;            // [s], Hayabusa Asteroid Explorer Pow- ered by Ion Engines on the way to Earth,
    }
    if (spacecraftName == "LunarIceCube") {
        thrustMagnitude = 1.0E-3;           // [N]
        initialMass = 14.0;                 // [kg]
        specificImpulse = 2500.0;            // [s]
    }


    spacecraftProperties( 0 ) = 0.0;  // Nondimensional thrust magnitude
    spacecraftProperties( 1 ) = initialMass / initialMass; //nondimensional mass
    spacecraftProperties( 2 ) = ( -thrustMagnitude * length_asterix ) / ( specificImpulse * gravNul * time_asterix );
    spacecraftProperties( 3 ) = 1.0; //TODO,CHANGE INTO INPUT PARAMETER

    return spacecraftProperties;
}

double computeIntegralOfMotion ( const Eigen::VectorXd currentStateVector, const std::string spacecraftName, const std::string thrustPointing ,const double massParameter, const double currentTime) {

    Eigen::MatrixXd spatialStateVector = Eigen::MatrixXd::Zero(6,1);
    double integralOfMotion;

    // Fill vector with x,y position and velocity components of currentStateVector and z position and velocity components with initialStateVector
    spatialStateVector.block(0,0,6,1) = currentStateVector.block(0,0,6,1);

    double JacobiEnergy = tudat::gravitation::computeJacobiEnergy(massParameter, spatialStateVector);

    if (thrustPointing == "left" || thrustPointing == "right") {
    integralOfMotion = JacobiEnergy;

    } else
    {
        if (currentTime == 0.0) {
            integralOfMotion = JacobiEnergy / -2.0;
        } else {
            // calculate the accelerations
            Eigen::MatrixXd characteristics = retrieveSpacecraftProperties( spacecraftName );
            double thrustMagnitude = characteristics(0);
            double alpha = std::stod(thrustPointing);
            double xTermRelatedToThrust = (thrustMagnitude / currentStateVector.coeff(6,0) ) * std::cos( alpha * 2.0 * tudat::mathematical_constants::PI / 180.0 );
            double yTermRelatedToThrust = (thrustMagnitude / currentStateVector.coeff(6,0) ) * std::sin( alpha * 2.0 * tudat::mathematical_constants::PI / 180.0 );
            double innerProduct = spatialStateVector.coeff(0,0) * xTermRelatedToThrust  + spatialStateVector.coeff(1,0) * yTermRelatedToThrust;
            integralOfMotion = JacobiEnergy / -2.0 - innerProduct;
        }
    }
    return integralOfMotion;
}

bool checkIoMOnManifoldAugmentedOutsideBounds( Eigen::VectorXd currentStateVector, const double referenceIoM,
                                         const double massParameter, const std::string spacecraftName, const std::string thrustPointing, const double currentTime, const double maxIoMDeviation )

{

    //Declare function specific variables
    bool IoMDeviationOutsideBounds;

    double currentIoM = computeIntegralOfMotion(currentStateVector, spacecraftName, thrustPointing, massParameter, currentTime);

        if (std::abs(currentIoM - referenceIoM) < maxIoMDeviation)
        {
            IoMDeviationOutsideBounds = false;
            //std::cout << "Integral of motion deviation on manifold WITHIN bounds" << std::endl;
            //std::cout << "The current time is: " << currentTime << std::endl;
        } else
        {
            IoMDeviationOutsideBounds = true;
            std::cout << "Integral of motion deviation on manifold exceeded bounds" << std::endl;
        }

    return IoMDeviationOutsideBounds;
}

void reduceOvershootAtPoincareSectionU1U4Augmented( std::pair< Eigen::MatrixXd, double >& stateVectorInclSTMAndTime,
                                           std::pair< Eigen::MatrixXd, double >& previousStateVectorInclSTMAndTime,
                                           Eigen::MatrixXd& stateVectorInclSTM, double& currentTime, double& ySign,
                                           int& integrationDirection, const double& massParameter, std::string spacecraftName, std::string thrustPointing )
{
    // TODO join together with reduceOvershootAtPoincareSectionU2U3
    stateVectorInclSTMAndTime = previousStateVectorInclSTMAndTime;
    stateVectorInclSTM        = stateVectorInclSTMAndTime.first;
    currentTime               = stateVectorInclSTMAndTime.second;
    std::cout << "||y|| = " << stateVectorInclSTM(1, 0) << ", at start of iterative procedure" << std::endl;

    for ( int i = 5; i <= 12; i++ ) {

        double initialStepSize = pow(10,(static_cast<float>(-i)));
        double maximumStepSize = pow(10,(static_cast<float>(-i) + 1.0));

        while ( stateVectorInclSTMAndTime.first(1, 0) * ySign > 0 ) {
            stateVectorInclSTM                = stateVectorInclSTMAndTime.first;
            currentTime                       = stateVectorInclSTMAndTime.second;
            previousStateVectorInclSTMAndTime = stateVectorInclSTMAndTime;
            stateVectorInclSTMAndTime         = propagateOrbitAugmented(stateVectorInclSTM, massParameter, currentTime,
                                                               integrationDirection, spacecraftName, thrustPointing, initialStepSize, maximumStepSize);

            if ( stateVectorInclSTMAndTime.first(1, 0) * ySign < 0 ) {
                stateVectorInclSTMAndTime = previousStateVectorInclSTMAndTime;
                break;
            }
        }
    }
    std::cout << "||y|| = " << stateVectorInclSTM(1, 0) << ", at end of iterative procedure" << std::endl;
}


void reduceOvershootAtPoincareSectionU2U3Augmented( std::pair< Eigen::MatrixXd, double >& stateVectorInclSTMAndTime,
                                           std::pair< Eigen::MatrixXd, double >& previousStateVectorInclSTMAndTime,
                                           Eigen::MatrixXd& stateVectorInclSTM, double& currentTime, double& xDiffSign,
                                           int& integrationDirection, const double& massParameter, std::string spacecraftName, std::string thrustPointing )
{
    stateVectorInclSTMAndTime = previousStateVectorInclSTMAndTime;
    stateVectorInclSTM        = stateVectorInclSTMAndTime.first;
    currentTime               = stateVectorInclSTMAndTime.second;
    std::cout << "||x - (1-mu)|| = "                 << (stateVectorInclSTM(0, 0) - (1.0 - massParameter))
              << ", at start of iterative procedure" << std::endl;

    for ( int i = 5; i <= 12; i++ ) {

        double initialStepSize = pow(10,(static_cast<float>(-i)));
        double maximumStepSize = pow(10,(static_cast<float>(-i) + 1.0));

        while ( (stateVectorInclSTMAndTime.first(0, 0) - (1.0 - massParameter)) * xDiffSign > 0 ) {
            stateVectorInclSTM                = stateVectorInclSTMAndTime.first;
            currentTime                       = stateVectorInclSTMAndTime.second;
            previousStateVectorInclSTMAndTime = stateVectorInclSTMAndTime;
            stateVectorInclSTMAndTime         = propagateOrbitAugmented(stateVectorInclSTM, massParameter, currentTime,
                                                               integrationDirection, spacecraftName, thrustPointing, initialStepSize, maximumStepSize);

            if ( (stateVectorInclSTMAndTime.first(0, 0) - (1.0 - massParameter)) * xDiffSign < 0 ) {
                stateVectorInclSTMAndTime = previousStateVectorInclSTMAndTime;
                break;
            }
        }
    }
    std::cout << "||x - (1-mu)|| = "               << (stateVectorInclSTM(0, 0) - 1.0 + massParameter)
              << ", at end of iterative procedure" << std::endl;
}

void reduceOverShootInitialMass(std::pair< Eigen::MatrixXd, double >& stateVectorInclSTMAndTime,
                                std::pair< Eigen::MatrixXd, double >& previousStateVectorInclSTMAndTime,
                             Eigen::MatrixXd& stateVectorInclSTM, double& currentTime, int& integrationDirection,
                                const double& massParameter, std::string spacecraftName, std::string thrustPointing)
{
    stateVectorInclSTMAndTime = previousStateVectorInclSTMAndTime;
    stateVectorInclSTM        = stateVectorInclSTMAndTime.first;
    currentTime               = stateVectorInclSTMAndTime.second;
    std::cout << "||m - 1.0|| = "                 << (stateVectorInclSTM(6, 0) - (1.0))
              << ", at start of iterative procedure" << std::endl;

    for ( int i = 5; i <= 12; i++ ) {

        double initialStepSize = pow(10,(static_cast<float>(-i)));
        double maximumStepSize = pow(10,(static_cast<float>(-i) + 1.0));

        while ( (stateVectorInclSTMAndTime.first(6, 0) - 1.0 ) > 0 ) {
            stateVectorInclSTM                = stateVectorInclSTMAndTime.first;
            currentTime                       = stateVectorInclSTMAndTime.second;
            previousStateVectorInclSTMAndTime = stateVectorInclSTMAndTime;
            stateVectorInclSTMAndTime         = propagateOrbitAugmented(stateVectorInclSTM, massParameter, currentTime,
                                                               integrationDirection, spacecraftName, thrustPointing, initialStepSize, maximumStepSize);

            if ( stateVectorInclSTMAndTime.first(6, 0) - 1.0 <= 0 ) {
                stateVectorInclSTMAndTime = previousStateVectorInclSTMAndTime;
                break;
            }
        }
    }
    std::cout << "||m - 1.0|| = "               << (stateVectorInclSTM(6, 0) - 1.0 )
              << ", at end of iterative procedure" << std::endl;

}

double ycontourStoppingCondition ( const double referenceIoM, const double massParameter) {

    double contourCondition;

    // Create object containing the functions.
    // boost::shared_ptr< LibrationPointLocationFunction1 > LibrationPointLocationFunction = boost::make_shared< LibrationPointLocationFunction1 >( 1, massParameter );
    std::shared_ptr<contourLocationFunction1> contourLocationFunction = std::make_shared< contourLocationFunction1 > (1, massParameter, referenceIoM);

    // The termination condition.
    tudat::root_finders::NewtonRaphson::TerminationFunction terminationConditionFunction =
            boost::bind( &tudat::root_finders::termination_conditions::RootAbsoluteToleranceTerminationCondition< double >::checkTerminationCondition,
                         boost::make_shared< tudat::root_finders::termination_conditions::RootAbsoluteToleranceTerminationCondition< double > >(
                                 contourLocationFunction->getTrueRootAccuracy( ), contourLocationFunction->getMaxIterations( ) ), _1, _2, _3, _4, _5 );

    // Test Newton-Raphson object.
    tudat::root_finders::NewtonRaphson newtonRaphson( terminationConditionFunction );

    // Let Newton-Raphson search for the root.
    contourCondition = newtonRaphson.execute( contourLocationFunction, contourLocationFunction->getInitialGuess( ) );
    return contourCondition;
}
void writeAugmentedManifoldStateHistoryToFile( std::map< int, std::map< int, std::map< double, Eigen::Vector7d > > >& manifoldAugmentedStateHistory,
                                      const int& orbitNumber, const int& librationPointNr, const std::string& orbitType, const std::string spacecraftName, const std::string thrustPointing ) {
    std::string fileNameStateVector;
    std::ofstream textFileStateVectors;
    Eigen::MatrixXd characteristics = retrieveSpacecraftProperties(spacecraftName);

    std::vector<std::string> fileNamesStateVectors = {"L" + std::to_string(librationPointNr) + "_" + orbitType + "_" + std::to_string(orbitNumber) + "_" + spacecraftName + "_" + std::to_string(characteristics(0)) + "_" + thrustPointing + "_W_S_plus.txt",
                                                      "L" + std::to_string(librationPointNr) + "_" + orbitType + "_" + std::to_string(orbitNumber) + "_" + spacecraftName + "_" + std::to_string(characteristics(0)) + "_" + thrustPointing + "_W_S_min.txt",
                                                      "L" + std::to_string(librationPointNr) + "_" + orbitType + "_" + std::to_string(orbitNumber) + "_" + spacecraftName + "_" + std::to_string(characteristics(0)) + "_" + thrustPointing + "_W_U_plus.txt",
                                                      "L" + std::to_string(librationPointNr) + "_" + orbitType + "_" + std::to_string(orbitNumber) + "_" + spacecraftName + "_" + std::to_string(characteristics(0)) + "_" + thrustPointing + "_W_U_min.txt"};

    // For all four manifolds
    for( auto const &ent1 : manifoldAugmentedStateHistory ) {
        fileNameStateVector = fileNamesStateVectors.at(ent1.first);

        remove(("../data/raw/manifolds/augmented/" + fileNameStateVector).c_str());
        textFileStateVectors.open(("../data/raw/manifolds/augmented/" + fileNameStateVector));
        textFileStateVectors.precision(14);

        // For all numberOfTrajectoriesPerManifold
        for( auto const &ent2 : ent1.second ) {
            // For all states on manifold trajectory
            for( auto const &ent3 : ent2.second ) {
                textFileStateVectors << std::left    << std::scientific << std::setw(25) << ent3.first
                                     << std::setw(25) << ent3.second(0) << std::setw(25) << ent3.second(1)
                                     << std::setw(25) << ent3.second(2) << std::setw(25) << ent3.second(3)
                                     << std::setw(25) << ent3.second(4) << std::setw(25) << ent3.second(5)
                                     << std::setw(25) << ent3.second(6) << std::endl;
            }
        }
        textFileStateVectors.close();
        textFileStateVectors.clear();
    }

};

void computeManifoldsAugmented( const Eigen::Vector6d initialStateVector, const double orbitalPeriod, const int orbitNumber,
                       const int librationPointNr, const std::string orbitType, const std::string spacecraftName, const std::string thrustPointing, const double massParameter,
                       const double eigenvectorDisplacementFromOrbit, const int numberOfTrajectoriesPerManifold,
                       const int saveFrequency, const bool saveEigenvectors,
                       const double maximumIntegrationTimeManifoldTrajectories, const double maxEigenvalueDeviation )
{
    // Set output maximum precision

    std::cout.precision(std::numeric_limits<double>::digits10);

    double integralOfMotionOnOrbit = computeIntegralOfMotion( initialStateVector, spacecraftName, thrustPointing, massParameter, 0.0 );
    std::cout << "\nInitial state vector:" << std::endl << initialStateVector       << std::endl
              << "\nwith IOM: " << integralOfMotionOnOrbit   << ", T: " << orbitalPeriod << std::endl;
    double referenceIoMOnManifold;

    //double contourCondition         = ycontourStoppingCondition(integralOfMotionOnOrbit, massParameter );

    //std::cout << "CONTOUR STOPPING CONDITION IS" << contourCondition << std::endl;

    // Propagate the initialStateVector for a full period and write output to file.
    std::map< double, Eigen::MatrixXd > stateTransitionMatrixHistory;
    Eigen::MatrixXd stateVectorInclSTM = propagateOrbitWithStateTransitionMatrixToFinalCondition(getFullInitialState( initialStateVector ), massParameter, orbitalPeriod, 1, stateTransitionMatrixHistory, 1, 0.0 ).first;

    const unsigned int numberOfPointsOnPeriodicOrbit = stateTransitionMatrixHistory.size();
    std::cout << "numberOfPointsOnPeriodicOrbit: " << numberOfPointsOnPeriodicOrbit << std::endl;

    // Determine the eigenvector directions of the (un)stable subspace of the monodromy matrix
    Eigen::MatrixXd monodromyMatrix = stateVectorInclSTM.block(0,1,6,6);

    Eigen::Vector6d stableEigenvector;
    Eigen::Vector6d unstableEigenvector;

    try {
        determineStableUnstableEigenvectors( monodromyMatrix, stableEigenvector, unstableEigenvector, maxEigenvalueDeviation );
    }
    catch( const std::exception& ) {
        return;
    }
    std::cout << "EIGEN VECTORS COMPUTED: " <<  std::endl;
    std::cout << "STABLE EIGENVECTOR : " << stableEigenvector << std::endl;
    std::cout << "UNSTABLE EIGENVECTOR : " << unstableEigenvector << std::endl;
    // The sign of the x-component of the eigenvector is determined, which is used to determine the eigenvector offset direction (interior/exterior manifold)
    double stableEigenvectorSign   = determineEigenvectorSign( stableEigenvector );
    double unstableEigenvectorSign = determineEigenvectorSign( unstableEigenvector );

    int integrationDirection;
    double currentTime;
    double offsetSign;
    Eigen::VectorXd monodromyMatrixEigenvector;
    Eigen::Vector6d localStateVector           = Eigen::Vector6d::Zero(6);
    Eigen::Vector6d localNormalizedEigenvector = Eigen::Vector6d::Zero(6);
    Eigen::MatrixXd manifoldStartingState      = Eigen::MatrixXd::Zero(6, 7);

    Eigen::MatrixXd manifoldAugmentedStartingState  = Eigen::MatrixXd::Zero(7, 8);

    std::vector<double> offsetSigns            = {1.0 * stableEigenvectorSign, -1.0 * stableEigenvectorSign, 1.0 * unstableEigenvectorSign, -1.0 * unstableEigenvectorSign};
    std::vector<Eigen::VectorXd> eigenVectors  = {stableEigenvector, stableEigenvector, unstableEigenvector, unstableEigenvector};
    std::vector<int> integrationDirections     = {-1, -1, 1, 1};
    std::pair< Eigen::MatrixXd, double >                                            stateVectorInclSTMAndTime;
    std::pair< Eigen::MatrixXd, double >                                            previousStateVectorInclSTMAndTime;
    std::map< int, std::map< int, std::map< double, Eigen::Vector7d > > >           manifoldAugmentedStateHistory;  // 1. per manifold 2. per trajectory 3. per time-step
    std::map< int, std::map< int, std::pair< Eigen::Vector6d, Eigen::Vector6d > > > eigenvectorStateHistory;  // 1. per manifold 2. per trajectory 3. direction and location

    std::cout << "START MANIFOLD INITIAL CONDITION AND INTEGRATION COMPUTATION: " <<  std::endl;

    for ( int manifoldNumber = 0; manifoldNumber < 4; manifoldNumber++ ) {

        bool fullManifoldComputed       = false;
        bool IoMOutsideBounds           = false;
        bool ySignSet                   = false;
        bool xDiffSignSet               = false;
        double ySign                    = 0.0;
        double xDiffSign                = 0.0;
        //double contourCondition         = ycontourStoppingCondition(integralOfMotionOnOrbit, massParameter );
        double contourCondition         = 0.5;
        //std::cout << "CONTOUR STOPPING CONDITION IS" << contourCondition << std::endl;
        offsetSign                 = offsetSigns.at(manifoldNumber);
        monodromyMatrixEigenvector = eigenVectors.at(manifoldNumber);
        integrationDirection       = integrationDirections.at(manifoldNumber);

        // TODO replace with text (like interior/exterior unstable/stable)
        std::cout << "\n\nManifold: " << manifoldNumber << "\n" << std::endl;

        // Determine the total number of points along the periodic orbit to start the manifolds.
        for ( int trajectoryOnManifoldNumber = 0; trajectoryOnManifoldNumber < numberOfTrajectoriesPerManifold; trajectoryOnManifoldNumber++ ) {

            int indexCount    = 0;
            int stepCounter   = 1;
            auto indexOnOrbit = static_cast <int> (std::floor(trajectoryOnManifoldNumber * numberOfPointsOnPeriodicOrbit / numberOfTrajectoriesPerManifold));

            Eigen::MatrixXd stateTransitionMatrix;
            for ( auto const& it : stateTransitionMatrixHistory ) {
                if ( indexCount == indexOnOrbit ) {
                    stateTransitionMatrix = it.second.block(0, 1, 6, 6);
                    localStateVector = it.second.block(0, 0, 6, 1);
                    break;
                }
                indexCount += 1;
            }

            std::cout << "APPLY DISPLACEMENTS: " <<  std::endl;
            // Apply displacement epsilon from the periodic orbit at <numberOfTrajectoriesPerManifold> locations on the final orbit.
            localNormalizedEigenvector = (stateTransitionMatrix * monodromyMatrixEigenvector).normalized();
            manifoldStartingState      = getFullInitialState( localStateVector + offsetSign * eigenvectorDisplacementFromOrbit * localNormalizedEigenvector );

            // Obtain the CR3BP-LT State
            Eigen::MatrixXd satelliteCharacteristic  = retrieveSpacecraftProperties(spacecraftName);
            auto initialMass = static_cast<Eigen::Vector1d>( satelliteCharacteristic(1) );
            auto stableInitialMass = static_cast<Eigen::Vector1d>( satelliteCharacteristic(3) );
            manifoldAugmentedStartingState = getFullAugmentedInitialState(manifoldStartingState, initialMass, stableInitialMass, integrationDirection );

//            std::cout << std::endl
//                      << "================================================"                               << std::endl
//                      << "Integral of Motion before offset                "   << integralOfMotionOnOrbit    << "    " << std::endl
//                      << "Time after offset                               "   << currentTime                << "    " << std::endl
//                      << "Integral of Motion after offset:                 " << computeIntegralOfMotion(manifoldAugmentedStartingState, spacecraftName, thrustPointing, massParameter, currentTime)   << "    " << std::endl
//                      << "================================================"                               << std::endl;
            if ( saveEigenvectors ) {
                eigenvectorStateHistory[ manifoldNumber ][ trajectoryOnManifoldNumber ] = std::make_pair(localNormalizedEigenvector, localStateVector);
            }
            if ( saveFrequency >= 0 ) {
                manifoldAugmentedStateHistory[ manifoldNumber ][ trajectoryOnManifoldNumber ][ 0.0 ] = manifoldAugmentedStartingState.block( 0, 0, 7, 1 );
            }

            stateVectorInclSTMAndTime = propagateOrbitAugmented(manifoldAugmentedStartingState, massParameter, 0.0, integrationDirection, spacecraftName, thrustPointing);
            stateVectorInclSTM        = stateVectorInclSTMAndTime.first;
            currentTime               = stateVectorInclSTMAndTime.second;
//            std::cout << "================================================"                               << std::endl
//                      << "currentTime BEFORE START OF LOOP                "   << currentTime   << "    " << std::endl
//                      << "stateVectorInclSTM                              "   << stateVectorInclSTM   << "    " << std::endl
//                      << "stateVectorInclSTMBLOCK                         "   << stateVectorInclSTM.block(0,0,7,1)   << "    " << std::endl
//                      << "currentTime BEFORE START OF LOOP                "   << currentTime   << "    " << std::endl
//                      << "================================================"                               << std::endl;

            // Set the reference IOM
            if (thrustPointing == "left" || thrustPointing == "right") {
                referenceIoMOnManifold = integralOfMotionOnOrbit;
            } else {
                referenceIoMOnManifold = computeIntegralOfMotion(stateVectorInclSTM.block(0,0,7,1), spacecraftName, thrustPointing, massParameter, currentTime);
            }
            std::cout << "Trajectory on manifold number: " << trajectoryOnManifoldNumber << std::endl;

            while ( (std::abs( currentTime ) <= maximumIntegrationTimeManifoldTrajectories) && !fullManifoldComputed ) {

                // Check whether trajectory still belongs to the same energy level
                IoMOutsideBounds = checkIoMOnManifoldAugmentedOutsideBounds(stateVectorInclSTM.block(0,0,7,1), referenceIoMOnManifold, massParameter, spacecraftName, thrustPointing, currentTime);
                fullManifoldComputed      = IoMOutsideBounds;

                // Check whether the spacecraft comes above its initial wet mass
                if ( (stateVectorInclSTM(6, 0) > 1.0 ) && (manifoldNumber == 0 || manifoldNumber == 1)) {

                    reduceOverShootInitialMass(stateVectorInclSTMAndTime, previousStateVectorInclSTMAndTime,
                                                 stateVectorInclSTM, currentTime, integrationDirection,
                                                 massParameter, spacecraftName, thrustPointing);
                    fullManifoldComputed = true;
                }

                // Determine sign of y when crossing x = 0  (U1, U4)
                if ( (stateVectorInclSTM(0, 0) < 0) && !ySignSet ) {
                    if ( stateVectorInclSTM(1, 0) < 0 ){
                        ySign = -1.0;
                    }
                    if ( stateVectorInclSTM(1, 0) > 0 ) {
                        ySign = 1.0;
                    }
                    ySignSet = true;
                }

                // Determine whether the trajectory approaches U2, U3 from the right or left (U2, U3)
                if ( !xDiffSignSet ) {
                    if ( (stateVectorInclSTM(0, 0) - (1.0 - massParameter)) < 0 ) {
                        xDiffSign = -1.0;
                    }
                    if ( (stateVectorInclSTM(0, 0) - (1.0 - massParameter)) > 0 ) {
                        xDiffSign = 1.0;
                    }
                    xDiffSignSet = true;
                }

                // Determine when the manifold crosses the x-axis again (U1, U4)
                if ( (stateVectorInclSTM(1, 0) * ySign < 0) && ySignSet ) {
                    reduceOvershootAtPoincareSectionU1U4Augmented(stateVectorInclSTMAndTime, previousStateVectorInclSTMAndTime,
                                                         stateVectorInclSTM, currentTime, ySign, integrationDirection,
                                                         massParameter, spacecraftName, thrustPointing);
                    fullManifoldComputed = true;
                }

                // Cancel the stopping condition if the manifold crosses the Poincare section near the second primary outside of the Hill surface, only applicable for L2
                if ( ((stateVectorInclSTM(0, 0) - (1.0 - massParameter)) * xDiffSign < 0) &&
                         librationPointNr == 2 && abs( stateVectorInclSTM(1, 0) ) > contourCondition )  {

                    xDiffSign = -xDiffSign;
                }

                // Determine when the manifold crosses the Poincare section near the second primary (U2, U3)
                if ( ((stateVectorInclSTM(0, 0) - (1.0 - massParameter)) * xDiffSign < 0) &&
                        (librationPointNr == 1 ||
                         (librationPointNr == 2 && abs( stateVectorInclSTM(1, 0) ) < contourCondition ))) {
                    reduceOvershootAtPoincareSectionU2U3Augmented(stateVectorInclSTMAndTime,
                                                         previousStateVectorInclSTMAndTime,
                                                         stateVectorInclSTM, currentTime, xDiffSign,
                                                         integrationDirection, massParameter, spacecraftName, thrustPointing);
                    fullManifoldComputed = true;
                }

                // Write every nth integration step to file.

                if ( saveFrequency > 0 && ((stepCounter % saveFrequency == 0 || fullManifoldComputed) && !IoMOutsideBounds ) ) {
                    manifoldAugmentedStateHistory[ manifoldNumber ][ trajectoryOnManifoldNumber ][ currentTime ] = stateVectorInclSTM.block( 0, 0, 6, 1 );
                }

                if ( !fullManifoldComputed ){
                    // Propagate to next time step
                    previousStateVectorInclSTMAndTime = stateVectorInclSTMAndTime;
                    stateVectorInclSTMAndTime         = propagateOrbitAugmented(stateVectorInclSTM, massParameter, currentTime, integrationDirection, spacecraftName, thrustPointing);
                    stateVectorInclSTM                = stateVectorInclSTMAndTime.first;
                    currentTime                       = stateVectorInclSTMAndTime.second;
                    stepCounter++;

                    }
                }
            ySignSet             = false;
            xDiffSignSet         = false;
            fullManifoldComputed = false;

            }

    }
    if( saveFrequency >= 0 ) {
        std::cout << "manifold state is being saved: " << saveFrequency  << std::endl;
        writeAugmentedManifoldStateHistoryToFile( manifoldAugmentedStateHistory, orbitNumber, librationPointNr, orbitType, spacecraftName, thrustPointing );
    }
    if ( saveEigenvectors ) {
        writeEigenvectorStateHistoryToFile( eigenvectorStateHistory, orbitNumber, librationPointNr, orbitType );
    }

    std::cout << std::endl
              << "================================================"                             << std::endl
              << "                          "   << orbitNumber    << "                        " << std::endl
              << "Mass parameter: "             << massParameter                                << std::endl
              << "Spacecraft: "                 << spacecraftName                               << std::endl
              << "Thrust restriction: "         << thrustPointing                               << std::endl
              << "IoM at initial conditions: "    << referenceIoMOnManifold                          << std::endl
              << "IoM at end of manifold orbit: " << computeIntegralOfMotion(stateVectorInclSTM.block(0,0,5,1), spacecraftName, thrustPointing, massParameter) << std::endl
              << "T: " << orbitalPeriod                                                         << std::endl
              << "================================================"                             << std::endl;

}
