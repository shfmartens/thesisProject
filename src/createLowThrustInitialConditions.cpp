#include <fstream>
#include <iomanip>
#include <iostream>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>
#include <Eigen/Dense>
#include <boost/function.hpp>

#include "Tudat/Astrodynamics/BasicAstrodynamics/celestialBodyConstants.h"
#include "Tudat/Astrodynamics/Gravitation/librationPoint.h"
#include "Tudat/Astrodynamics/Gravitation/jacobiEnergy.h"

#include "createInitialConditions.h"
#include "createLowThrustInitialConditions.h"
#include "applyDifferentialCorrection.h"
#include "checkEigenvalues.h"
#include "propagateOrbit.h"
#include "propagateOrbitAugmented.h"
//#include "richardsonThirdOrderApproximation.h"
#include "applyDifferentialCorrectionAugmented.h"

void appendResultsVectorAugmented(const double hamiltonian, const double orbitalPeriod, const Eigen::VectorXd& initialStateVector,
        const Eigen::MatrixXd& stateVectorInclSTM, std::vector< Eigen::VectorXd >& initialConditions )
{
    Eigen::VectorXd tempInitialCondition = Eigen::VectorXd( 112 );

    // Add Jacobi energy and orbital period
    tempInitialCondition( 0 ) = hamiltonian;
    tempInitialCondition( 1 ) = orbitalPeriod;

    // Add initial condition of periodic solution
    for (int i = 0; i <= 9; i++){
        tempInitialCondition( i + 2 ) = initialStateVector( i );
    }

    // Add Monodromy matrix
    tempInitialCondition( 10 ) = stateVectorInclSTM(0, 1);
    tempInitialCondition( 11 ) = stateVectorInclSTM(0, 2);
    tempInitialCondition( 12 ) = stateVectorInclSTM(0, 3);
    tempInitialCondition( 13 ) = stateVectorInclSTM(0, 4);
    tempInitialCondition( 14 ) = stateVectorInclSTM(0, 5);
    tempInitialCondition( 15 ) = stateVectorInclSTM(0, 6);
    tempInitialCondition( 16 ) = stateVectorInclSTM(0, 7);
    tempInitialCondition( 17 ) = stateVectorInclSTM(0, 8);
    tempInitialCondition( 18 ) = stateVectorInclSTM(0, 9);
    tempInitialCondition( 19 ) = stateVectorInclSTM(0, 10);

    tempInitialCondition( 20 ) = stateVectorInclSTM(1, 1);
    tempInitialCondition( 21 ) = stateVectorInclSTM(1, 2);
    tempInitialCondition( 22 ) = stateVectorInclSTM(1, 3);
    tempInitialCondition( 23 ) = stateVectorInclSTM(1, 4);
    tempInitialCondition( 24 ) = stateVectorInclSTM(1, 5);
    tempInitialCondition( 25 ) = stateVectorInclSTM(1, 6);
    tempInitialCondition( 26 ) = stateVectorInclSTM(1, 7);
    tempInitialCondition( 27 ) = stateVectorInclSTM(1, 8);
    tempInitialCondition( 28 ) = stateVectorInclSTM(1, 9);
    tempInitialCondition( 29 ) = stateVectorInclSTM(1, 10);

    tempInitialCondition( 30 ) = stateVectorInclSTM(2, 1);
    tempInitialCondition( 31 ) = stateVectorInclSTM(2, 2);
    tempInitialCondition( 32 ) = stateVectorInclSTM(2, 3);
    tempInitialCondition( 33 ) = stateVectorInclSTM(2, 4);
    tempInitialCondition( 34 ) = stateVectorInclSTM(2, 5);
    tempInitialCondition( 35 ) = stateVectorInclSTM(2, 6);
    tempInitialCondition( 36 ) = stateVectorInclSTM(2, 7);
    tempInitialCondition( 37 ) = stateVectorInclSTM(2, 8);
    tempInitialCondition( 38 ) = stateVectorInclSTM(2, 9);
    tempInitialCondition( 39 ) = stateVectorInclSTM(2, 10);

    tempInitialCondition( 40 ) = stateVectorInclSTM(3, 1);
    tempInitialCondition( 41 ) = stateVectorInclSTM(3, 2);
    tempInitialCondition( 42 ) = stateVectorInclSTM(3, 3);
    tempInitialCondition( 43 ) = stateVectorInclSTM(3, 4);
    tempInitialCondition( 44 ) = stateVectorInclSTM(3, 5);
    tempInitialCondition( 45 ) = stateVectorInclSTM(3, 6);
    tempInitialCondition( 46 ) = stateVectorInclSTM(3, 7);
    tempInitialCondition( 47 ) = stateVectorInclSTM(3, 8);
    tempInitialCondition( 48 ) = stateVectorInclSTM(3, 9);
    tempInitialCondition( 49 ) = stateVectorInclSTM(3, 10);

    tempInitialCondition( 50 ) = stateVectorInclSTM(4, 1);
    tempInitialCondition( 51 ) = stateVectorInclSTM(4, 2);
    tempInitialCondition( 52 ) = stateVectorInclSTM(4, 3);
    tempInitialCondition( 53 ) = stateVectorInclSTM(4, 4);
    tempInitialCondition( 54 ) = stateVectorInclSTM(4, 5);
    tempInitialCondition( 55 ) = stateVectorInclSTM(4, 6);
    tempInitialCondition( 56 ) = stateVectorInclSTM(4, 7);
    tempInitialCondition( 57 ) = stateVectorInclSTM(4, 8);
    tempInitialCondition( 58 ) = stateVectorInclSTM(4, 9);
    tempInitialCondition( 59 ) = stateVectorInclSTM(4, 10);

    tempInitialCondition( 60 ) = stateVectorInclSTM(5, 1);
    tempInitialCondition( 61 ) = stateVectorInclSTM(5, 2);
    tempInitialCondition( 62 ) = stateVectorInclSTM(5, 3);
    tempInitialCondition( 63 ) = stateVectorInclSTM(5, 4);
    tempInitialCondition( 64 ) = stateVectorInclSTM(5, 5);
    tempInitialCondition( 65 ) = stateVectorInclSTM(5, 6);
    tempInitialCondition( 66 ) = stateVectorInclSTM(5, 7);
    tempInitialCondition( 67 ) = stateVectorInclSTM(5, 8);
    tempInitialCondition( 68 ) = stateVectorInclSTM(5, 9);
    tempInitialCondition( 69 ) = stateVectorInclSTM(5, 10);

    tempInitialCondition( 70 ) = stateVectorInclSTM(6, 1);
    tempInitialCondition( 71 ) = stateVectorInclSTM(6, 2);
    tempInitialCondition( 72 ) = stateVectorInclSTM(6, 3);
    tempInitialCondition( 73 ) = stateVectorInclSTM(6, 4);
    tempInitialCondition( 74 ) = stateVectorInclSTM(6, 5);
    tempInitialCondition( 75 ) = stateVectorInclSTM(6, 6);
    tempInitialCondition( 76 ) = stateVectorInclSTM(6, 7);
    tempInitialCondition( 77 ) = stateVectorInclSTM(6, 8);
    tempInitialCondition( 78 ) = stateVectorInclSTM(6, 9);
    tempInitialCondition( 79 ) = stateVectorInclSTM(6, 10);

    tempInitialCondition( 80 ) = stateVectorInclSTM(7, 1);
    tempInitialCondition( 81 ) = stateVectorInclSTM(7, 2);
    tempInitialCondition( 82 ) = stateVectorInclSTM(7, 3);
    tempInitialCondition( 83 ) = stateVectorInclSTM(7, 4);
    tempInitialCondition( 84 ) = stateVectorInclSTM(7, 5);
    tempInitialCondition( 85 ) = stateVectorInclSTM(7, 6);
    tempInitialCondition( 86 ) = stateVectorInclSTM(7, 7);
    tempInitialCondition( 87 ) = stateVectorInclSTM(7, 8);
    tempInitialCondition( 88 ) = stateVectorInclSTM(7, 9);
    tempInitialCondition( 89 ) = stateVectorInclSTM(7, 10);

    tempInitialCondition( 90 ) = stateVectorInclSTM(8, 1);
    tempInitialCondition( 91 ) = stateVectorInclSTM(8, 2);
    tempInitialCondition( 92 ) = stateVectorInclSTM(8, 3);
    tempInitialCondition( 93 ) = stateVectorInclSTM(8, 4);
    tempInitialCondition( 94 ) = stateVectorInclSTM(8, 5);
    tempInitialCondition( 95 ) = stateVectorInclSTM(8, 6);
    tempInitialCondition( 96 ) = stateVectorInclSTM(8, 7);
    tempInitialCondition( 97 ) = stateVectorInclSTM(8, 8);
    tempInitialCondition( 98 ) = stateVectorInclSTM(8, 9);
    tempInitialCondition( 99 ) = stateVectorInclSTM(8, 10);

    tempInitialCondition( 100 ) = stateVectorInclSTM(9, 1);
    tempInitialCondition( 101 ) = stateVectorInclSTM(9, 2);
    tempInitialCondition( 102 ) = stateVectorInclSTM(9, 3);
    tempInitialCondition( 103 ) = stateVectorInclSTM(9, 4);
    tempInitialCondition( 104 ) = stateVectorInclSTM(9, 5);
    tempInitialCondition( 105 ) = stateVectorInclSTM(9, 6);
    tempInitialCondition( 106 ) = stateVectorInclSTM(9, 7);
    tempInitialCondition( 107 ) = stateVectorInclSTM(9, 8);
    tempInitialCondition( 108 ) = stateVectorInclSTM(9, 9);
    tempInitialCondition( 109 ) = stateVectorInclSTM(9, 10);


    initialConditions.push_back(tempInitialCondition);
}

void appendDifferentialCorrectionResultsVectorAugmented(
        const double hamiltonianHalfPeriod,  const Eigen::VectorXd& differentialCorrectionResult,
        std::vector< Eigen::VectorXd >& differentialCorrections )
{

    Eigen::VectorXd tempDifferentialCorrection = Eigen::VectorXd( 13 );
    tempDifferentialCorrection( 0 ) = differentialCorrectionResult( 22 );  // numberOfIterations
    tempDifferentialCorrection( 1 ) = hamiltonianHalfPeriod;  // jacobiEnergyHalfPeriod
    tempDifferentialCorrection( 2 ) = differentialCorrectionResult( 21 );  // currentTime
    for (int i = 11; i <= 20; i++)
    {
        tempDifferentialCorrection( i - 8 ) = differentialCorrectionResult( i );  // halfPeriodStateVector
    }


    differentialCorrections.push_back(tempDifferentialCorrection);

}

double getDefaultArcLengthAugmented(
        const double distanceIncrement,
        const Eigen::Vector6d& currentState )
{
   return distanceIncrement / currentState.segment( 0, 3 ).norm( );
}

double computeHamiltonian (const double massParameter, const Eigen::VectorXd stateVector) {

    double Hamiltonian;
    double jacobiEnergy;
    double alpha;
    double beta;
    double innerProduct;

    jacobiEnergy = tudat::gravitation::computeJacobiEnergy( massParameter, stateVector.segment(0,6) );
    alpha = stateVector(7) * tudat::mathematical_constants::PI / 180.0;
    beta = stateVector(8) * tudat::mathematical_constants::PI / 180.0;
    innerProduct = stateVector(0) * stateVector(6) * std::cos( alpha ) * std::cos( beta )
            + stateVector(1) * stateVector(6) * std::sin( alpha ) * std::cos( beta ) + stateVector(1) * stateVector(2) * std::sin( beta );;

    Hamiltonian = -2.0 * jacobiEnergy - innerProduct;
    return Hamiltonian;

}

Eigen::MatrixXd getCorrectedAugmentedInitialState( const Eigen::Vector6d& initialStateGuess, double orbitalPeriod, const int orbitNumber,
                                          const int librationPointNr, const double accelerationMagnitude, const double accelerationAngle,
                                          const double accelerationAngle2, const double initialMass, const double massParameter,
                                          std::vector< Eigen::VectorXd >& initialConditions,
                                          std::vector< Eigen::VectorXd >& differentialCorrections,
                                                   const double maxPositionDeviationFromPeriodicOrbit, double maxVelocityDeviationFromPeriodicOrbit ) {

    //Eigen::MatrixXd stateVectorInclSTM = Eigen::MatrixXd::Zero(8,9);

    // Construct CR3BP-LT state vector guess
    Eigen::VectorXd initialStateVector = Eigen::VectorXd::Zero(10);
    initialStateVector.segment(0, 6) = initialStateGuess.segment(0, 6);
    initialStateVector(6)            = accelerationMagnitude;
    initialStateVector(7)            = accelerationAngle;
    initialStateVector(8)            = accelerationAngle2;
    initialStateVector(9)            = initialMass;

    // Correct state vector guess
    Eigen::VectorXd differentialCorrectionResult = applyDifferentialCorrectionAugmented(
                librationPointNr, initialStateVector, orbitalPeriod, massParameter,
                maxPositionDeviationFromPeriodicOrbit, maxVelocityDeviationFromPeriodicOrbit, 1 );
    std::cout << "DC result: " << differentialCorrectionResult << std::endl;
    initialStateVector = differentialCorrectionResult.segment( 0, 10 );
    orbitalPeriod = differentialCorrectionResult( 11 );

    // Propagate the initialStateVector for a full period and write output to file.
    std::map< double, Eigen::VectorXd > stateHistory;
    Eigen::MatrixXd stateVectorInclSTM = propagateOrbitAugmentedToFinalCondition(
                getFullInitialStateAugmented( initialStateVector ), massParameter, orbitalPeriod, 1, stateHistory, 1000, 0.0 ).first;
    writeStateHistoryToFileAugmented( stateHistory, accelerationMagnitude, accelerationAngle, orbitNumber, librationPointNr, 1000, false );

    // Save results
    double hamiltonianHalfPeriod = computeHamiltonian( massParameter, differentialCorrectionResult.segment( 11, 10 ) );
    appendDifferentialCorrectionResultsVector( hamiltonianHalfPeriod, differentialCorrectionResult, differentialCorrections );

    double hamiltonian = computeHamiltonian( massParameter, stateVectorInclSTM.block( 0, 0, 10, 1 ));
    appendResultsVector( hamiltonian, orbitalPeriod, initialStateVector, stateVectorInclSTM, initialConditions );
return stateVectorInclSTM;
}

void writeFinalResultsToFilesAugmented( const int librationPointNr, const double accelerationMagnitude, const double accelerationAngle,
                               std::vector< Eigen::VectorXd > initialConditions,
                               std::vector< Eigen::VectorXd > differentialCorrections )
{
    // Prepare file for initial conditions
    remove(("../data/raw/orbits/L" + std::to_string(librationPointNr) + "_" + std::to_string( accelerationMagnitude) + "_" + std::to_string( accelerationAngle ) + "_initial_conditions.txt").c_str());
    std::ofstream textFileInitialConditions;
    textFileInitialConditions.open(("../data/raw/orbits/L" + std::to_string(librationPointNr) + "_" + std::to_string( accelerationMagnitude) + "_" + std::to_string( accelerationAngle ) + "_initial_conditions.txt"));
    textFileInitialConditions.precision(std::numeric_limits<double>::digits10);

    // Prepare file for differential correction
    remove(("../data/raw/orbits/L" + std::to_string(librationPointNr) + "_" + std::to_string( accelerationMagnitude) + "_" + std::to_string( accelerationAngle ) + "_differential_correction.txt").c_str());
    std::ofstream textFileDifferentialCorrection;
    textFileDifferentialCorrection.open(("../data/raw/orbits/L" + std::to_string(librationPointNr) + "_" + std::to_string( accelerationMagnitude) + "_" + std::to_string( accelerationAngle ) + "_differential_correction.txt"));
    textFileDifferentialCorrection.precision(std::numeric_limits<double>::digits10);

    // Write initial conditions to file
    for (unsigned int i=0; i<initialConditions.size(); i++) {

        textFileInitialConditions << std::left << std::scientific                                          << std::setw(25)
                                  << initialConditions[i][0]  << std::setw(25) << initialConditions[i][1]  << std::setw(25)
                                  << initialConditions[i][2]  << std::setw(25) << initialConditions[i][3]  << std::setw(25)
                                  << initialConditions[i][4]  << std::setw(25) << initialConditions[i][5]  << std::setw(25)
                                  << initialConditions[i][6]  << std::setw(25) << initialConditions[i][7]  << std::setw(25)
                                  << initialConditions[i][8]  << std::setw(25) << initialConditions[i][9]  << std::setw(25)
                                  << initialConditions[i][10] << std::setw(25) << initialConditions[i][11] << std::setw(25)
                                  << initialConditions[i][12] << std::setw(25) << initialConditions[i][13] << std::setw(25)
                                  << initialConditions[i][14] << std::setw(25) << initialConditions[i][15] << std::setw(25)
                                  << initialConditions[i][16] << std::setw(25) << initialConditions[i][17] << std::setw(25)
                                  << initialConditions[i][18] << std::setw(25) << initialConditions[i][19] << std::setw(25)
                                  << initialConditions[i][20] << std::setw(25) << initialConditions[i][21] << std::setw(25)
                                  << initialConditions[i][22] << std::setw(25) << initialConditions[i][23] << std::setw(25)
                                  << initialConditions[i][24] << std::setw(25) << initialConditions[i][25] << std::setw(25)
                                  << initialConditions[i][26] << std::setw(25) << initialConditions[i][27] << std::setw(25)
                                  << initialConditions[i][28] << std::setw(25) << initialConditions[i][29] << std::setw(25)
                                  << initialConditions[i][30] << std::setw(25) << initialConditions[i][31] << std::setw(25)
                                  << initialConditions[i][32] << std::setw(25) << initialConditions[i][33] << std::setw(25)
                                  << initialConditions[i][34] << std::setw(25) << initialConditions[i][35] << std::setw(25)
                                  << initialConditions[i][36] << std::setw(25) << initialConditions[i][37] << std::setw(25)
                                  << initialConditions[i][38] << std::setw(25) << initialConditions[i][39] << std::setw(25)
                                  << initialConditions[i][40] << std::setw(25) << initialConditions[i][41] << std::setw(25)
                                  << initialConditions[i][44] << std::setw(25) << initialConditions[i][45] << std::setw(25)
                                  << initialConditions[i][46] << std::setw(25) << initialConditions[i][47] << std::setw(25)
                                  << initialConditions[i][48] << std::setw(25) << initialConditions[i][49] << std::setw(25)
                                  << initialConditions[i][50] << std::setw(25) << initialConditions[i][51] << std::setw(25)
                                  << initialConditions[i][52] << std::setw(25) << initialConditions[i][53] << std::setw(25)
                                  << initialConditions[i][54] << std::setw(25) << initialConditions[i][55] << std::setw(25)
                                  << initialConditions[i][56] << std::setw(25) << initialConditions[i][57] << std::setw(25)
                                  << initialConditions[i][58] << std::setw(25) << initialConditions[i][59] << std::setw(25)
                                  << initialConditions[i][60] << std::setw(25) << initialConditions[i][61] << std::setw(25)
                                  << initialConditions[i][62] << std::setw(25) << initialConditions[i][63] << std::setw(25)
                                  << initialConditions[i][64] << std::setw(25) << initialConditions[i][65] << std::setw(25)
                                  << initialConditions[i][66] << std::setw(25) << initialConditions[i][67] << std::setw(25)
                                  << initialConditions[i][68] << std::setw(25) << initialConditions[i][69] << std::setw(25)
                                  << initialConditions[i][70] << std::setw(25) << initialConditions[i][71] << std::setw(25)
                                  << initialConditions[i][72] << std::setw(25) << initialConditions[i][73] << std::setw(25)
                                  << initialConditions[i][74] << std::setw(25) << initialConditions[i][75] << std::setw(25)
                                  << initialConditions[i][76] << std::setw(25) << initialConditions[i][77] << std::setw(25)
                                  << initialConditions[i][78] << std::setw(25) << initialConditions[i][79] << std::setw(25)
                                  << initialConditions[i][80] << std::setw(25) << initialConditions[i][81] << std::setw(25)
                                  << initialConditions[i][82] << std::setw(25) << initialConditions[i][83] << std::setw(25)
                                  << initialConditions[i][84] << std::setw(25) << initialConditions[i][85] << std::setw(25)
                                  << initialConditions[i][86] << std::setw(25) << initialConditions[i][87] << std::setw(25)
                                  << initialConditions[i][88] << std::setw(25) << initialConditions[i][89] << std::setw(25)
                                  << initialConditions[i][90] << std::setw(25) << initialConditions[i][91] << std::setw(25)
                                  << initialConditions[i][92] << std::setw(25) << initialConditions[i][93] << std::setw(25)
                                  << initialConditions[i][94] << std::setw(25) << initialConditions[i][95] << std::setw(25)
                                  << initialConditions[i][96] << std::setw(25) << initialConditions[i][97] << std::setw(25)
                                  << initialConditions[i][98] << std::setw(25) << initialConditions[i][99] << std::setw(25)
                                  << initialConditions[i][100] << std::setw(25) << initialConditions[i][101] << std::setw(25)
                                  << initialConditions[i][102] << std::setw(25) << initialConditions[i][103] << std::setw(25)
                                  << initialConditions[i][104] << std::setw(25) << initialConditions[i][105] << std::setw(25)
                                  << initialConditions[i][106] << std::setw(25) << initialConditions[i][107] << std::setw(25)
                                  << initialConditions[i][108] << std::setw(25) << initialConditions[i][109] << std::setw(25)
                                  << initialConditions[i][110] << std::setw(25) << initialConditions[i][111] << std::endl;

        textFileDifferentialCorrection << std::left << std::scientific   << std::setw(25)
                                       << differentialCorrections[i][0]  << std::setw(25) << differentialCorrections[i][1 ]  << std::setw(25)
                                       << differentialCorrections[i][2 ]  << std::setw(25) << differentialCorrections[i][3]  << std::setw(25)
                                       << differentialCorrections[i][4]  << std::setw(25) << differentialCorrections[i][5]  << std::setw(25)
                                       << differentialCorrections[i][6]  << std::setw(25) << differentialCorrections[i][7]  << std::setw(25)
                                       << differentialCorrections[i][8]  << std::setw(25) << differentialCorrections[i][9]  << std::setw(25)
                                       << differentialCorrections[i][10]  << std::endl;
    }
}

void createLowThrustInitialConditions( const int librationPointNr, const double accelerationMagnitude, const double accelerationAngle,
                                       const double accelerationAngle2, const double initialMass,
                              const double massParameter,
                              const double maxPositionDeviationFromPeriodicOrbit, const double maxVelocityDeviationFromPeriodicOrbit,
                              const double maxEigenvalueDeviation,
                              const boost::function< double( const Eigen::Vector6d& ) > pseudoArcLengthFunction ) {

    std::cout << "\nCreate initial conditions:\n" << std::endl;
    std::string orbitType = "horizontal";

    // Set output maximum precision
    std::cout.precision(std::numeric_limits<double>::digits10);

    // Initialize state vectors and orbital periods
    Eigen::Vector6d initialStateVector = Eigen::VectorXd::Zero( 10 );
    Eigen::MatrixXd stateVectorInclSTM = Eigen::MatrixXd::Zero( 10, 11 );

    std::vector< Eigen::VectorXd > initialConditions;
    std::vector< Eigen::VectorXd > differentialCorrections;

    // Obtain ballistic initial guesses and refine them
    Eigen::Vector7d richardsonThirdOrderApproximationResultIteration1 =
            getInitialStateVectorGuess( librationPointNr, orbitType, 0 );
    Eigen::Vector7d richardsonThirdOrderApproximationResultIteration2 =
            getInitialStateVectorGuess( librationPointNr, orbitType, 1 );
    stateVectorInclSTM =  getCorrectedAugmentedInitialState(
                richardsonThirdOrderApproximationResultIteration1.segment(0,6), richardsonThirdOrderApproximationResultIteration1( 6 ), 0,
               librationPointNr, accelerationMagnitude, accelerationAngle, accelerationAngle2, initialMass, massParameter, initialConditions, differentialCorrections,
                maxPositionDeviationFromPeriodicOrbit, maxVelocityDeviationFromPeriodicOrbit );
//    //std::cout << "The stateVectorInclSTM is equal to TEST RESULT: " << stateVectorInclSTM << std::endl;

    writeFinalResultsToFilesAugmented( librationPointNr, accelerationMagnitude, accelerationAngle, initialConditions, differentialCorrections );

}
