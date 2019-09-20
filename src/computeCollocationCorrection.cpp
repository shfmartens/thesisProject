#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/LU>
#include <iostream>
#include <math.h>

#include "Tudat/Astrodynamics/Gravitation/librationPoint.h"
#include "Tudat/Astrodynamics/BasicAstrodynamics/celestialBodyConstants.h"
#include "stateDerivativeModelAugmented.h"
#include "propagateOrbitAugmented.h"
#include <Eigen/Eigenvalues>
#include "applyCollocation.h"
#include "applyMeshRefinement.h"
#include "interpolatePolynomials.h"
#include "computeCollocationCorrection.h"

double computeComplexPhaseDerivative(const Eigen::VectorXcd currentDesignVector, const int numberOfCollocationPoints, const Eigen::VectorXd previousDesignVector, const double epsilon)
{

    // initialize variables
    double phaseIntegralDerivative = 0.0;
    int currentNumberOfSegments = numberOfCollocationPoints - 1;
    int currentNumberOfOddPoints = 3*currentNumberOfSegments+1;

    int previousNumberOfSegments = (((previousDesignVector.rows())/11)-1)/3;
    int previousNumberOfCollocationPoints = previousNumberOfSegments+1;
    int previousNumberOfOddPoints = 3*previousNumberOfSegments+1;


    // Compute the properties of the previous guess, necessary for interpolation and put them in complex format
    Eigen::VectorXd thrustAndMassParameters = previousDesignVector.segment(6,4);
    Eigen::MatrixXd oddStates(6*previousNumberOfSegments,4);
    Eigen::MatrixXd oddStateDerivatives(6*previousNumberOfSegments,4);
    Eigen::VectorXd timeIntervals(previousNumberOfSegments);


    computeSegmentProperties(previousDesignVector, thrustAndMassParameters, previousNumberOfCollocationPoints, oddStates,
                             oddStateDerivatives, timeIntervals);

    Eigen::VectorXcd thrustAndMassParametersComplex(4); thrustAndMassParameters.setZero();
    thrustAndMassParametersComplex= thrustAndMassParameters;
    Eigen::MatrixXcd oddStatesComplex(6*previousNumberOfSegments,4);  oddStatesComplex.setZero();
    oddStatesComplex = oddStates;
    Eigen::MatrixXcd oddStateDerivativesComplex(6*previousNumberOfSegments,4); oddStateDerivativesComplex.setZero();
    oddStateDerivativesComplex = oddStateDerivatives;
    Eigen::VectorXcd timeIntervalsComplex(previousNumberOfSegments); timeIntervalsComplex.setZero();
    timeIntervalsComplex = timeIntervals;

    // compute Time and Segment Information From phase information of the current guess
    Eigen::VectorXcd oddPointTimesDimensional(currentNumberOfOddPoints); oddPointTimesDimensional.setZero();
    Eigen::VectorXcd oddPointTimesNormalized(currentNumberOfOddPoints);  oddPointTimesNormalized.setZero();
    Eigen::VectorXd segmentVector(currentNumberOfOddPoints);            segmentVector.setZero();
    Eigen::VectorXcd previousDesignVectorComplex = previousDesignVector;
    Eigen::VectorXcd currentDesignVectorFullFormat(11*currentNumberOfOddPoints);  currentDesignVectorFullFormat.setZero();


    rewriteDesignVectorToFullFormatComplex(currentDesignVector, numberOfCollocationPoints, thrustAndMassParametersComplex, currentDesignVectorFullFormat);



    computeTimeAndSegmentInformationFromPhaseComplex(currentDesignVectorFullFormat, previousDesignVectorComplex, numberOfCollocationPoints, oddStatesComplex, previousNumberOfCollocationPoints,
                                                     oddPointTimesDimensional, oddPointTimesNormalized, segmentVector);


    // perform Interpolation and compute the derivatives
    Eigen::VectorXcd incrementOddPoints(6*currentNumberOfOddPoints);              incrementOddPoints.setZero();
    Eigen::VectorXcd currentGuessOddPoints(6*currentNumberOfOddPoints);           currentGuessOddPoints.setZero();
    Eigen::VectorXcd previousGuessOddPointsSynced(6*currentNumberOfOddPoints);    previousGuessOddPointsSynced.setZero();
    Eigen::VectorXcd previousGuessOddDerivatesSynced(6*currentNumberOfOddPoints);  previousGuessOddDerivatesSynced.setZero();
    Eigen::VectorXcd oddPointPhaseConstraints(currentNumberOfOddPoints); oddPointPhaseConstraints.setZero();

    for(int i = 0; i < currentNumberOfOddPoints; i++)
    {
        Eigen::VectorXcd oddPointStateVectorCurrentGuess(6); oddPointStateVectorCurrentGuess.setZero();
        Eigen::VectorXcd oddPointStateVectorPreviousGuess(6); oddPointStateVectorPreviousGuess.setZero();

        Eigen::VectorXcd oddPointDerivativePreviousGuess(6); oddPointDerivativePreviousGuess.setZero();

        // select relevant parameters for interpolation
        auto segmentNumber = static_cast<int>(segmentVector(i));
        std::complex<double> interpolationTime = oddPointTimesNormalized(i);
        std::complex<double> oddPointTime = oddPointTimesDimensional(i);
        std::complex<double> segmentTimeInterval = timeIntervalsComplex(segmentNumber);

        Eigen::MatrixXcd segmentOddStates = oddStatesComplex.block(6*segmentNumber,0,6,4);
        Eigen::MatrixXcd segmentOddStateDerivatives = oddStateDerivatives.block(6*segmentNumber,0,6,4);

        // perform interpolation
        Eigen::VectorXcd interpolatedOddPoint = computeStateViaPolynomialInterpolationComplex(segmentOddStates, segmentOddStateDerivatives, segmentTimeInterval, interpolationTime);
        Eigen::VectorXcd oddPointStateVectorInclParameters(10);
        oddPointStateVectorInclParameters.segment(0,6) = interpolatedOddPoint;
        oddPointStateVectorInclParameters.segment(6,4) = thrustAndMassParametersComplex;

        // Fill relevant variables
        oddPointStateVectorPreviousGuess= interpolatedOddPoint;
        oddPointDerivativePreviousGuess = computeComplexStateDerivative( interpolatedOddPoint, thrustAndMassParameters );
        oddPointStateVectorCurrentGuess = currentDesignVectorFullFormat.segment(i*11,6);

        // Store in constraint components in the vectors
        previousGuessOddPointsSynced.segment(6*i,6)     = oddPointStateVectorPreviousGuess;
        previousGuessOddDerivatesSynced.segment(6*i,6)  = oddPointDerivativePreviousGuess;
        currentGuessOddPoints.segment(6*i,6)            = oddPointStateVectorCurrentGuess;
        incrementOddPoints.segment(6*i,6) = oddPointStateVectorCurrentGuess - oddPointStateVectorPreviousGuess;

    }


    // Compute versions of the integral constraint
    Eigen::VectorXcd phaseConstraintPoincare (currentNumberOfOddPoints);    phaseConstraintPoincare.setZero();
    Eigen::VectorXcd phaseConstraintLiterature (currentNumberOfOddPoints);  phaseConstraintLiterature.setZero();

    double quantityCheck = 0.0;
    for(int i = 0; i < currentNumberOfOddPoints; i++)
    {
        Eigen::VectorXcd currentIncrement = incrementOddPoints.segment(6*i,6);
        Eigen::VectorXcd currentOddPoint = currentGuessOddPoints.segment(6*i,6);
        Eigen::VectorXcd previousOddDerivative = previousGuessOddDerivatesSynced.segment(6*i,6);

        std::complex<double> localPhaseConstraintPoincare = currentIncrement.transpose() * previousOddDerivative;
        std::complex<double> localphaseConstraintLiterature = currentOddPoint.transpose() * previousOddDerivative;

        phaseConstraintPoincare(i) = localPhaseConstraintPoincare;
        phaseConstraintLiterature(i) = localphaseConstraintLiterature;

    }




    // Compute derivative via complex multistep method
    phaseIntegralDerivative = ((phaseConstraintPoincare.sum()).imag())/epsilon;

//    std::cout  << "\nphaseConstraintPoincare: \n" << phaseConstraintPoincare << std::endl
//               << "phaseConstraintPoincare.sum(): \n" << phaseConstraintPoincare.sum() << std::endl
//                  << "phaseConstraintPoincare.sum().imag: \n" << (phaseConstraintPoincare.sum()).imag() << std::endl
//                  << "phaseIntegralDerivative: \n" << phaseIntegralDerivative << std::endl
//               << "====== FINISHED CHECK OF INTEGRAL CONSTRAINT ====== "<< std::endl;

    return phaseIntegralDerivative;
}

Eigen::VectorXcd computeComplexStateDerivative(const Eigen::VectorXcd singleOddState, Eigen::VectorXd thrustAndMassParameters)
{
    // declare relevant variables
    extern double massParameter;
    std::complex<double> massParameterComplex(massParameter,0.0);

    Eigen::VectorXcd outputVector(6);
    outputVector.setZero();

    // set the derivatives of position equal to velocities:
    outputVector.segment(0,3) = singleOddState.segment(3,3);

    // follow state derivative augmented principle
    std::complex<double> complexOne(1.0,0.0);

    std::complex<double> xPositionScaledSquared = (singleOddState(0)+massParameterComplex) * (singleOddState(0)+massParameterComplex);
    std::complex<double> xPositionScaledSquared2 = (complexOne - massParameterComplex-singleOddState(0)) * (complexOne - massParameterComplex-singleOddState(0));
    std::complex<double> yPositionScaledSquared = (singleOddState(1) * singleOddState(1) );
    std::complex<double> zPositionScaledSquared = (singleOddState(2) * singleOddState(2) );


    // Compute distances to primaries.
    std::complex<double> distanceToPrimaryBody   = sqrt(xPositionScaledSquared     + yPositionScaledSquared + zPositionScaledSquared);
    std::complex<double> distanceToSecondaryBody = sqrt(xPositionScaledSquared2 + yPositionScaledSquared + zPositionScaledSquared);

    std::complex<double> distanceToPrimaryCubed = distanceToPrimaryBody * distanceToPrimaryBody * distanceToPrimaryBody;
    std::complex<double> distanceToSecondaryCubed = distanceToSecondaryBody * distanceToSecondaryBody * distanceToSecondaryBody;

    // Set the derivative of the velocities to the accelerations including the low-thrust terms
    std::complex<double>  termRelatedToPrimaryBody   = (complexOne-massParameterComplex)/distanceToPrimaryCubed;
    std::complex<double>  termRelatedToSecondaryBody = massParameterComplex      /distanceToSecondaryCubed;
    double alpha = thrustAndMassParameters(1) * tudat::mathematical_constants::PI / 180.0;
    double beta = thrustAndMassParameters(2) * tudat::mathematical_constants::PI / 180.0;

    std::complex<double> thrustTermX  ( (thrustAndMassParameters(0) /thrustAndMassParameters(3)) * std::cos( alpha ) * std::cos( beta ), 0.0 );
    std::complex<double> thrustTermY  ( (thrustAndMassParameters(0) /thrustAndMassParameters(3)) * std::sin( alpha ) * std::cos( beta ), 0.0 );
    std::complex<double> thrustTermZ  ( (thrustAndMassParameters(0) /thrustAndMassParameters(3)) * std::sin( beta ), 0.0 );

    outputVector(3) = -termRelatedToPrimaryBody*(massParameterComplex+singleOddState(0)) + termRelatedToSecondaryBody*(complexOne-massParameterComplex-singleOddState(0)) + singleOddState(0) + 2.0*singleOddState(4) + thrustTermX;
    outputVector(4) = -termRelatedToPrimaryBody*singleOddState(1)                        - termRelatedToSecondaryBody*singleOddState(1)                                   + singleOddState(1) - 2.0*singleOddState(3) + thrustTermY;
    outputVector(5) = -termRelatedToPrimaryBody*singleOddState(2)                        - termRelatedToSecondaryBody*singleOddState(2) + thrustTermZ ;

    // verify construction by inputting an plotting oddStateDerivative in defectFunction and computing it here with complex values (without increment)

    return outputVector;
}

Eigen::VectorXd computeDerivativesUsingComplexStep(Eigen::VectorXcd designVector, std::complex<double> currentTime, Eigen::VectorXd thrustAndMassParameters, const double epsilon)
{
    // Retrieve relevant LGL quantities
    Eigen::MatrixXd oddTimesMatrix(8,8);            Eigen::MatrixXd evenTimesMatrix(8,3);
    Eigen::MatrixXd evenTimesMatrixDerivative(8,3); Eigen::MatrixXd weightingMatrixEvenStates(3,3);
    Eigen::MatrixXd AConstants(3,4);                Eigen::MatrixXd VConstants(3,4);
    Eigen::MatrixXd BConstants(3,4);                Eigen::MatrixXd WConstants(3,5);

    retrieveLegendreGaussLobattoConstaints("oddTimesMatrix", oddTimesMatrix);
    retrieveLegendreGaussLobattoConstaints("evenTimesMatrix", evenTimesMatrix);
    retrieveLegendreGaussLobattoConstaints("evenTimesMatrixDerivative", evenTimesMatrixDerivative);
    retrieveLegendreGaussLobattoConstaints("weightingMatrixEvenStates", weightingMatrixEvenStates);
    retrieveLegendreGaussLobattoConstaints("AConstants", AConstants);
    retrieveLegendreGaussLobattoConstaints("VConstants", VConstants);
    retrieveLegendreGaussLobattoConstaints("BConstants", BConstants);
    retrieveLegendreGaussLobattoConstaints("WConstants", WConstants);

    // store the design vector from 25x1 into 6x4 method called oddStates, exclude the time at starting node of segment!
    Eigen::MatrixXcd oddStates(6,4);
    oddStates.block(0,0,6,1) = designVector.segment(0,6);
    oddStates.block(0,1,6,1) = designVector.segment(7,6);
    oddStates.block(0,2,6,1) = designVector.segment(13,6);
    oddStates.block(0,3,6,1) = designVector.segment(19,6);

    // determine the oddStateDerivatives via another function (do not make a whole stateDerivativeAugmented model but only compute first column!)
    Eigen::MatrixXcd oddStateDerivatives(6,4);
    for (int i = 0; i < 4; i++)
    {
        Eigen::VectorXcd singleOddState(6,1);
        Eigen::VectorXcd singleOddStateDerivative(6,1);

        singleOddState = oddStates.block(0,i,6,1);
        singleOddStateDerivative.setZero();
        singleOddStateDerivative = computeComplexStateDerivative( singleOddState, thrustAndMassParameters );

        oddStateDerivatives.block(0,i,6,1) = singleOddStateDerivative;

    }

    // compute the evenStates using Tom's method!
    Eigen::MatrixXcd evenStates(6,3);
    evenStates.block(0,0,6,1) = AConstants(0,0) * oddStates.block(0,0,6,1) + AConstants(0,1) * oddStates.block(0,1,6,1) +
                                        AConstants(0,2) * oddStates.block(0,2,6,1) + AConstants(0,3) * oddStates.block(0,3,6,1) +
                          currentTime * ( VConstants(0,0) * oddStateDerivatives.block(0,0,6,1) + VConstants(0,1) * oddStateDerivatives.block(0,1,6,1) +
                                        VConstants(0,2) * oddStateDerivatives.block(0,2,6,1) + VConstants(0,3) * oddStateDerivatives.block(0,3,6,1) );
    evenStates.block(0,1,6,1) = AConstants(1,0) * oddStates.block(0,0,6,1) + AConstants(1,1) * oddStates.block(0,1,6,1) +
                                        AConstants(1,2) * oddStates.block(0,2,6,1) + AConstants(1,3) * oddStates.block(0,3,6,1) +
                          currentTime * ( VConstants(1,0) * oddStateDerivatives.block(0,0,6,1) + VConstants(1,1) * oddStateDerivatives.block(0,1,6,1) +
                                        VConstants(1,2) * oddStateDerivatives.block(0,2,6,1) + VConstants(1,3) * oddStateDerivatives.block(0,3,6,1) );
    evenStates.block(0,2,6,1) = AConstants(2,0) * oddStates.block(0,0,6,1) + AConstants(2,1) * oddStates.block(0,1,6,1) +
                                        AConstants(2,2) * oddStates.block(0,2,6,1) + AConstants(2,3) * oddStates.block(0,3,6,1) +
                          currentTime * ( VConstants(2,0) * oddStateDerivatives.block(0,0,6,1) + VConstants(2,1) * oddStateDerivatives.block(0,1,6,1) +
                                        VConstants(2,2) * oddStateDerivatives.block(0,2,6,1) + VConstants(2,3) * oddStateDerivatives.block(0,3,6,1) );

    // compute local defect state derivatives
    Eigen::MatrixXcd evenStateDerivatives(6,3);
    for (int j = 0; j < 3; j++)
    {

        Eigen::VectorXcd singleEvenState(6,1);
        Eigen::VectorXcd singleEvenStateDerivative(6,1);

        singleEvenState = evenStates.block(0,j,6,1);
        singleEvenStateDerivative.setZero();
        singleEvenStateDerivative = computeComplexStateDerivative( singleEvenState, thrustAndMassParameters );

        evenStateDerivatives.block(0,j,6,1) = singleEvenStateDerivative;

    }

    // compute the constraints xsi1, xsi2, xsi3
    Eigen::MatrixXcd defectConstraints(6,3);
    Eigen::VectorXcd xsi1(6);
    Eigen::VectorXcd xsi2(6);
    Eigen::VectorXcd xsi3(6);

    xsi1 = BConstants(0,0)*oddStates.block(0,0,6,1) + BConstants(0,1)*oddStates.block(0,1,6,1) +
           BConstants(0,2)*oddStates.block(0,2,6,1) + BConstants(0,3)*oddStates.block(0,3,6,1) + currentTime * (
           WConstants(0,0) * oddStateDerivatives.block(0,0,6,1) + WConstants(0,1) * evenStateDerivatives.block(0,0,6,1)  +
           WConstants(0,2) * oddStateDerivatives.block(0,1,6,1) + WConstants(0,3) * oddStateDerivatives.block(0,2,6,1)  +
           WConstants(0,4) * oddStateDerivatives.block(0,3,6,1));

    xsi2 = BConstants(1,0)*oddStates.block(0,0,6,1) + BConstants(1,1)*oddStates.block(0,1,6,1) +
           BConstants(1,2)*oddStates.block(0,2,6,1) + BConstants(1,3)*oddStates.block(0,3,6,1) + currentTime * (
           WConstants(1,0) * oddStateDerivatives.block(0,0,6,1) + WConstants(1,1) * oddStateDerivatives.block(0,1,6,1)  +
           WConstants(1,2) * evenStateDerivatives.block(0,1,6,1) + WConstants(1,3) * oddStateDerivatives.block(0,2,6,1)  +
           WConstants(1,4) * oddStateDerivatives.block(0,3,6,1));

    xsi3 = BConstants(2,0)*oddStates.block(0,0,6,1) + BConstants(2,1)*oddStates.block(0,1,6,1) +
           BConstants(2,2)*oddStates.block(0,2,6,1) + BConstants(2,3)*oddStates.block(0,3,6,1) + currentTime * (
           WConstants(2,0) * oddStateDerivatives.block(0,0,6,1) + WConstants(2,1) * oddStateDerivatives.block(0,1,6,1)  +
           WConstants(2,2) * oddStateDerivatives.block(0,2,6,1) + WConstants(2,3) * evenStateDerivatives.block(0,2,6,1)  +
           WConstants(2,4) * oddStateDerivatives.block(0,3,6,1));

    defectConstraints.block(0,0,6,1) = xsi1;
    defectConstraints.block(0,1,6,1) = xsi2;
    defectConstraints.block(0,2,6,1) = xsi3;


    // store the constraints in an 18x1 vector, extract imaginary parts and put them in a real vector, divide by 1.0E-10
    Eigen::VectorXcd defectVector(18);
    Eigen::VectorXd outputVector(18);
    defectVector.setZero();
    outputVector.setZero();

    defectVector.segment(0,6) = xsi1;
    defectVector.segment(6,6) = xsi2;
    defectVector.segment(12,6) = xsi3;

    // apply complex step method
    outputVector = (defectVector.imag() ) / epsilon;

    return outputVector;


}

Eigen::VectorXd computePeriodicityDerivativeUsingComplexStep(Eigen::VectorXcd initialState, Eigen::VectorXcd finalState, const double epsilon)
{

    Eigen::VectorXd outputVector(6);
    outputVector.setZero();

    Eigen::VectorXcd periodicityDefect =  initialState - finalState;

    //std::cout << "\nperiodicityDefect: \n"<< periodicityDefect << std::endl;

    outputVector = (periodicityDefect.imag() ) / epsilon;

    return outputVector;

}

double computePhasePeriodicityDerivativeUsingComplexStep(const Eigen::VectorXcd columnInitialState, const Eigen::MatrixXd phaseConstraintVector, const double epsilon)
{
    double outputVector = 0.0;


    Eigen::VectorXcd previousState = phaseConstraintVector.block(0,0,6,1);
    Eigen::VectorXcd initialDerivative = phaseConstraintVector.block(0,1,6,1);
    Eigen::VectorXcd increment = columnInitialState - previousState;
    std::complex<double> defect = increment.transpose()*initialDerivative;

    outputVector = defect.imag() / epsilon;

    return outputVector;

}

std::complex<double> computeComplexJacobi(const Eigen::VectorXcd currentState, const double massParameter)
{
    std::complex<double> outputVector(0.0,0.0);

    // compute distance w.r.t to primaries
    std::complex<double> xCoordinateToPrimaryBodySquared = ( currentState(0) + massParameter ) * ( currentState(0) + massParameter );
    std::complex<double> xCoordinateToSecondaryBodySquared = ( currentState(0) -1.0 + massParameter ) * ( currentState(0) -1.0 + massParameter  );
    std::complex<double> yCoordinateSquared = ( currentState(1) ) * ( currentState(1) );

    std::complex<double> distanceToPrimaryBody = sqrt(xCoordinateToPrimaryBodySquared + yCoordinateSquared );
    std::complex<double> distanceToSecondaryBody = sqrt(xCoordinateToSecondaryBodySquared + yCoordinateSquared );

    // compute velocity Norm
    std::complex<double> velocitySquared = currentState(3)*currentState(3) + currentState(4)*currentState(4) + currentState(5)*currentState(5);

    // compute complex Jacobi
    outputVector = ( currentState(0)*currentState(0) ) + ( currentState(1)*currentState(1) )
            + 2.0 * (1.0 - massParameter)/ distanceToPrimaryBody + 2.0 * massParameter/distanceToSecondaryBody
            - velocitySquared;

    return outputVector;
}

double computeHamiltonianDerivativeUsingComplexStep( const Eigen::VectorXcd currentState, const Eigen::VectorXd thrustAndMassParameters, const double hamiltonianTarget, const double epsilon, const double massParameter )
{

    // compute complex jacobi
    std::complex<double> complexJacobi = computeComplexJacobi(currentState,  massParameter);

    // compute the innter product
    double acceleration = thrustAndMassParameters(0);
    double alpha = thrustAndMassParameters(1) * tudat::mathematical_constants::PI / 180.0;
    double beta = thrustAndMassParameters(2) * tudat::mathematical_constants::PI / 180.0;

    std::complex<double> innerProduct = currentState(0) * acceleration * std::cos( alpha ) * std::cos( beta )
                                      + currentState(1) * acceleration * std::sin( alpha ) * std::cos( beta )
                                      + currentState(2) * acceleration * std::sin( beta );

    std::complex<double> adaptedHamiltonian = -0.5*complexJacobi -  innerProduct;
    std::complex<double> defect = hamiltonianTarget - adaptedHamiltonian;


   double outputVector = ( defect.imag() / epsilon );

    // compute a complex version of the Hamiltonian!
   return outputVector;


}

Eigen::VectorXd computeCollocationCorrection(const Eigen::MatrixXd defectVector, const Eigen::MatrixXd designVector, const Eigen::VectorXd timeIntervals, Eigen::VectorXd thrustAndMassParameters, const int numberOfCollocationPoints, const int continuationIndex, const Eigen::MatrixXd phaseConstraintVector)
{
    // Declare and initialize main variables
    Eigen::VectorXd outputVector(designVector.rows());
    outputVector.setZero();
    double epsilon = 1.0e-10;
    std::complex<double> increment(0.0,epsilon);
    int numberOfNodes = 3*(numberOfCollocationPoints - 1) + 1;
    Eigen::MatrixXd jacobiMatrix(defectVector.rows(), ( designVector.rows() ) );
    jacobiMatrix.setZero();

    std::cout << "designVector.rows(): " << designVector.rows() << std::endl;

    // Construct the partial derivatives by computing the derivatives per segment
    Eigen::MatrixXd jacobiSegment(18,26);
    Eigen::MatrixXd jacobiIntegralPhaseSegment(1,26);


    Eigen::MatrixXd jacobiPeriodicitySegment(6,jacobiMatrix.cols());
    Eigen::MatrixXd jacobiIntegralPhaseConstraint(1,jacobiMatrix.cols());
    Eigen::MatrixXd jacobiPhaseHamiltonianSegment(1,jacobiMatrix.cols());


    jacobiSegment.setZero();
    jacobiIntegralPhaseSegment.setZero();
    jacobiPeriodicitySegment.setZero();
    jacobiIntegralPhaseConstraint.setZero();
    jacobiPhaseHamiltonianSegment.setZero();

    for(int i = 0; i < (numberOfCollocationPoints - 1); i++)
    {
        // select local designVector
        Eigen::VectorXcd localDesignVector = designVector.block(i*19,0,26,1);

        for(int j = 0; j < 26; j++)
        {
            // create the designVector for the specific column of the Jacobian
            Eigen::VectorXcd columnDesignVector = localDesignVector;
            columnDesignVector(j) = columnDesignVector(j) + increment;
            std::complex<double> currentTime;
            currentTime = columnDesignVector( 25 ) - columnDesignVector(6);


            // compute the derivatives /// One MISTAKE INTO STATE DERIVATIVE COMPUTATION with x Acceleration, keeps giving slightly other numbers
            Eigen::VectorXd jacobiColumn(18);
            jacobiColumn.setZero();


            jacobiColumn = computeDerivativesUsingComplexStep(columnDesignVector, currentTime, thrustAndMassParameters,epsilon);

            jacobiSegment.block(0,j,18,1) = jacobiColumn;

        }

        // add segment to jacobiMatrix
        jacobiMatrix.block(i*18,i*19,18,26) = jacobiSegment;

        if (i == 0)
        {


            // Compute periodicity constraints at first state and phase constraint
            Eigen::VectorXcd initialState = designVector.block(0,0,6,1);
            Eigen::VectorXcd finalState = designVector.block(designVector.rows()-7,0,6,1);


            for (int j = 0; j <6; j++)
            {
                Eigen::VectorXcd columnInitialState = initialState;
                columnInitialState(j) = columnInitialState(j) + increment;

                Eigen::VectorXd periodicityColumn(6);
                double phaseDerivative = 0.0;

                periodicityColumn.setZero();

                periodicityColumn = computePeriodicityDerivativeUsingComplexStep(columnInitialState, finalState, epsilon);
                phaseDerivative = computePhasePeriodicityDerivativeUsingComplexStep(columnInitialState, phaseConstraintVector, epsilon);

                jacobiPhaseHamiltonianSegment(0,j) =  phaseDerivative;
                jacobiPeriodicitySegment.block(0,j,6,1) = periodicityColumn;
            }


        }

        if (i == (numberOfCollocationPoints - 2 ))
        {
            // Compute periodicity constraints at final constraint
            Eigen::VectorXcd initialState = designVector.block(0,0,6,1);
            Eigen::VectorXcd finalState = designVector.block((designVector.rows()-1)*7,0,6,1);

            for (int j = 0; j <6; j++)
            {
                Eigen::VectorXcd columnFinalState = finalState;
                columnFinalState(j) = columnFinalState(j) + increment;

                Eigen::VectorXd periodicityColumn(6);
                periodicityColumn.setZero();

                periodicityColumn = computePeriodicityDerivativeUsingComplexStep(initialState, columnFinalState, epsilon);


                jacobiPeriodicitySegment.block(0,( jacobiPeriodicitySegment.cols()-7+j ),6,1) = periodicityColumn;
            }


        }
    }



    for (int i = 0; i < jacobiMatrix.cols(); i++)
    {
        Eigen::VectorXcd inputDesignVector(jacobiMatrix.cols()); inputDesignVector.setZero();

        inputDesignVector = designVector.block(0,0,jacobiMatrix.cols(),1);

        inputDesignVector(i) = inputDesignVector(i) + increment;

         double phaseDerivative = computeComplexPhaseDerivative(inputDesignVector, numberOfCollocationPoints, phaseConstraintVector, epsilon );
         jacobiIntegralPhaseConstraint(0,i) =phaseDerivative;

    }

    std::cout << "computing JacobiIntegralPhaseConstraint completed " << std::endl;


    if (continuationIndex == 1)
    {
        jacobiMatrix.block( ( jacobiMatrix.rows()-7 ), 0, 6, jacobiMatrix.cols()) = jacobiPeriodicitySegment;
        jacobiMatrix.block( ( jacobiMatrix.rows()-1 ), 0, 1, jacobiMatrix.cols()) = jacobiIntegralPhaseConstraint;
    } else
    {
        jacobiMatrix.block( ( jacobiMatrix.rows()-6 ), 0, 6, jacobiMatrix.cols()) = jacobiPeriodicitySegment;

    }


    // should I use umfpack or other things, compare to two different methods for sparsity
    Eigen::VectorXd outputVectorBDC(designVector.rows());
    Eigen::VectorXd outputVectorPIV(designVector.rows());


    outputVector = -1.0*jacobiMatrix.transpose()*(jacobiMatrix * jacobiMatrix.transpose()).inverse()*defectVector;
    outputVectorBDC = jacobiMatrix.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(defectVector);
    outputVectorPIV = jacobiMatrix.colPivHouseholderQr().solve(defectVector);


    //::cout << "\noutputVector - outputVectorBDC: " << (outputVector - outputVectorBDC).norm() << std::endl;
    //std::cout << "outputVector - outputVectorPIV: " << (outputVector - outputVectorPIV).norm() << std::endl;
    //std::cout << "outputVectorBDC - outputVector PIV: " << (outputVectorBDC - outputVectorPIV).norm() << std::endl;




    return outputVector;
}
