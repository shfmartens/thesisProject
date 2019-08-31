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

Eigen::VectorXd computeDerivativesUsingComplexStep(Eigen::VectorXcd designVector, double currentTime, Eigen::VectorXd thrustAndMassParameters, const double epsilon)
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

    // store the design vector from 24x1 into 6x4 method called oddStates
    Eigen::MatrixXcd oddStates(6,4);
    for (int i = 0; i < 4; i++)
    {
        oddStates.block(0,i,6,1) = designVector.segment(6*i,6);
    }

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

    // compute the evenStates using Tom's method and the MCOLL Method, see if they are still similar!
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

    //std::cout << "\noddStates: \n" << oddStates << std::endl;
    //std::cout << "oddStateDerivatives: \n" << oddStateDerivatives << std::endl;
    //std::cout << "evenStates: \n" << evenStates << std::endl;
    //std::cout << "evenStateDerivatives: \n" << evenStateDerivatives << std::endl;
    //std::cout << "defectConstraints: \n" << defectConstraints << std::endl;


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

    //std::cout << "outputVector complex multistep: \n" << outputVector << std::endl;

    return outputVector;


}

Eigen::VectorXd computeCollocationCorrection(const Eigen::MatrixXd defectVector, const Eigen::MatrixXd designVector, const Eigen::VectorXd timeIntervals, Eigen::VectorXd thrustAndMassParameters, const int numberOfCollocationPoints)
{
    // Declare and initialize main variables
    Eigen::VectorXd outputVector(designVector.rows());
    outputVector.setZero();
    double epsilon = 1.0E-10;
    std::complex<double> increment(0,epsilon);

    Eigen::MatrixXd jacobiMatrix(defectVector.rows(), designVector.rows() );
    jacobiMatrix.setZero();


    // Construct the partial derivatives by computing the derivatives per segment
    Eigen::MatrixXd jacobiSegment(18,24);
    jacobiSegment.setZero();

    for(int i = 0; i < (numberOfCollocationPoints - 1); i++)
    {
        // select local designVector
        Eigen::VectorXcd localDesignVector = designVector.block(i*18,0,24,1);
        double currentTime = timeIntervals(i);

        for(int j = 0; j < 24; j++)
        {
            // create the designVector for the specific column of the Jacobian
            Eigen::VectorXcd columnDesignVector = localDesignVector;
            columnDesignVector(j) = columnDesignVector(j) + increment;

            // compute the derivatives /// One MISTAKE INTO STATE DERIVATIVE COMPUTATION with x Acceleration, keeps giving slightly other numbers
            Eigen::VectorXd jacobiColumn(18);
            jacobiColumn.setZero();
            jacobiColumn = computeDerivativesUsingComplexStep(columnDesignVector, currentTime, thrustAndMassParameters,epsilon);

            jacobiSegment.block(0,j,18,1) = jacobiColumn;

        }

        jacobiMatrix.block(i*18,i*18,18,24) = jacobiSegment;
    }




    // should I use umfpack or other things, compare to two different methods for sparsity
    Eigen::VectorXd outputVectorBDC(designVector.rows());
    Eigen::VectorXd outputVectorPIV(designVector.rows());



    outputVector = -1.0*jacobiMatrix.transpose()*(jacobiMatrix * jacobiMatrix.transpose()).inverse()*defectVector;
    outputVectorBDC = jacobiMatrix.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(defectVector);
    outputVectorPIV = jacobiMatrix.colPivHouseholderQr().solve(defectVector);


    std::cout << "\noutputVector - outputVectorBDC: " << (outputVector - outputVectorBDC).norm() << std::endl;
    std::cout << "outputVector - outputVectorPIV: " << (outputVector - outputVectorPIV).norm() << std::endl;
    std::cout << "outputVectorBDC - outputVector PIV: " << (outputVectorBDC - outputVectorPIV).norm() << std::endl;




    return outputVector;
}
