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
#include "applyLineSearchAttenuation.h"

void rearrangeTemporaryVectors(const Eigen::VectorXd temporaryDesignVector, const Eigen::VectorXd temporaryDerivativeVector, Eigen::MatrixXd& oddStates, Eigen::MatrixXd& oddStatesDerivatives, const int numberOfCollocationPoints)
{
    for(int i = 0; i < (numberOfCollocationPoints - 1); i++)
    {
        // extract nodes and interior points of the segment
        Eigen::VectorXd segmentStates = temporaryDesignVector.segment(18*i,24);
        Eigen::VectorXd segmentDerivatives = temporaryDerivativeVector.segment(18*i,24);

        for(int j = 0; j < 4; j++)
        {
            Eigen::VectorXd particularState      = segmentStates.segment(j*6,6);
            Eigen::VectorXd particularDerivative = segmentDerivatives.segment(j*6,6);

            oddStates.block(i*6,j,6,1) = particularState;
            oddStatesDerivatives.block(i*6,j,6,1) = particularDerivative;

        }



    }

}


Eigen::VectorXd computeDesignVectorDerivatives(const Eigen::VectorXd temporaryDesignVector, const Eigen::VectorXd thrustAndMassParameters, const int numberOfCollocationPoints  )
{
    // declare relevant Variables
    Eigen::VectorXd outputVector(temporaryDesignVector.rows());
    outputVector.setZero();

    int numberOfInteriorNodes = (numberOfCollocationPoints - 1)*3+1;
    for(int i = 0; i < numberOfInteriorNodes; i++)
    {
        Eigen::VectorXd currentState = temporaryDesignVector.segment(6*i,6);
        Eigen::VectorXd fullInitialState(10,11);

        fullInitialState.block(0,0,6,1) = currentState;
        fullInitialState.block(6,0,4,1) = thrustAndMassParameters;
        fullInitialState.block(0,1,10,10).setIdentity();

        Eigen::MatrixXd stateDerivativeInclSTM = computeStateDerivativeAugmented(0.0, fullInitialState);

        outputVector.segment(6*i,6) = stateDerivativeInclSTM.block(0,0,6,1);

    }

    return outputVector;

}

void applyLineSearchAttenuation(const Eigen::VectorXd collocationCorrectionVector,  Eigen::MatrixXd& collocationDefectVector,  Eigen::MatrixXd& collocationDesignVector, const Eigen::VectorXd timeIntervals, const Eigen::VectorXd thrustAndMassParameters, const int numberOfCollocationPoints)
{

    Eigen::MatrixXd minimumNormDesignVector;
    Eigen::MatrixXd minimumNormDefectVector;


    Eigen::ArrayXd attenuationFactor = Eigen::ArrayXd::LinSpaced(10,0.1,1.0);
    for (int i = 0; i < 10; i++)
    {

        Eigen::MatrixXd collocationDefectVectorTemp((numberOfCollocationPoints-1)*18,1);
        Eigen::MatrixXd collocationDesignVectorTemp((numberOfCollocationPoints-1)*18+6,1);

        collocationDefectVectorTemp.setZero();
        collocationDesignVectorTemp.setZero();

        // correct the design vector
        Eigen::VectorXd temporaryDesignVector = collocationDesignVector + attenuationFactor(i) * collocationCorrectionVector;

        // compute the derivatives at each node and interior point
        Eigen::VectorXd temporaryDerivativeVector = computeDesignVectorDerivatives(temporaryDesignVector, thrustAndMassParameters, numberOfCollocationPoints  );

        // Store the design vector and derivative vector in correct format to be able to use the defectConstraints function
        Eigen::MatrixXd oddStates(6*(numberOfCollocationPoints-1),4);
        Eigen::MatrixXd oddStatesDerivatives(6*(numberOfCollocationPoints-1),4);

        rearrangeTemporaryVectors(temporaryDesignVector, temporaryDerivativeVector, oddStates, oddStatesDerivatives, numberOfCollocationPoints);

        computeCollocationDefects(collocationDefectVectorTemp, collocationDesignVectorTemp, oddStates, oddStatesDerivatives, timeIntervals, thrustAndMassParameters, numberOfCollocationPoints );



        if (i == 0)
        {
            //std::cout << "REACHED alpha: " << attenuationFactor(i) << std::endl;
            minimumNormDesignVector = collocationDesignVectorTemp;
            minimumNormDefectVector = collocationDefectVectorTemp;

        } else if ( collocationDefectVectorTemp.norm() < minimumNormDefectVector.norm() )
        {
            //std::cout << "\nREACHED alpha: " << attenuationFactor(i) << std::endl;
            //std::cout << "startingNormBeforeCorrection: " <<  collocationDefectVector.norm() << std::endl;
            //std::cout << "minimumNormDefectVectorNorm: " << minimumNormDefectVector.norm()<< std::endl;
            //std::cout << "collocationDefectVectorTemp: " << collocationDefectVectorTemp.norm()<< std::endl;


            minimumNormDesignVector = collocationDesignVectorTemp;
            minimumNormDefectVector = collocationDefectVectorTemp;
        }

    }

// Store the solution with minimum norm as the new vectors
collocationDefectVector =  minimumNormDefectVector;
collocationDesignVector =  minimumNormDesignVector;

}

