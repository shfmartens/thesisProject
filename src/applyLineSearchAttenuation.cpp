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
        Eigen::VectorXd segmentStates = temporaryDesignVector.segment(19*i,26);
        Eigen::VectorXd segmentDerivatives = temporaryDerivativeVector.segment(19*i,26);

        for(int j = 0; j < 4; j++)
        {
            Eigen::VectorXd particularState (6);
            Eigen::VectorXd particularDerivative(6);

            if (j == 0)
            {
                particularState = segmentStates.segment(0,6);
                particularDerivative = segmentDerivatives.segment(0,6);
            } else
            {
                particularState = segmentStates.segment(j*6+1,6);
                particularDerivative = segmentDerivatives.segment(j*6+1,6);
            }

            oddStates.block(6*i,j,6,1) = particularState;
            oddStatesDerivatives.block(6*i,j,6,1) = particularDerivative;

        }



    }

}


Eigen::VectorXd computeDesignVectorDerivatives(const Eigen::VectorXd temporaryDesignVector, const Eigen::VectorXd thrustAndMassParameters, const int numberOfCollocationPoints  )
{
    // declare relevant Variables
    Eigen::VectorXd outputVector(temporaryDesignVector.rows());
    outputVector.setZero();

    // rewrite per segment!
    for(int i = 0; i < (numberOfCollocationPoints-1); i++)
    {
        Eigen::VectorXd segmentStateVector(26); Eigen::VectorXd segmentStateDerivative(26);
        segmentStateVector.setZero(); segmentStateDerivative.setZero();

        segmentStateVector = temporaryDesignVector.segment(i*19,26);

        for(int j = 0; j < 4; j++)
        {
             Eigen::VectorXd fullInitialState(10,11);
            if (j == 0)
            {
                fullInitialState.block(0,0,6,1) = segmentStateVector.segment(0,6);
                fullInitialState.block(6,0,4,1) = thrustAndMassParameters;
                fullInitialState.block(0,1,10,10).setIdentity();

                Eigen::MatrixXd stateDerivativeInclSTM = computeStateDerivativeAugmented(0.0, fullInitialState);
                segmentStateDerivative.segment(0,6) = stateDerivativeInclSTM.block(0,0,6,1);

            } else
            {
                fullInitialState.block(0,0,6,1) = segmentStateVector.segment(j*6+1,6);
                fullInitialState.block(6,0,4,1) = thrustAndMassParameters;
                fullInitialState.block(0,1,10,10).setIdentity();
                Eigen::MatrixXd stateDerivativeInclSTM = computeStateDerivativeAugmented(0.0, fullInitialState);
                segmentStateDerivative.segment(j*6+1,6) = stateDerivativeInclSTM.block(0,0,6,1);
            }

        }

         outputVector.segment(19*i,26) = segmentStateDerivative;
    }

//    int numberOfInteriorNodes = (numberOfCollocationPoints - 1)*3+1;
//    for(int i = 0; i < numberOfInteriorNodes; i++)
//    {
//        Eigen::VectorXd currentState = temporaryDesignVector.segment(6*i,6);
//        Eigen::VectorXd fullInitialState(10,11);

//        fullInitialState.block(0,0,6,1) = currentState;
//        fullInitialState.block(6,0,4,1) = thrustAndMassParameters;
//        fullInitialState.block(0,1,10,10).setIdentity();

//        Eigen::MatrixXd stateDerivativeInclSTM = computeStateDerivativeAugmented(0.0, fullInitialState);

//        outputVector.segment(6*i,6) = stateDerivativeInclSTM.block(0,0,6,1);

//    }

    return outputVector;

}

void recomputeTimeProperties(const Eigen::MatrixXd temporaryDesignVector, double& initialTime, Eigen::VectorXd& timeIntervals, const int numberOfCollocationPoints)
{
    initialTime = temporaryDesignVector(6,0);
    Eigen::VectorXd newTimeIntervals(numberOfCollocationPoints-1);
    newTimeIntervals.setZero();

    for(int i = 0; i < (numberOfCollocationPoints-1); i++)
    {
        Eigen::VectorXd segmentStates(26);
        segmentStates = temporaryDesignVector.block(19*i,0,26,1);
        newTimeIntervals(i) = segmentStates(25) - segmentStates(6);

    }

    timeIntervals = newTimeIntervals;
}


void applyLineSearchAttenuation(const Eigen::VectorXd collocationCorrectionVector,  Eigen::MatrixXd& collocationDefectVector,  Eigen::MatrixXd& collocationDesignVector, Eigen::VectorXd timeIntervals, const Eigen::VectorXd thrustAndMassParameters, const int numberOfCollocationPoints, const int continuationIndex, const Eigen::MatrixXd phaseConstraintVector)
{

    Eigen::MatrixXd minimumNormDesignVector(collocationDesignVector.rows(),collocationDesignVector.cols());;
    Eigen::MatrixXd minimumNormDefectVector(collocationDefectVector.rows(),collocationDefectVector.cols());;


    Eigen::ArrayXd attenuationFactor = Eigen::ArrayXd::LinSpaced(10,0.1,1.0);
    for (int i = 0; i < 10; i++)
    {

        //std::cout << "i: " << i << std::endl;
        Eigen::MatrixXd collocationDefectVectorTemp(collocationDefectVector.rows(),collocationDefectVector.cols());
        Eigen::MatrixXd collocationDesignVectorTemp(collocationDesignVector.rows(),collocationDesignVector.cols());

        collocationDefectVectorTemp.setZero();
        collocationDesignVectorTemp.setZero();


        // correct the design vector
        Eigen::VectorXd temporaryDesignVector = collocationDesignVector + attenuationFactor(i) * collocationCorrectionVector;


        // compute the derivatives at each node and interior point
        Eigen::VectorXd temporaryDerivativeVector = computeDesignVectorDerivatives(temporaryDesignVector, thrustAndMassParameters, numberOfCollocationPoints  );

        //std::cout << "computeDesignVectorDerivatives completed: " << i << std::endl;


        // Store the design vector and derivative vector in correct format to be able to use the defectConstraints function
        Eigen::MatrixXd oddStates(6*(numberOfCollocationPoints-1),4);
        Eigen::MatrixXd oddStatesDerivatives(6*(numberOfCollocationPoints-1),4);

        rearrangeTemporaryVectors(temporaryDesignVector, temporaryDerivativeVector, oddStates, oddStatesDerivatives, numberOfCollocationPoints);

        //std::cout << "rearrangeTemporaryVectors completed: " << i << std::endl;


        // recompute the initial time and time intervals
        double initialTime;

        recomputeTimeProperties(temporaryDesignVector, initialTime, timeIntervals, numberOfCollocationPoints);

        //std::cout << "recomputeTimeProperties: " << timeIntervals << std::endl;
        //std::cout << "initialTime: " << initialTime << std::endl;



        computeCollocationDefects(collocationDefectVectorTemp, collocationDesignVectorTemp, oddStates, oddStatesDerivatives, timeIntervals, thrustAndMassParameters, numberOfCollocationPoints, initialTime, continuationIndex, phaseConstraintVector );

        //std::cout << "computeCollocationDefectsCompleted: " << std::endl;


        if (i == 0)
        {
            //std::cout << "REACHED alpha: " << attenuationFactor(i) << std::endl;
            minimumNormDesignVector = collocationDesignVectorTemp;
            minimumNormDefectVector = collocationDefectVectorTemp;

        } else if ( collocationDefectVectorTemp.norm() < minimumNormDefectVector.norm() )
        {
//            std::cout << "\nREACHED alpha: " << attenuationFactor(i) << std::endl;
//            std::cout << "startingNormBeforeCorrection: " <<  collocationDefectVector.norm() << std::endl;
//            std::cout << "minimumNormDefectVectorNorm: " << minimumNormDefectVector.norm()<< std::endl;
//            std::cout << "collocationDefectVectorTemp: " << collocationDefectVectorTemp.norm()<< std::endl;


            minimumNormDesignVector = collocationDesignVectorTemp;
            minimumNormDefectVector = collocationDefectVectorTemp;
        }

    }

// Store the solution with minimum norm as the new vectors
collocationDefectVector =  minimumNormDefectVector;
collocationDesignVector =  minimumNormDesignVector;

}

