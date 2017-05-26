// Include-statements.
#include <Eigen/Core>
#include <sstream>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include "readFile.h"

// Function start.

Eigen::MatrixXd readFile(string fileName) {

    std::ifstream fin(fileName);
    const int nCol = 42; // read from file
    std::vector< std::vector <double> > dataFromFile;  // your entire data-set of values

    std::vector<double> line(nCol, -1.0);  // create one line of nCol size and fill with -1
    bool done = false;
    while (!done)
    {
        for (int i = 0; !done && i < nCol; i++)
        {
            done = !(fin >> line[i]);
        }
        dataFromFile.push_back(line);
    }
    Eigen::MatrixXd matrixFromFile(dataFromFile.size(),42);
    for (int iRow = 0; iRow < dataFromFile.size(); iRow++)
    {
        for (int iCol = 0; iCol < 42; iCol++)
        {
            matrixFromFile(iRow,iCol) = dataFromFile[iRow][iCol];
        }
    }

    return matrixFromFile;
}
