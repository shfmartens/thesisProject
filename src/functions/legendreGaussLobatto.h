
#ifndef TUDATBUNDLE_LEGENDREGAUSSLOBATTO_H
#define TUDATBUNDLE_LEGENDREGAUSSLOBATTO_H

#include "Tudat/Mathematics/BasicMathematics/basicFunction.h"
#include "Eigen/Core"

//! Extract all relevant Collocation Quantities.
struct LegendreGaussLobatto
{
    //! Default destructor.
    virtual ~LegendreGaussLobatto( ) { }

    //! Expected true location of the root.
    virtual Eigen::VectorXd getNormalizedNodeTimes( ) = 0;

};


#endif // TUDATBUNDLE_LEGENDREGAUSSLOBATTO_H
