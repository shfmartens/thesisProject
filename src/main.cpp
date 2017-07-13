#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
//#include <omp.h>
#include "thesisProject/src/computeManifolds.h"


// Function declaration
Eigen::VectorXd create_initial_state_vector(string orbit_type, string selected_orbit);

int main (){

    // Load configuration parameters
    boost::property_tree::ptree jsontree;
//    boost::property_tree::read_json("../config/ver_folta_eigenvectors.json", jsontree);
//    boost::property_tree::read_json("../config/config.json", jsontree);
    boost::property_tree::read_json("../config/config_lp.json", jsontree);
//    boost::property_tree::read_json("../src/verification/halo_verification_l1.json", jsontree);
    Eigen::VectorXd initialStateVector = Eigen::VectorXd::Zero(6);

    for (auto orbit_type : jsontree) {

        auto tree_initial_states = jsontree.get_child( orbit_type.first);
//        #pragma omp parallel num_threads(14)
        {
//            #pragma omp for
            for (unsigned int i=1; i<=tree_initial_states.size(); i++) {

                string selected_orbit = orbit_type.first + "_" + to_string(i);

                std::cout << std::endl;
                std::cout << "==================================================================" << std::endl;
                std::cout << "                          " << selected_orbit << "                        " << std::endl;
                std::cout << "==================================================================" << std::endl;

                initialStateVector = create_initial_state_vector(orbit_type.first, selected_orbit);
                computeManifolds(orbit_type.first, selected_orbit, initialStateVector);
//                computeManifolds(orbit_type.first, selected_orbit, initialStateVector, 0.96, 0.04);
            }
        }
    }
    return 0;
}


Eigen::VectorXd create_initial_state_vector(string orbit_type, string selected_orbit){
    boost::property_tree::ptree jsontree;
//    boost::property_tree::read_json("../config/ver_folta_eigenvectors.json", jsontree);
//    boost::property_tree::read_json("../config/config.json", jsontree);
    boost::property_tree::read_json("../config/config_lp.json", jsontree);
//    boost::property_tree::read_json("../src/verification/halo_verification_l1.json", jsontree);

    Eigen::VectorXd initial_state_vector = Eigen::VectorXd::Zero(6);
    initial_state_vector(0) = jsontree.get<double>( orbit_type + "." + selected_orbit + ".x");;
    initial_state_vector(1) = jsontree.get<double>( orbit_type + "." + selected_orbit + ".y");;
    initial_state_vector(2) = jsontree.get<double>( orbit_type + "." + selected_orbit + ".z");;
    initial_state_vector(3) = jsontree.get<double>( orbit_type + "." + selected_orbit + ".x_dot");;
    initial_state_vector(4) = jsontree.get<double>( orbit_type + "." + selected_orbit + ".y_dot");;
    initial_state_vector(5) = jsontree.get<double>( orbit_type + "." + selected_orbit + ".z_dot");;

    return initial_state_vector;
}

