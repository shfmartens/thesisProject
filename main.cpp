#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include "computeManifolds.h"


// Function declaration
Eigen::VectorXd create_initial_state_vector(string orbit_type, string selected_orbit);

int main (){

    // Load configuration parameters
    boost::property_tree::ptree jsontree;
    boost::property_tree::read_json("config.json", jsontree);
    Eigen::VectorXd initialStateVector = Eigen::VectorXd::Zero(6);

    for (auto orbit_type : jsontree) {
        cout << "\n" << endl << orbit_type.first  << endl;

        auto tree_initial_states = jsontree.get_child( orbit_type.first);

        #pragma omp parallel num_threads(3)

        // This code will be executed by three threads.
        // Chunks of this loop will be divided amongst
        // the (three) threads of the current team.
        {
            #pragma omp for
            for (auto selected_orbit : tree_initial_states) {
//                int tid = omp_get_thread_num();
//                cout << tid << endl;
                cout << "\n" << endl << selected_orbit.first << endl;

                initialStateVector = create_initial_state_vector(orbit_type.first, selected_orbit.first);
                computeManifolds(orbit_type.first, selected_orbit.first, initialStateVector);
            }
        }
    }
    return 0;
}


Eigen::VectorXd create_initial_state_vector(string orbit_type, string selected_orbit){
    boost::property_tree::ptree jsontree;
    boost::property_tree::read_json("config.json", jsontree);

    Eigen::VectorXd initial_state_vector = Eigen::VectorXd::Zero(6);
    initial_state_vector(0) = jsontree.get<double>( orbit_type + "." + selected_orbit + ".x");;
    initial_state_vector(1) = jsontree.get<double>( orbit_type + "." + selected_orbit + ".y");;
    initial_state_vector(2) = jsontree.get<double>( orbit_type + "." + selected_orbit + ".z");;
    initial_state_vector(3) = jsontree.get<double>( orbit_type + "." + selected_orbit + ".x_dot");;
    initial_state_vector(4) = jsontree.get<double>( orbit_type + "." + selected_orbit + ".y_dot");;
    initial_state_vector(5) = jsontree.get<double>( orbit_type + "." + selected_orbit + ".z_dot");;

    return initial_state_vector;
}

