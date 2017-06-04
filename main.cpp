#include "computeManifolds.h"

// Function declaration
Eigen::VectorXd create_initial_state_vector(string orbit_type, string selected_orbit);

int main (){

    // Load configuration parameters
    boost::property_tree::ptree jsontree;
    boost::property_tree::read_json("config.json", jsontree);
    Eigen::VectorXd initialStateVector = Eigen::VectorXd::Zero(6);

    auto tree_orbit_types = jsontree.get_child("initial_states");

    for (auto orbit_type : tree_orbit_types) {
        cout << "\n" << endl << orbit_type.first  << endl;

        auto tree_initial_states = jsontree.get_child("initial_states." + orbit_type.first);

        for (auto selected_orbit : tree_initial_states) {
            cout << "\n" << endl << selected_orbit.first  << endl;

            initialStateVector = create_initial_state_vector( orbit_type.first, selected_orbit.first );
            computeManifolds( orbit_type.first, selected_orbit.first, initialStateVector);
        }
    }
    return 0;
}


Eigen::VectorXd create_initial_state_vector(string orbit_type, string selected_orbit){
    boost::property_tree::ptree jsontree;
    boost::property_tree::read_json("config.json", jsontree);

    Eigen::VectorXd initial_state_vector = Eigen::VectorXd::Zero(6);
    initial_state_vector(0) = jsontree.get<double>("initial_states." + orbit_type + "." + selected_orbit + ".x");;
    initial_state_vector(1) = jsontree.get<double>("initial_states." + orbit_type + "." + selected_orbit + ".y");;
    initial_state_vector(2) = jsontree.get<double>("initial_states." + orbit_type + "." + selected_orbit + ".z");;
    initial_state_vector(3) = jsontree.get<double>("initial_states." + orbit_type + "." + selected_orbit + ".x_dot");;
    initial_state_vector(4) = jsontree.get<double>("initial_states." + orbit_type + "." + selected_orbit + ".y_dot");;
    initial_state_vector(5) = jsontree.get<double>("initial_states." + orbit_type + "." + selected_orbit + ".z_dot");;

    return initial_state_vector;
}

