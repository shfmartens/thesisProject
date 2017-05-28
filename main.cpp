#include "computeManifolds.h"


int main (){

    boost::property_tree::ptree jsontree;
    boost::property_tree::read_json("config.json", jsontree);

    auto tree_orbit_types = jsontree.get_child("initial_states");

    for (auto orbit_type : tree_orbit_types) {
        cout << "\n" << endl << orbit_type.first  << endl;

        auto tree_initial_states = jsontree.get_child("initial_states." + orbit_type.first);

        for (auto initial_state : tree_initial_states) {
            cout << "\n" << endl << initial_state.first  << endl;
            computeManifolds( orbit_type.first, initial_state.first );
        }
    }
    return 0;
}