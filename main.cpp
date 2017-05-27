#include "computeManifolds.h"


int main (){

    boost::property_tree::ptree jsontree;
    boost::property_tree::read_json("config.json", jsontree);

    auto tree_initial_states = jsontree.get_child("initial_states.halo");
    for (auto initial_state : tree_initial_states) {
        cout << "\n" << endl << initial_state.first  << endl;
        computeManifolds( initial_state.first );
    }

    return 0;
}