#include <iostream>
#include "wsample/mysampling.h"

void test_sampling() {
    // Sample graph represented by colptr and row arrays
    torch::Tensor colptr = torch::tensor({0, 2, 5, 7, 9});
    torch::Tensor row = torch::tensor({1, 2, 0, 2, 3, 0, 2, 3, 0, 3});

    // Input nodes for sampling
    torch::Tensor input_node = torch::tensor({0, 1, 2});

    // Number of neighbors to sample for each input node
    vector<int64_t> num_neighbors = {2, 3};

    // Call the sampling function
    auto result = sample<false, true>(colptr, row, input_node, num_neighbors);

    // Unpack the result
    torch::Tensor samples = get<0>(result);
    torch::Tensor rows = get<1>(result);
    torch::Tensor cols = get<2>(result);
    torch::Tensor edges = get<3>(result);

    // Print the results for verification
    cout << "Sampled Nodes: " << samples << endl;
    cout << "Sampled Rows: " << rows << endl;
    cout << "Sampled Cols: " << cols << endl;
    cout << "Sampled Edges: " << edges << endl;

    // Perform additional assertions/tests based on the expected output
    // For example, you can compare the results with the expected output.
    // The expected output will depend on the random sampling involved, so it will vary in each run.
}


int main() {
    // Run the test case
    test_sampling();

    cout<<"sidy good luck"<<endl;
}

