#include "fuzzer_utils.h"
#include <iostream>
#include <sstream>
#include <vector>
#include <torch/script.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        // Need at least some bytes to form a potentially valid model
        if (Size < 8) {
            return 0;
        }

        // Create a stringstream from the input data to avoid file I/O
        std::string data_str(reinterpret_cast<const char*>(Data), Size);
        std::istringstream input_stream(data_str);

        try {
            // Try to load the model from the stream
            auto module = torch::jit::load(input_stream);

            // If the model loads successfully, try to run it with some inputs
            size_t offset = 0;
            
            // Create a tensor to use as input for the loaded model
            torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Try to get the method names from the module
            auto methods = module.get_methods();
            
            // If there are methods, try to run forward
            if (!methods.empty()) {
                try {
                    std::vector<torch::jit::IValue> inputs;
                    inputs.push_back(input);
                    auto output = module.forward(inputs);
                } catch (...) {
                    // Ignore exceptions from running the model
                    // Shape mismatches, type errors, etc. are expected
                }
            }

            // Also try calling named methods
            for (const auto& method : methods) {
                try {
                    std::vector<torch::jit::IValue> inputs;
                    inputs.push_back(input);
                    method.function()(inputs);
                } catch (...) {
                    // Ignore exceptions - expected for invalid inputs
                }
            }

        } catch (...) {
            // Ignore exceptions from loading the model
            // Most fuzz inputs won't be valid TorchScript models
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}