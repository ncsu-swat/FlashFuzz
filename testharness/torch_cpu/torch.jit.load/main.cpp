#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <fstream>        // For file operations
#include <vector>         // For vector operations
#include <torch/script.h> // For torch::jit::load

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        if (Size < 4) {
            return 0;
        }

        size_t offset = 0;

        // Create a temporary file to store the model
        std::string tempFilename = "temp_model.pt";
        std::ofstream tempFile(tempFilename, std::ios::binary);
        if (!tempFile) {
            return 0;
        }

        // Write the data to the temporary file
        tempFile.write(reinterpret_cast<const char*>(Data), Size);
        tempFile.close();

        try {
            // Try to load the model
            auto module = torch::jit::load(tempFilename);

            // If the model loads successfully, try to run it with some inputs
            if (offset < Size) {
                // Create a tensor to use as input for the loaded model
                torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Try to get the method names from the module
                auto methodNames = module.get_methods();
                
                // If there are methods, try to run the first one
                if (!methodNames.empty()) {
                    try {
                        std::vector<torch::jit::IValue> inputs;
                        inputs.push_back(input);
                        auto output = module.forward(inputs);
                    } catch (...) {
                        // Ignore exceptions from running the model
                    }
                }
            }
        } catch (...) {
            // Ignore exceptions from loading the model
        }

        // Clean up the temporary file
        std::remove(tempFilename.c_str());
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}