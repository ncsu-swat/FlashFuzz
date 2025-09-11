#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <vector>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Need at least some data to create a module
        if (Size < 4) {
            return 0;
        }
        
        // Create a simple module to replicate
        torch::nn::Linear module = nullptr;
        
        // Parse the number of features from the input data
        int64_t in_features = 1;
        int64_t out_features = 1;
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&in_features, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure in_features is reasonable
            in_features = std::abs(in_features) % 100 + 1;
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&out_features, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure out_features is reasonable
            out_features = std::abs(out_features) % 100 + 1;
        }
        
        // Create the module
        module = torch::nn::Linear(in_features, out_features);
        
        // Parse the number of devices to replicate to
        uint8_t num_devices_byte = 0;
        if (offset < Size) {
            num_devices_byte = Data[offset++];
        }
        
        // Ensure we have at least 1 device and not too many
        int num_devices = (num_devices_byte % 8) + 1;
        
        // Create a list of devices (all CPU in this case)
        std::vector<torch::Device> devices;
        for (int i = 0; i < num_devices; i++) {
            devices.push_back(torch::Device(torch::kCPU));
        }
        
        // Create an input tensor for the module
        torch::Tensor input;
        if (offset < Size) {
            input = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // Create a default tensor if we've consumed all data
            input = torch::randn({1, in_features});
        }
        
        // Manually replicate the module across devices (since torch::nn::parallel::replicate doesn't exist)
        std::vector<torch::nn::Linear> replicated_modules;
        for (const auto& device : devices) {
            auto replicated_module = torch::nn::Linear(in_features, out_features);
            replicated_module->to(device);
            // Copy parameters from original module
            auto original_params = module->named_parameters();
            auto replicated_params = replicated_module->named_parameters();
            for (const auto& param_pair : original_params) {
                if (replicated_params.find(param_pair.key()) != replicated_params.end()) {
                    replicated_params[param_pair.key()].copy_(param_pair.value().to(device));
                }
            }
            replicated_modules.push_back(replicated_module);
        }
        
        // Test the replicated modules
        for (auto& replicated_module : replicated_modules) {
            // Reshape input if necessary to match module's expected input
            torch::Tensor reshaped_input;
            try {
                if (input.dim() == 0) {
                    reshaped_input = input.reshape({1, in_features});
                } else if (input.dim() == 1) {
                    if (input.size(0) == in_features) {
                        reshaped_input = input.reshape({1, in_features});
                    } else {
                        reshaped_input = torch::randn({1, in_features});
                    }
                } else {
                    // Try to use the input as is, or reshape if possible
                    if (input.size(-1) == in_features) {
                        reshaped_input = input;
                    } else {
                        reshaped_input = torch::randn({1, in_features});
                    }
                }
                
                // Forward pass through the replicated module
                auto output = replicated_module->forward(reshaped_input);
            } catch (const std::exception& e) {
                // Catch exceptions from the forward pass but continue testing
            }
        }
        
        // Test with different input shapes
        if (offset < Size) {
            try {
                auto another_input = fuzzer_utils::createTensor(Data, Size, offset);
                auto batch_size = another_input.size(0);
                auto reshaped = torch::randn({batch_size, in_features});
                
                for (auto& replicated_module : replicated_modules) {
                    auto output = replicated_module->forward(reshaped);
                }
            } catch (const std::exception& e) {
                // Ignore exceptions from this additional test
            }
        }
        
        // Test with empty input
        try {
            auto empty_input = torch::empty({0, in_features});
            for (auto& replicated_module : replicated_modules) {
                auto output = replicated_module->forward(empty_input);
            }
        } catch (const std::exception& e) {
            // Ignore exceptions from empty input test
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
