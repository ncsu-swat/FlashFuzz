#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <vector>
#include <thread>

// Simple module for testing
struct TestModule : torch::nn::Module {
    TestModule() {}
    
    torch::Tensor forward(torch::Tensor x) {
        return x.sigmoid();
    }
};

// Simple parallel apply implementation for testing
template<typename ModuleType>
std::vector<torch::Tensor> parallel_apply(
    const std::vector<std::shared_ptr<ModuleType>>& modules,
    const std::vector<torch::Tensor>& inputs,
    const std::vector<torch::Device>& devices,
    int num_threads = 0) {
    
    if (modules.size() != inputs.size() || modules.size() != devices.size()) {
        throw std::runtime_error("Size mismatch between modules, inputs, and devices");
    }
    
    std::vector<torch::Tensor> outputs(modules.size());
    
    for (size_t i = 0; i < modules.size(); ++i) {
        if (!modules[i]) {
            throw std::runtime_error("Null module encountered");
        }
        
        torch::Tensor input = inputs[i].to(devices[i]);
        outputs[i] = modules[i]->forward(input);
    }
    
    return outputs;
}

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Need at least some data to work with
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensors
        std::vector<torch::Tensor> inputs;
        uint8_t num_inputs = Data[offset++] % 8 + 1; // 1-8 inputs
        
        for (uint8_t i = 0; i < num_inputs && offset < Size; ++i) {
            try {
                torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
                inputs.push_back(tensor);
            } catch (const std::exception&) {
                // If tensor creation fails, just continue with what we have
                break;
            }
        }
        
        // If we couldn't create any inputs, return
        if (inputs.empty()) {
            return 0;
        }
        
        // Create modules
        std::vector<std::shared_ptr<TestModule>> modules;
        uint8_t num_modules = (offset < Size) ? (Data[offset++] % 8 + 1) : 2; // 1-8 modules
        
        for (uint8_t i = 0; i < num_modules; ++i) {
            modules.push_back(std::make_shared<TestModule>());
        }
        
        // Create device list (all CPU for simplicity)
        std::vector<torch::Device> devices;
        for (uint8_t i = 0; i < num_modules; ++i) {
            devices.push_back(torch::kCPU);
        }
        
        // Prepare inputs for each module
        std::vector<torch::Tensor> inputs_per_module;
        for (uint8_t i = 0; i < num_modules; ++i) {
            // Use modulo to cycle through available inputs
            inputs_per_module.push_back(inputs[i % inputs.size()]);
        }
        
        // Test different numbers of threads
        uint8_t num_threads = 0;
        if (offset < Size) {
            num_threads = Data[offset++] % 8; // 0-7 threads
        }
        
        // Apply parallel_apply
        std::vector<torch::Tensor> outputs;
        
        if (num_threads > 0) {
            outputs = parallel_apply(
                modules, inputs_per_module, devices, num_threads);
        } else {
            outputs = parallel_apply(
                modules, inputs_per_module, devices);
        }
        
        // Verify outputs
        if (outputs.size() != modules.size()) {
            throw std::runtime_error("Output size mismatch");
        }
        
        // Test edge cases
        if (offset + 1 < Size) {
            uint8_t edge_case = Data[offset++];
            
            // Edge case: empty module list
            if (edge_case % 5 == 0) {
                std::vector<std::shared_ptr<TestModule>> empty_modules;
                std::vector<torch::Tensor> empty_inputs;
                std::vector<torch::Device> empty_devices;
                
                try {
                    auto empty_outputs = parallel_apply(
                        empty_modules, empty_inputs, empty_devices);
                } catch (const std::exception&) {
                    // Expected to throw
                }
            }
            
            // Edge case: mismatched sizes
            if (edge_case % 5 == 1 && !inputs.empty()) {
                std::vector<std::shared_ptr<TestModule>> one_module = {std::make_shared<TestModule>()};
                std::vector<torch::Tensor> two_inputs = {inputs[0]};
                if (inputs.size() > 1) two_inputs.push_back(inputs[1]);
                std::vector<torch::Device> one_device = {torch::kCPU};
                
                try {
                    auto mismatched_outputs = parallel_apply(
                        one_module, two_inputs, one_device);
                } catch (const std::exception&) {
                    // May throw due to size mismatch
                }
            }
            
            // Edge case: excessive threads
            if (edge_case % 5 == 2 && !modules.empty() && !inputs_per_module.empty()) {
                try {
                    auto excessive_threads = parallel_apply(
                        modules, inputs_per_module, devices, 1000);
                } catch (const std::exception&) {
                    // May throw or handle gracefully
                }
            }
            
            // Edge case: nullptr module
            if (edge_case % 5 == 3) {
                std::vector<std::shared_ptr<TestModule>> null_modules = {nullptr};
                std::vector<torch::Tensor> one_input;
                if (!inputs.empty()) one_input.push_back(inputs[0]);
                else one_input.push_back(torch::ones({1}));
                std::vector<torch::Device> one_device = {torch::kCPU};
                
                try {
                    auto null_outputs = parallel_apply(
                        null_modules, one_input, one_device);
                } catch (const std::exception&) {
                    // Expected to throw
                }
            }
            
            // Edge case: invalid device
            if (edge_case % 5 == 4 && !modules.empty() && !inputs_per_module.empty()) {
                std::vector<torch::Device> invalid_devices = {torch::Device(torch::kCUDA, static_cast<torch::DeviceIndex>(99))};
                while (invalid_devices.size() < modules.size()) {
                    invalid_devices.push_back(torch::Device(torch::kCUDA, static_cast<torch::DeviceIndex>(99)));
                }
                
                try {
                    auto invalid_device_outputs = parallel_apply(
                        modules, inputs_per_module, invalid_devices);
                } catch (const std::exception&) {
                    // May throw due to invalid device
                }
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
