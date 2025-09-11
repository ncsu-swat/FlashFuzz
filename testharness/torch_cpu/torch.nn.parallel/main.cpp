#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensors
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a simple model
        torch::nn::Linear model(input.size(-1), 10);
        
        // Test basic parallel operations that are available
        if (torch::cuda::is_available() && torch::cuda::device_count() > 1) {
            // Move model to GPU for parallel testing
            model->to(torch::kCUDA);
            input = input.to(torch::kCUDA);
            
            // Test basic forward pass on GPU
            auto output = model->forward(input);
        }
        
        // Test manual data parallelism by replicating computations
        std::vector<torch::nn::Linear> modules;
        for (int i = 0; i < 3; i++) {
            modules.push_back(torch::nn::Linear(input.size(-1), 10));
        }
        
        // Create inputs for manual parallel processing
        std::vector<torch::Tensor> inputs;
        for (int i = 0; i < 3; i++) {
            inputs.push_back(input);
        }
        
        // Test manual parallel apply
        try {
            std::vector<torch::Tensor> outputs;
            for (size_t i = 0; i < modules.size() && i < inputs.size(); i++) {
                outputs.push_back(modules[i]->forward(inputs[i]));
            }
        } catch (...) {
            // Manual parallel operations might throw for various reasons
        }
        
        // Test tensor operations that support parallelism
        try {
            // Test tensor splitting and concatenation
            auto split_tensors = input.chunk(2, 0);
            std::vector<torch::Tensor> processed_tensors;
            for (auto& tensor : split_tensors) {
                processed_tensors.push_back(model->forward(tensor));
            }
            auto concatenated = torch::cat(processed_tensors, 0);
        } catch (...) {
            // These operations might throw for various reasons
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
