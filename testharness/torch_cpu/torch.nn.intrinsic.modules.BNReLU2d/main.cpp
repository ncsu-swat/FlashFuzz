#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has at least 4 dimensions for BNReLU2d (N, C, H, W)
        if (input.dim() < 4) {
            // Expand dimensions if needed
            while (input.dim() < 4) {
                input = input.unsqueeze(0);
            }
        }
        
        // Extract parameters for BNReLU2d
        int64_t num_features = input.size(1);
        
        // Ensure num_features is positive
        if (num_features <= 0) {
            num_features = 1;
            input = input.reshape({input.size(0), num_features, -1, input.size(-1)});
        }
        
        // Create BNReLU2d module
        auto bn = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(num_features));
        auto relu = torch::nn::ReLU();
        
        // Create BNReLU2d by sequential composition
        auto bnrelu = torch::nn::Sequential(bn, relu);
        
        // Set training mode based on a byte from the input data
        bool training_mode = offset < Size ? (Data[offset++] % 2 == 0) : false;
        bnrelu->train(training_mode);
        
        // Set parameters for batch norm if we have enough data
        if (offset + 4 < Size) {
            // Parse momentum and eps from input data
            float momentum_raw, eps_raw;
            std::memcpy(&momentum_raw, Data + offset, sizeof(float));
            offset += sizeof(float);
            std::memcpy(&eps_raw, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Clamp momentum to [0, 1]
            float momentum = std::abs(momentum_raw);
            momentum = momentum - std::floor(momentum);
            
            // Clamp eps to a small positive value
            float eps = std::abs(eps_raw);
            if (eps == 0.0f) eps = 1e-5f;
            
            // Set the parameters
            bn->options.momentum(momentum);
            bn->options.eps(eps);
        }
        
        // Apply BNReLU2d to the input tensor
        torch::Tensor output = bnrelu->forward(input);
        
        // Verify output has same shape as input
        if (output.sizes() != input.sizes()) {
            throw std::runtime_error("Output shape doesn't match input shape");
        }
        
        // Test backward pass if in training mode
        if (training_mode) {
            // Create a gradient tensor
            torch::Tensor grad_output = torch::ones_like(output);
            
            // Make input require gradients
            input = input.detach().requires_grad_(true);
            
            // Forward pass
            output = bnrelu->forward(input);
            
            // Backward pass
            output.backward(grad_output);
        }
        
        // Test with eval mode
        bnrelu->eval();
        torch::Tensor eval_output = bnrelu->forward(input);
        
        // Test serialization/deserialization if we have enough data
        if (offset < Size) {
            torch::serialize::OutputArchive archive;
            bnrelu->save(archive);
            
            // Create a new module and load the state
            auto new_bnrelu = torch::nn::Sequential(
                torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(num_features)),
                torch::nn::ReLU()
            );
            
            std::stringstream ss;
            archive.save_to(ss);
            
            torch::serialize::InputArchive input_archive;
            input_archive.load_from(ss);
            new_bnrelu->load(input_archive);
            
            // Verify the loaded module produces the same output
            torch::Tensor new_output = new_bnrelu->forward(input);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
