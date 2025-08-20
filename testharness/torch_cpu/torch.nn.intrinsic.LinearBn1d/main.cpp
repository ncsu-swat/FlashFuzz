#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <sstream>        // For stringstream

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for Linear + BatchNorm1d combination
        int64_t in_features = 0;
        int64_t out_features = 0;
        
        // Determine in_features from input tensor
        if (input.dim() == 2) {
            in_features = input.size(1);
        } else if (input.dim() == 3) {
            in_features = input.size(2);
        } else {
            // For other dimensions, use a value from the data
            if (offset + sizeof(int64_t) <= Size) {
                std::memcpy(&in_features, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                in_features = std::abs(in_features) % 100 + 1;
            } else {
                in_features = 10;
            }
        }
        
        // Determine out_features from data
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&out_features, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            out_features = std::abs(out_features) % 100 + 1;
        } else {
            out_features = 20;
        }
        
        // Get bias flag
        bool bias = true;
        if (offset < Size) {
            bias = Data[offset++] & 0x1;
        }
        
        // Create Linear and BatchNorm1d modules separately since intrinsic module doesn't exist
        torch::nn::Linear linear(torch::nn::LinearOptions(in_features, out_features).bias(bias));
        torch::nn::BatchNorm1d bn(torch::nn::BatchNorm1dOptions(out_features));
        
        // Set module to evaluation mode to test both training and eval paths
        if (offset < Size) {
            bool train_mode = Data[offset++] & 0x1;
            if (train_mode) {
                linear->train();
                bn->train();
            } else {
                linear->eval();
                bn->eval();
            }
        }
        
        // Set some parameters to test different values
        if (offset + 4 < Size) {
            float eps = static_cast<float>(Data[offset++]) / 255.0f;
            float momentum = static_cast<float>(Data[offset++]) / 255.0f;
            bool affine = Data[offset++] & 0x1;
            bool track_running_stats = Data[offset++] & 0x1;
            
            bn->options.eps(std::max(eps, 1e-5f));
            bn->options.momentum(std::clamp(momentum, 0.0f, 1.0f));
            bn->options.affine(affine);
            bn->options.track_running_stats(track_running_stats);
        }
        
        // Reshape input if needed to match expected dimensions for Linear + BatchNorm1d
        if (input.dim() != 2 && input.dim() != 3) {
            // Linear expects 2D input, BatchNorm1d expects 2D or 3D
            std::vector<int64_t> new_shape;
            if (input.dim() < 2) {
                // Add dimensions to make it 2D
                new_shape.push_back(1);
                new_shape.push_back(in_features);
            } else {
                // Take first dimension, add in_features as last dimension
                new_shape.push_back(input.size(0));
                if (input.dim() > 2) {
                    new_shape.push_back(input.size(1));
                }
                new_shape.push_back(in_features);
            }
            
            // Resize the tensor, handling potential errors
            try {
                input = input.reshape(new_shape);
            } catch (const std::exception& e) {
                // If reshape fails, create a new tensor with the right shape
                input = torch::ones(new_shape, input.options());
            }
        }
        
        // Apply Linear followed by BatchNorm1d
        torch::Tensor linear_output = linear->forward(input);
        torch::Tensor output = bn->forward(linear_output);
        
        // Perform some operations on the output to ensure it's used
        torch::Tensor sum = output.sum();
        float sum_val = sum.item<float>();
        
        // Test serialization/deserialization
        torch::serialize::OutputArchive output_archive;
        linear->save(output_archive);
        bn->save(output_archive);
        
        std::stringstream ss;
        output_archive.save_to(ss);
        
        torch::serialize::InputArchive input_archive;
        input_archive.load_from(ss);
        
        torch::nn::Linear loaded_linear(torch::nn::LinearOptions(in_features, out_features).bias(bias));
        torch::nn::BatchNorm1d loaded_bn(torch::nn::BatchNorm1dOptions(out_features));
        loaded_linear->load(input_archive);
        loaded_bn->load(input_archive);
        
        // Test the loaded modules
        torch::Tensor loaded_linear_output = loaded_linear->forward(input);
        torch::Tensor output2 = loaded_bn->forward(loaded_linear_output);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}