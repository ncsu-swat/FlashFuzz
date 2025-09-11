#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <sstream>        // For stringstream

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create meaningful input
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for BatchNorm from the remaining data
        double momentum = 0.1;
        double eps = 1e-5;
        bool affine = true;
        bool track_running_stats = true;
        
        // If we have more data, use it to set parameters
        if (offset + 4 <= Size) {
            // Extract momentum (0-1 range)
            momentum = static_cast<double>(Data[offset++]) / 255.0;
            
            // Extract epsilon (small positive value)
            eps = std::max(1e-10, static_cast<double>(Data[offset++]) / 1e4);
            
            // Extract boolean parameters
            affine = Data[offset++] % 2 == 0;
            track_running_stats = Data[offset++] % 2 == 0;
        }
        
        // Ensure input has correct shape for BatchNorm
        // BatchNorm expects input of shape [N, C, ...] where N is batch size, C is channels
        if (input.dim() < 2) {
            // Add dimensions if needed
            if (input.dim() == 0) {
                input = input.unsqueeze(0).unsqueeze(0);
            } else if (input.dim() == 1) {
                input = input.unsqueeze(0);
            }
        }
        
        // Create BatchNorm module (using regular BatchNorm since SyncBatchNorm is not available)
        auto batchnorm = torch::nn::BatchNorm1d(
            torch::nn::BatchNorm1dOptions(input.size(1))
                .momentum(momentum)
                .eps(eps)
                .affine(affine)
                .track_running_stats(track_running_stats)
        );
        
        // Apply BatchNorm
        torch::Tensor output = batchnorm->forward(input);
        
        // Test with eval mode
        batchnorm->eval();
        torch::Tensor eval_output = batchnorm->forward(input);
        
        // Test with train mode again
        batchnorm->train();
        torch::Tensor train_output = batchnorm->forward(input);
        
        // Test serialization/deserialization
        torch::serialize::OutputArchive output_archive;
        batchnorm->save(output_archive);
        
        std::stringstream ss;
        output_archive.save_to(ss);
        
        torch::serialize::InputArchive input_archive;
        input_archive.load_from(ss);
        
        auto loaded_batchnorm = torch::nn::BatchNorm1d(
            torch::nn::BatchNorm1dOptions(input.size(1))
        );
        loaded_batchnorm->load(input_archive);
        
        // Test the loaded module
        torch::Tensor loaded_output = loaded_batchnorm->forward(input);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
