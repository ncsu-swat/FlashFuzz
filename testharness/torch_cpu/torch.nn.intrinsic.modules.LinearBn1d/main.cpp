#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>
#include <sstream>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has at least 2 dimensions for Linear layer
        // If not, reshape it to have a batch dimension and a feature dimension
        if (input.dim() < 2) {
            if (input.dim() == 0) {
                // Scalar tensor, reshape to [1, 1]
                input = input.reshape({1, 1});
            } else if (input.dim() == 1) {
                // 1D tensor, add batch dimension
                input = input.unsqueeze(0);
            }
        }
        
        // Get input dimensions
        int64_t batch_size = input.size(0);
        int64_t in_features = input.size(1);
        
        // Ensure we have valid dimensions for Linear layer
        if (in_features <= 0) {
            in_features = 1;
            input = input.reshape({batch_size, in_features});
        }
        
        // Get out_features from the remaining data
        int64_t out_features = 1;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&out_features, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure out_features is positive
            out_features = std::abs(out_features) % 32 + 1;
        }
        
        // Create Linear + BatchNorm1d combination manually since intrinsic module doesn't exist
        auto linear = torch::nn::Linear(in_features, out_features);
        auto bn1d = torch::nn::BatchNorm1d(out_features);
        
        // Set modules to evaluation mode to test both training and eval paths
        if (offset < Size && Data[offset++] % 2 == 0) {
            linear->train();
            bn1d->train();
        } else {
            linear->eval();
            bn1d->eval();
        }
        
        // Convert input to float if needed for compatibility
        if (input.scalar_type() != torch::kFloat && 
            input.scalar_type() != torch::kDouble) {
            input = input.to(torch::kFloat);
        }
        
        // Apply the Linear + BatchNorm1d combination
        torch::Tensor linear_output = linear->forward(input);
        torch::Tensor output = bn1d->forward(linear_output);
        
        // Test with different configurations
        if (offset < Size) {
            // Try with different bias settings
            bool bias = Data[offset++] % 2 == 0;
            auto linear2 = torch::nn::Linear(torch::nn::LinearOptions(in_features, out_features).bias(bias));
            auto bn1d2 = torch::nn::BatchNorm1d(out_features);
            
            torch::Tensor linear_output2 = linear2->forward(input);
            output = bn1d2->forward(linear_output2);
            
            // Try with different momentum and eps values
            if (offset + 2 < Size) {
                double momentum = static_cast<double>(Data[offset++]) / 255.0;
                double eps = std::max(1e-5, static_cast<double>(Data[offset++]) / 1000.0);
                auto linear3 = torch::nn::Linear(torch::nn::LinearOptions(in_features, out_features).bias(bias));
                auto bn1d3 = torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(out_features).momentum(momentum).eps(eps));
                
                torch::Tensor linear_output3 = linear3->forward(input);
                output = bn1d3->forward(linear_output3);
            }
        }
        
        // Test serialization/deserialization
        torch::serialize::OutputArchive output_archive;
        linear->save(output_archive);
        bn1d->save(output_archive);
        
        std::ostringstream stream;
        output_archive.save_to(stream);
        std::istringstream input_stream(stream.str());
        
        torch::serialize::InputArchive input_archive;
        input_archive.load_from(input_stream);
        
        auto linear_loaded = torch::nn::Linear(in_features, out_features);
        auto bn1d_loaded = torch::nn::BatchNorm1d(out_features);
        linear_loaded->load(input_archive);
        bn1d_loaded->load(input_archive);
        
        torch::Tensor linear_output_loaded = linear_loaded->forward(input);
        torch::Tensor output_loaded = bn1d_loaded->forward(linear_output_loaded);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}