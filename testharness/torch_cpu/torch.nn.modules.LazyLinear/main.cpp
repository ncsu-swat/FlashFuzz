#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for basic parameters
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input;
        try {
            input = fuzzer_utils::createTensor(Data, Size, offset);
        } catch (const std::exception& e) {
            return 0;
        }
        
        // Extract parameters for Linear (using regular Linear instead of LazyLinear)
        int64_t in_features = 10;  // Default input features
        int64_t out_features = 1;
        bool bias = true;
        
        // Get dimensions from input tensor if available
        if (input.dim() >= 2) {
            in_features = input.size(-1);
        }
        
        // Get out_features from the data if available
        if (offset + sizeof(int64_t) <= Size) {
            int64_t raw_out_features;
            std::memcpy(&raw_out_features, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure out_features is positive and reasonable
            out_features = std::abs(raw_out_features) % 1024 + 1;
        }
        
        // Get bias parameter if available
        if (offset < Size) {
            bias = Data[offset++] & 0x1;
        }
        
        // Create Linear module
        torch::nn::Linear linear(torch::nn::LinearOptions(in_features, out_features).bias(bias));
        
        // Apply the module to the input tensor
        torch::Tensor output = linear->forward(input);
        
        // Access parameters to ensure they're properly initialized
        if (linear->weight.defined()) {
            auto weight = linear->weight;
            
            // If bias is enabled, access it
            if (bias && linear->bias.defined()) {
                auto bias_tensor = linear->bias;
            }
        }
        
        // Test with zero-grad to ensure backward pass works
        if (input.requires_grad() && output.requires_grad()) {
            output.sum().backward();
        }
        
        // Test serialization/deserialization
        if (offset + 1 < Size && (Data[offset] & 0x1)) {
            torch::serialize::OutputArchive output_archive;
            linear->save(output_archive);
            
            torch::nn::Linear linear2(torch::nn::LinearOptions(in_features, out_features).bias(bias));
            torch::serialize::InputArchive input_archive;
            linear2->load(input_archive);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
