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
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensors
        torch::Tensor input1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Check if we have enough data left for the second tensor
        if (offset >= Size) {
            return 0;
        }
        
        torch::Tensor input2 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for the Bilinear module
        int64_t in1_features = 0;
        int64_t in2_features = 0;
        int64_t out_features = 0;
        bool bias = true;
        
        // Get in1_features from input1
        if (input1.dim() > 0) {
            in1_features = input1.size(-1);
        } else {
            // Default value if input1 is a scalar
            in1_features = 1;
        }
        
        // Get in2_features from input2
        if (input2.dim() > 0) {
            in2_features = input2.size(-1);
        } else {
            // Default value if input2 is a scalar
            in2_features = 1;
        }
        
        // Get out_features from remaining data
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&out_features, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure out_features is reasonable
            out_features = std::abs(out_features) % 32 + 1;
        } else {
            // Default value
            out_features = 1;
        }
        
        // Get bias flag
        if (offset < Size) {
            bias = Data[offset++] & 0x1;
        }
        
        // Create the Bilinear module
        torch::nn::Bilinear bilinear(
            torch::nn::BilinearOptions(in1_features, in2_features, out_features).bias(bias)
        );
        
        // Reshape inputs if needed to match expected dimensions for Bilinear
        // Bilinear expects inputs of shape (..., in1_features) and (..., in2_features)
        if (input1.dim() == 0) {
            input1 = input1.reshape({1, 1});
        } else if (input1.size(-1) != in1_features) {
            // If last dimension doesn't match in1_features, reshape
            std::vector<int64_t> new_shape = input1.sizes().vec();
            if (!new_shape.empty()) {
                new_shape.back() = in1_features;
                input1 = input1.reshape(new_shape);
            }
        }
        
        if (input2.dim() == 0) {
            input2 = input2.reshape({1, 1});
        } else if (input2.size(-1) != in2_features) {
            // If last dimension doesn't match in2_features, reshape
            std::vector<int64_t> new_shape = input2.sizes().vec();
            if (!new_shape.empty()) {
                new_shape.back() = in2_features;
                input2 = input2.reshape(new_shape);
            }
        }
        
        // Ensure inputs have the same batch dimensions
        if (input1.dim() > 1 && input2.dim() > 1) {
            std::vector<int64_t> shape1 = input1.sizes().vec();
            std::vector<int64_t> shape2 = input2.sizes().vec();
            
            // Remove the last dimension (features dimension)
            shape1.pop_back();
            shape2.pop_back();
            
            // If batch dimensions don't match, reshape one of the inputs
            if (shape1 != shape2) {
                // Use the simpler approach: reshape to 2D tensors
                int64_t batch_size1 = 1;
                for (auto dim : shape1) {
                    batch_size1 *= dim;
                }
                
                int64_t batch_size2 = 1;
                for (auto dim : shape2) {
                    batch_size2 *= dim;
                }
                
                // Use the smaller batch size for both
                int64_t common_batch_size = std::min(batch_size1, batch_size2);
                if (common_batch_size <= 0) common_batch_size = 1;
                
                input1 = input1.reshape({common_batch_size, in1_features});
                input2 = input2.reshape({common_batch_size, in2_features});
            }
        }
        
        // Apply the Bilinear module
        torch::Tensor output = bilinear->forward(input1, input2);
        
        // Perform some operations on the output to ensure it's used
        auto sum = output.sum();
        
        // Convert to CPU scalar to force computation
        float result = sum.item<float>();
        
        // Use the result in a way that prevents the compiler from optimizing it away
        if (std::isnan(result) || std::isinf(result)) {
            return 0;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
