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
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensors
        torch::Tensor input1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have enough data for the second tensor
        if (offset >= Size) {
            return 0;
        }
        
        torch::Tensor input2 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for the Bilinear module
        int64_t in1_features = 0;
        int64_t in2_features = 0;
        int64_t out_features = 0;
        bool bias = true;
        
        // Get dimensions from input tensors if possible
        if (input1.dim() >= 1) {
            in1_features = input1.size(-1);
        } else {
            // Default value if tensor doesn't have enough dimensions
            in1_features = 5;
        }
        
        if (input2.dim() >= 1) {
            in2_features = input2.size(-1);
        } else {
            // Default value if tensor doesn't have enough dimensions
            in2_features = 5;
        }
        
        // Get out_features from remaining data if available
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&out_features, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure out_features is reasonable
            out_features = std::abs(out_features) % 10 + 1;
        } else {
            // Default value
            out_features = 3;
        }
        
        // Get bias flag if data available
        if (offset < Size) {
            bias = Data[offset++] & 0x1;
        }
        
        // Create the Bilinear module using BilinearOptions
        torch::nn::BilinearOptions options(in1_features, in2_features, out_features);
        options.bias(bias);
        torch::nn::Bilinear bilinear(options);
        
        // Reshape inputs if needed to match expected dimensions for bilinear
        if (input1.dim() == 0) {
            input1 = input1.reshape({1, in1_features});
        } else if (input1.dim() == 1) {
            input1 = input1.reshape({1, input1.size(0)});
            if (input1.size(1) != in1_features) {
                input1 = input1.narrow(1, 0, std::min(input1.size(1), in1_features));
                if (input1.size(1) < in1_features) {
                    input1 = torch::pad(input1, {0, in1_features - input1.size(1)});
                }
            }
        } else {
            // For higher dimensions, ensure the last dimension matches in1_features
            if (input1.size(-1) != in1_features) {
                auto sizes = input1.sizes().vec();
                sizes.back() = in1_features;
                input1 = input1.reshape(sizes);
            }
        }
        
        if (input2.dim() == 0) {
            input2 = input2.reshape({1, in2_features});
        } else if (input2.dim() == 1) {
            input2 = input2.reshape({1, input2.size(0)});
            if (input2.size(1) != in2_features) {
                input2 = input2.narrow(1, 0, std::min(input2.size(1), in2_features));
                if (input2.size(1) < in2_features) {
                    input2 = torch::pad(input2, {0, in2_features - input2.size(1)});
                }
            }
        } else {
            // For higher dimensions, ensure the last dimension matches in2_features
            if (input2.size(-1) != in2_features) {
                auto sizes = input2.sizes().vec();
                sizes.back() = in2_features;
                input2 = input2.reshape(sizes);
            }
        }
        
        // Ensure both inputs have the same batch dimensions
        if (input1.dim() > 1 && input2.dim() > 1) {
            auto batch_dims1 = input1.sizes().slice(0, input1.dim() - 1);
            auto batch_dims2 = input2.sizes().slice(0, input2.dim() - 1);
            
            if (batch_dims1 != batch_dims2) {
                // Reshape to match batch dimensions
                std::vector<int64_t> new_shape1 = {1, in1_features};
                std::vector<int64_t> new_shape2 = {1, in2_features};
                
                input1 = input1.reshape(new_shape1);
                input2 = input2.reshape(new_shape2);
            }
        }
        
        // Convert tensors to the same dtype if they differ
        if (input1.dtype() != input2.dtype()) {
            if (input1.is_floating_point() && input2.is_floating_point()) {
                // Convert to higher precision
                if (input1.scalar_type() == torch::kDouble || input2.scalar_type() == torch::kDouble) {
                    input1 = input1.to(torch::kDouble);
                    input2 = input2.to(torch::kDouble);
                } else if (input1.scalar_type() == torch::kFloat || input2.scalar_type() == torch::kFloat) {
                    input1 = input1.to(torch::kFloat);
                    input2 = input2.to(torch::kFloat);
                } else {
                    input1 = input1.to(torch::kFloat);
                    input2 = input2.to(torch::kFloat);
                }
            } else {
                // If one is not floating point, convert both to float
                input1 = input1.to(torch::kFloat);
                input2 = input2.to(torch::kFloat);
            }
        }
        
        // Apply the bilinear module
        torch::Tensor output = bilinear->forward(input1, input2);
        
        // Perform some operations on the output to ensure it's used
        auto sum = output.sum();
        
        return 0; // keep the input
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
