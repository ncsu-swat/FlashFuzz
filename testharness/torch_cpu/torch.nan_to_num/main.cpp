#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Create input tensor
        if (offset < Size) {
            torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Parse replacement values for nan, posinf, neginf
            double nan_replacement = 0.0;
            double posinf_replacement = 0.0;
            double neginf_replacement = 0.0;
            
            // Parse nan_replacement if we have enough data
            if (offset + sizeof(double) <= Size) {
                std::memcpy(&nan_replacement, Data + offset, sizeof(double));
                offset += sizeof(double);
            }
            
            // Parse posinf_replacement if we have enough data
            if (offset + sizeof(double) <= Size) {
                std::memcpy(&posinf_replacement, Data + offset, sizeof(double));
                offset += sizeof(double);
            }
            
            // Parse neginf_replacement if we have enough data
            if (offset + sizeof(double) <= Size) {
                std::memcpy(&neginf_replacement, Data + offset, sizeof(double));
                offset += sizeof(double);
            }
            
            // Apply nan_to_num with different combinations of parameters
            
            // Case 1: Default parameters
            torch::Tensor result1 = torch::nan_to_num(input_tensor);
            
            // Case 2: With nan_replacement only
            torch::Tensor result2 = torch::nan_to_num(input_tensor, nan_replacement);
            
            // Case 3: With nan_replacement and posinf_replacement
            torch::Tensor result3 = torch::nan_to_num(input_tensor, nan_replacement, posinf_replacement);
            
            // Case 4: With all parameters
            torch::Tensor result4 = torch::nan_to_num(input_tensor, nan_replacement, posinf_replacement, neginf_replacement);
            
            // In-place version
            torch::Tensor input_copy = input_tensor.clone();
            torch::nan_to_num_(input_copy, nan_replacement, posinf_replacement, neginf_replacement);
            
            // Test with different dtypes if the tensor is floating point
            if (input_tensor.is_floating_point()) {
                // Convert to different floating point types and test
                std::vector<torch::ScalarType> float_types = {
                    torch::kFloat, torch::kDouble, torch::kHalf, torch::kBFloat16
                };
                
                for (auto dtype : float_types) {
                    if (input_tensor.scalar_type() != dtype) {
                        try {
                            torch::Tensor converted = input_tensor.to(dtype);
                            torch::Tensor result = torch::nan_to_num(converted, nan_replacement, posinf_replacement, neginf_replacement);
                        } catch (const std::exception &) {
                            // Ignore exceptions from type conversion
                        }
                    }
                }
            }
            
            // Test with tensors containing special values
            if (input_tensor.is_floating_point() && input_tensor.numel() != 0) {
                try {
                    // Create a tensor with NaN, +Inf, -Inf values
                    auto options = torch::TensorOptions().dtype(input_tensor.dtype());
                    torch::Tensor special_tensor;
                    
                    if (input_tensor.dim() == 0) {
                        special_tensor = torch::tensor(std::numeric_limits<double>::quiet_NaN(), options);
                    } else {
                        special_tensor = torch::ones_like(input_tensor);
                        
                        // Set some values to NaN, +Inf, -Inf if tensor is not empty
                        if (special_tensor.numel() > 2) {
                            // Set first element to NaN
                            special_tensor.flatten()[0] = std::numeric_limits<double>::quiet_NaN();
                            
                            // Set second element to +Inf
                            special_tensor.flatten()[1] = std::numeric_limits<double>::infinity();
                            
                            // Set third element to -Inf
                            if (special_tensor.numel() > 2) {
                                special_tensor.flatten()[2] = -std::numeric_limits<double>::infinity();
                            }
                        }
                    }
                    
                    // Apply nan_to_num to the special tensor
                    torch::Tensor special_result = torch::nan_to_num(special_tensor, nan_replacement, posinf_replacement, neginf_replacement);
                } catch (const std::exception &) {
                    // Ignore exceptions from creating special tensors
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