#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <cstring>        // For std::memcpy

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Need at least some data to create tensors
        if (Size < 4) {
            return 0;
        }
        
        // Create first input tensor
        torch::Tensor input1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create second input tensor if we have more data
        torch::Tensor input2;
        if (offset < Size) {
            input2 = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If not enough data for second tensor, create a scalar tensor
            input2 = torch::tensor(1.0, input1.options());
        }
        
        // Apply torch.minimum operation
        torch::Tensor result = torch::minimum(input1, input2);
        
        // Try element-wise minimum with other variant
        if (offset + 1 < Size) {
            // Use the next byte to decide whether to test other variants
            uint8_t variant_selector = Data[offset++];
            
            if (variant_selector % 3 == 0) {
                // Test out.minimum variant - wrap in inner try-catch for shape mismatches
                try {
                    torch::Tensor out = torch::empty_like(result);
                    torch::minimum_out(out, input1, input2);
                } catch (...) {
                    // Silently ignore expected failures
                }
            } else if (variant_selector % 3 == 1) {
                // Test minimum with a random scalar from data
                try {
                    float scalar_value = 0.0f;
                    if (offset + sizeof(float) <= Size) {
                        std::memcpy(&scalar_value, Data + offset, sizeof(float));
                        offset += sizeof(float);
                    }
                    // Clamp to reasonable range to avoid NaN issues
                    if (std::isnan(scalar_value) || std::isinf(scalar_value)) {
                        scalar_value = 0.0f;
                    }
                    torch::Tensor scalar_tensor = torch::tensor(scalar_value, input1.options());
                    torch::Tensor result2 = torch::minimum(input1, scalar_tensor);
                } catch (...) {
                    // Silently ignore expected failures
                }
            } else {
                // Test minimum with scalar tensor
                try {
                    double scalar_value = 0.0;
                    if (offset + sizeof(double) <= Size) {
                        std::memcpy(&scalar_value, Data + offset, sizeof(double));
                        offset += sizeof(double);
                    }
                    // Clamp to reasonable range
                    if (std::isnan(scalar_value) || std::isinf(scalar_value)) {
                        scalar_value = 0.0;
                    }
                    torch::Tensor scalar_tensor = torch::tensor(scalar_value, input1.options());
                    torch::Tensor result_scalar = torch::minimum(input1, scalar_tensor);
                } catch (...) {
                    // Silently ignore expected failures
                }
            }
        }
        
        // Test with empty tensors if we have more data
        if (offset + 1 < Size) {
            uint8_t empty_test = Data[offset++];
            if (empty_test % 2 == 0) {
                try {
                    // Create empty tensors with same dtype as input1
                    std::vector<int64_t> empty_shape = {0};
                    torch::Tensor empty_tensor1 = torch::empty(empty_shape, input1.options());
                    torch::Tensor empty_tensor2 = torch::empty(empty_shape, input1.options());
                    
                    // Test minimum with two empty tensors (same shape)
                    torch::Tensor empty_result = torch::minimum(empty_tensor1, empty_tensor2);
                } catch (...) {
                    // Silently ignore - shape mismatches are expected
                }
            }
        }
        
        // Test with tensors of different dtypes if we have more data
        if (offset + 1 < Size) {
            uint8_t dtype_test = Data[offset++];
            if (dtype_test % 2 == 0) {
                try {
                    // Create a tensor with different dtype
                    torch::ScalarType other_dtype = fuzzer_utils::parseDataType(dtype_test);
                    torch::Tensor other_dtype_tensor = input1.to(other_dtype);
                    
                    // Test minimum with different dtype tensor
                    torch::Tensor mixed_result = torch::minimum(input1, other_dtype_tensor);
                } catch (...) {
                    // Silently ignore - dtype conversion issues are expected
                }
            }
        }
        
        // Test with broadcasting if we have more data
        if (offset + 1 < Size) {
            uint8_t broadcast_test = Data[offset++];
            if (broadcast_test % 2 == 0 && input1.dim() > 0) {
                try {
                    // Create a tensor with shape that can be broadcast with input1
                    std::vector<int64_t> broadcast_shape;
                    if (input1.dim() > 1) {
                        // Take just the first dimension
                        broadcast_shape.push_back(input1.size(0));
                        broadcast_shape.push_back(1);
                    } else {
                        broadcast_shape.push_back(1);
                    }
                    
                    torch::Tensor broadcast_tensor = torch::ones(broadcast_shape, input1.options());
                    
                    // Test minimum with broadcasting
                    torch::Tensor broadcast_result = torch::minimum(input1, broadcast_tensor);
                } catch (...) {
                    // Silently ignore - broadcasting failures are expected
                }
            }
        }
        
        // Test with negative strides (non-contiguous tensor)
        if (offset + 1 < Size && input1.dim() > 0 && input1.numel() > 1) {
            uint8_t stride_test = Data[offset++];
            if (stride_test % 2 == 0) {
                try {
                    torch::Tensor flipped = input1.flip({0});
                    torch::Tensor stride_result = torch::minimum(input1, flipped);
                } catch (...) {
                    // Silently ignore
                }
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}