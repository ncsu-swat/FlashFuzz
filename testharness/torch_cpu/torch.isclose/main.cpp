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
        
        // Need at least some data to create tensors
        if (Size < 4) {
            return 0;
        }
        
        // Create two tensors to compare with isclose
        torch::Tensor tensor1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Check if we have enough data left for the second tensor
        if (offset >= Size) {
            // If not enough data, create a tensor with similar properties
            tensor1 = tensor1.clone();
        } else {
            // Create second tensor from remaining data
            torch::Tensor tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Parse parameters for isclose from remaining data
            double rtol = 1e-5;  // default value
            double atol = 1e-8;  // default value
            bool equal_nan = false;  // default value
            
            // Extract rtol if we have data
            if (offset + sizeof(double) <= Size) {
                std::memcpy(&rtol, Data + offset, sizeof(double));
                offset += sizeof(double);
                // Ensure rtol is non-negative
                rtol = std::abs(rtol);
            }
            
            // Extract atol if we have data
            if (offset + sizeof(double) <= Size) {
                std::memcpy(&atol, Data + offset, sizeof(double));
                offset += sizeof(double);
                // Ensure atol is non-negative
                atol = std::abs(atol);
            }
            
            // Extract equal_nan if we have data
            if (offset < Size) {
                equal_nan = Data[offset++] & 0x1;  // Use lowest bit to determine boolean
            }
            
            // Call isclose with various parameter combinations
            torch::Tensor result1 = torch::isclose(tensor1, tensor2);
            torch::Tensor result2 = torch::isclose(tensor1, tensor2, rtol, atol);
            torch::Tensor result3 = torch::isclose(tensor1, tensor2, rtol, atol, equal_nan);
            
            // Try with broadcasting
            if (tensor1.dim() > 0 && tensor2.dim() > 0) {
                // Create a tensor with just the first element of tensor2
                torch::Tensor scalar_tensor;
                try {
                    scalar_tensor = tensor2.index({0});
                    torch::Tensor broadcast_result = torch::isclose(tensor1, scalar_tensor);
                } catch (const std::exception &) {
                    // Indexing might fail for empty tensors, that's fine
                }
            }
            
            // Try with different dtypes
            try {
                torch::Tensor float_tensor1 = tensor1.to(torch::kFloat);
                torch::Tensor float_tensor2 = tensor2.to(torch::kFloat);
                torch::Tensor result_float = torch::isclose(float_tensor1, float_tensor2);
            } catch (const std::exception &) {
                // Conversion might fail, that's fine
            }
            
            // Try with NaN values if we have floating point tensors
            if (tensor1.is_floating_point() && tensor2.is_floating_point()) {
                try {
                    torch::Tensor nan_tensor1 = tensor1.clone();
                    torch::Tensor nan_tensor2 = tensor2.clone();
                    
                    // Set some values to NaN
                    if (nan_tensor1.numel() > 0) {
                        nan_tensor1.index_put_({0}, std::numeric_limits<float>::quiet_NaN());
                    }
                    if (nan_tensor2.numel() > 0) {
                        nan_tensor2.index_put_({0}, std::numeric_limits<float>::quiet_NaN());
                    }
                    
                    // Test with equal_nan=true and equal_nan=false
                    torch::Tensor nan_result1 = torch::isclose(nan_tensor1, nan_tensor2, rtol, atol, false);
                    torch::Tensor nan_result2 = torch::isclose(nan_tensor1, nan_tensor2, rtol, atol, true);
                } catch (const std::exception &) {
                    // Operations with NaN might fail, that's fine
                }
            }
            
            // Try with infinity values if we have floating point tensors
            if (tensor1.is_floating_point() && tensor2.is_floating_point()) {
                try {
                    torch::Tensor inf_tensor1 = tensor1.clone();
                    torch::Tensor inf_tensor2 = tensor2.clone();
                    
                    // Set some values to infinity
                    if (inf_tensor1.numel() > 0) {
                        inf_tensor1.index_put_({0}, std::numeric_limits<float>::infinity());
                    }
                    if (inf_tensor2.numel() > 0) {
                        inf_tensor2.index_put_({0}, std::numeric_limits<float>::infinity());
                    }
                    
                    torch::Tensor inf_result = torch::isclose(inf_tensor1, inf_tensor2);
                } catch (const std::exception &) {
                    // Operations with infinity might fail, that's fine
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
