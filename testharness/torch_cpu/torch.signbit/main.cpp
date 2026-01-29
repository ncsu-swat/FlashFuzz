#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <cstring>        // For memcpy
#include <limits>         // For numeric_limits

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
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply torch.signbit operation
        torch::Tensor result = torch::signbit(input_tensor);
        
        // Verify result is boolean type
        (void)result.dtype();
        
        // Try with different dtypes if we have enough data
        if (offset + 1 < Size) {
            uint8_t dtype_selector = Data[offset++];
            torch::ScalarType dtype = fuzzer_utils::parseDataType(dtype_selector);
            
            // Try to convert input tensor to a different dtype
            try {
                torch::Tensor converted = input_tensor.to(dtype);
                torch::Tensor result_converted = torch::signbit(converted);
                (void)result_converted;
            } catch (const std::exception&) {
                // Some dtype conversions might not be valid for signbit
            }
        }
        
        // Try with a scalar input if we have enough data
        if (offset + sizeof(double) <= Size) {
            double scalar_value;
            std::memcpy(&scalar_value, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            try {
                // Create a scalar tensor
                torch::Tensor scalar_tensor = torch::tensor(scalar_value);
                torch::Tensor scalar_result = torch::signbit(scalar_tensor);
                (void)scalar_result;
            } catch (const std::exception&) {
                // Some scalar operations might fail
            }
        }
        
        // Try with float tensor explicitly
        if (offset + sizeof(float) <= Size) {
            float float_value;
            std::memcpy(&float_value, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            try {
                torch::Tensor float_tensor = torch::tensor({float_value}, torch::kFloat32);
                torch::Tensor float_result = torch::signbit(float_tensor);
                (void)float_result;
            } catch (const std::exception&) {
                // Expected in some cases
            }
        }
        
        // Try with empty tensor
        try {
            torch::Tensor empty_tensor = torch::empty({0}, torch::kFloat32);
            torch::Tensor empty_result = torch::signbit(empty_tensor);
            (void)empty_result;
        } catch (const std::exception&) {
            // Empty tensor might cause issues
        }
        
        // Try with NaN and Inf values
        try {
            torch::Tensor special_values = torch::tensor({
                std::numeric_limits<float>::quiet_NaN(), 
                std::numeric_limits<float>::infinity(),
                -std::numeric_limits<float>::infinity(),
                0.0f, -0.0f, 1.0f, -1.0f
            });
            torch::Tensor special_result = torch::signbit(special_values);
            (void)special_result;
        } catch (const std::exception&) {
            // Special values might cause issues
        }
        
        // Try with multi-dimensional tensor
        try {
            torch::Tensor multi_dim = input_tensor.view({-1});
            if (multi_dim.numel() >= 4) {
                torch::Tensor reshaped = multi_dim.slice(0, 0, 4).view({2, 2});
                torch::Tensor reshaped_result = torch::signbit(reshaped);
                (void)reshaped_result;
            }
        } catch (const std::exception&) {
            // Reshape might fail
        }
        
        // Try with double precision
        try {
            torch::Tensor double_tensor = input_tensor.to(torch::kFloat64);
            torch::Tensor double_result = torch::signbit(double_tensor);
            (void)double_result;
        } catch (const std::exception&) {
            // Conversion might fail for some dtypes
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}