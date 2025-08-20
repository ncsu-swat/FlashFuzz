#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least 3 bytes for start, end, step values
        if (Size < 3) {
            return 0;
        }
        
        // Extract parameters for torch::arange
        double start = 0.0;
        double end = 0.0;
        double step = 1.0;
        
        // Parse start value
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&start, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        // Parse end value
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&end, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        // Parse step value
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&step, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        // Prevent step = 0 which causes infinite loop
        if (step == 0.0) {
            step = 1.0;
        }
        
        // Get a data type for the tensor
        torch::ScalarType dtype = torch::kFloat32;
        if (offset < Size) {
            dtype = fuzzer_utils::parseDataType(Data[offset++]);
        }
        
        // Get device option
        torch::Device device = torch::kCPU;
        
        // Create options
        auto options = torch::TensorOptions().dtype(dtype).device(device);
        
        // Test different variants of torch::arange
        
        // Variant 1: torch::arange(end)
        torch::Tensor t1 = torch::arange(end, options);
        
        // Variant 2: torch::arange(start, end)
        torch::Tensor t2 = torch::arange(start, end, options);
        
        // Variant 3: torch::arange(start, end, step)
        torch::Tensor t3 = torch::arange(start, end, step, options);
        
        // Test edge cases with different options
        if (offset + 1 < Size) {
            // Try with different data types
            auto alt_dtype = fuzzer_utils::parseDataType(Data[offset++]);
            auto alt_options = torch::TensorOptions().dtype(alt_dtype).device(device);
            
            // Edge case: very small step size
            double tiny_step = step * 1e-10;
            if (tiny_step == 0.0) tiny_step = 1e-10;
            
            torch::Tensor t4 = torch::arange(start, end, tiny_step, alt_options);
            
            // Edge case: very large range
            double large_start = start * 1e10;
            double large_end = end * 1e10;
            double large_step = step * 1e9;
            if (large_step == 0.0) large_step = 1e9;
            
            torch::Tensor t5 = torch::arange(large_start, large_end, large_step, alt_options);
            
            // Edge case: negative step
            torch::Tensor t6 = torch::arange(end, start, -std::abs(step), alt_options);
            
            // Edge case: start == end
            torch::Tensor t7 = torch::arange(start, start, step, alt_options);
        }
        
        // Test with integer types specifically
        if (offset < Size) {
            int64_t int_start = static_cast<int64_t>(start);
            int64_t int_end = static_cast<int64_t>(end);
            int64_t int_step = static_cast<int64_t>(step);
            if (int_step == 0) int_step = 1;
            
            auto int_options = torch::TensorOptions().dtype(torch::kInt64).device(device);
            torch::Tensor t8 = torch::arange(int_start, int_end, int_step, int_options);
            
            // Test with boolean type
            auto bool_options = torch::TensorOptions().dtype(torch::kBool).device(device);
            torch::Tensor t9 = torch::arange(int_start, int_end, int_step, bool_options);
        }
        
        // Test with complex types
        if (offset < Size) {
            auto complex_options = torch::TensorOptions().dtype(torch::kComplexFloat).device(device);
            torch::Tensor t10 = torch::arange(start, end, step, complex_options);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}