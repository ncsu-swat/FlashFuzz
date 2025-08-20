#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least 3 bytes for start, end, and step values
        if (Size < 3) {
            return 0;
        }
        
        // Extract start, end, and step values from the input data
        double start = 0.0, end = 0.0, step = 1.0;
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&start, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&end, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&step, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Avoid zero step which would cause infinite loop
            if (step == 0.0) {
                step = 1.0;
            }
        }
        
        // Get dtype for the range operation
        torch::ScalarType dtype = torch::kFloat32;
        if (offset < Size) {
            dtype = fuzzer_utils::parseDataType(Data[offset++]);
        }
        
        // Get device for the range operation
        torch::Device device = torch::kCPU;
        
        // Try different variants of the range operation
        
        // Variant 1: Basic range with default step=1
        try {
            auto result1 = torch::range(start, end, torch::TensorOptions().dtype(dtype).device(device));
        } catch (const std::exception&) {
            // Allow exceptions from the API
        }
        
        // Variant 2: Range with custom step
        try {
            auto result2 = torch::range(start, end, step, torch::TensorOptions().dtype(dtype).device(device));
        } catch (const std::exception&) {
            // Allow exceptions from the API
        }
        
        // Variant 3: Range with different dtypes
        try {
            auto result3 = torch::range(start, end, step, torch::TensorOptions().dtype(torch::kInt64).device(device));
        } catch (const std::exception&) {
            // Allow exceptions from the API
        }
        
        try {
            auto result4 = torch::range(start, end, step, torch::TensorOptions().dtype(torch::kDouble).device(device));
        } catch (const std::exception&) {
            // Allow exceptions from the API
        }
        
        // Variant 4: Edge cases
        // Try with very large range
        try {
            auto result5 = torch::range(start, start + 1e6, step, torch::TensorOptions().dtype(dtype).device(device));
        } catch (const std::exception&) {
            // Allow exceptions from the API
        }
        
        // Try with negative step
        try {
            auto result6 = torch::range(end, start, -std::abs(step), torch::TensorOptions().dtype(dtype).device(device));
        } catch (const std::exception&) {
            // Allow exceptions from the API
        }
        
        // Try with very small step
        try {
            double tiny_step = (step > 0) ? 1e-10 : -1e-10;
            auto result7 = torch::range(start, end, tiny_step, torch::TensorOptions().dtype(dtype).device(device));
        } catch (const std::exception&) {
            // Allow exceptions from the API
        }
        
        // Try with extreme values
        try {
            auto result8 = torch::range(std::numeric_limits<double>::lowest(), std::numeric_limits<double>::max(), 
                                       step, torch::TensorOptions().dtype(dtype).device(device));
        } catch (const std::exception&) {
            // Allow exceptions from the API
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}