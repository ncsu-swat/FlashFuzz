#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least enough data for basic parameters
        if (Size < 32) {
            return 0;
        }

        // Extract start value (double)
        double start = extract_double(Data, Size, offset);
        
        // Extract end value (double)
        double end = extract_double(Data, Size, offset);
        
        // Extract steps (int64_t, but ensure it's reasonable)
        int64_t steps_raw = extract_int64(Data, Size, offset);
        // Clamp steps to reasonable range to avoid memory issues
        int64_t steps = std::max(1L, std::min(steps_raw, 10000L));
        
        // Extract dtype choice
        uint8_t dtype_choice = extract_uint8(Data, Size, offset) % 6;
        torch::ScalarType dtype;
        switch (dtype_choice) {
            case 0: dtype = torch::kFloat32; break;
            case 1: dtype = torch::kFloat64; break;
            case 2: dtype = torch::kInt32; break;
            case 3: dtype = torch::kInt64; break;
            case 4: dtype = torch::kFloat16; break;
            default: dtype = torch::kFloat32; break;
        }
        
        // Extract device choice
        uint8_t device_choice = extract_uint8(Data, Size, offset) % 2;
        torch::Device device = (device_choice == 0) ? torch::kCPU : torch::kCUDA;
        
        // Test basic linspace
        auto result1 = torch::linspace(start, end, steps);
        
        // Test with dtype specified
        auto result2 = torch::linspace(start, end, steps, torch::TensorOptions().dtype(dtype));
        
        // Test with device specified (if CUDA available)
        if (torch::cuda::is_available() && device.is_cuda()) {
            auto result3 = torch::linspace(start, end, steps, torch::TensorOptions().device(device));
        }
        
        // Test with both dtype and device
        auto result4 = torch::linspace(start, end, steps, torch::TensorOptions().dtype(dtype).device(torch::kCPU));
        
        // Test edge cases
        if (offset + 8 < Size) {
            // Test with steps = 1
            auto result5 = torch::linspace(start, end, 1);
            
            // Test with same start and end
            auto result6 = torch::linspace(start, start, steps);
            
            // Test with negative range
            auto result7 = torch::linspace(end, start, steps);
        }
        
        // Test with extreme values
        if (offset + 16 < Size) {
            double extreme1 = extract_double(Data, Size, offset);
            double extreme2 = extract_double(Data, Size, offset);
            
            // Clamp extreme values to avoid overflow
            extreme1 = std::max(-1e10, std::min(extreme1, 1e10));
            extreme2 = std::max(-1e10, std::min(extreme2, 1e10));
            
            auto result8 = torch::linspace(extreme1, extreme2, std::min(steps, 100L));
        }
        
        // Verify results have correct shape
        if (result1.size(0) != steps) {
            std::cerr << "Unexpected tensor size" << std::endl;
        }
        
        // Test accessing elements to trigger potential issues
        if (result1.numel() > 0) {
            auto first = result1[0];
            if (result1.numel() > 1) {
                auto last = result1[-1];
            }
        }
        
        // Test with tensor operations to ensure validity
        if (result2.numel() > 1) {
            auto sum = torch::sum(result2);
            auto mean = torch::mean(result2);
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}