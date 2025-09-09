#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some bytes for parameters
        if (Size < 16) {
            return 0;
        }
        
        // Extract fuzzing parameters
        uint8_t variant = extract_uint8(Data, Size, offset);
        variant = variant % 6; // 6 different test variants
        
        switch (variant) {
            case 0: {
                // Test torch::arange(end)
                double end = extract_double(Data, Size, offset);
                // Clamp to reasonable range to avoid memory issues
                end = std::clamp(end, -10000.0, 10000.0);
                
                auto result = torch::arange(end);
                // Force evaluation
                result.sum();
                break;
            }
            
            case 1: {
                // Test torch::arange(start, end)
                double start = extract_double(Data, Size, offset);
                double end = extract_double(Data, Size, offset);
                
                // Clamp to reasonable ranges
                start = std::clamp(start, -10000.0, 10000.0);
                end = std::clamp(end, -10000.0, 10000.0);
                
                auto result = torch::arange(start, end);
                result.sum();
                break;
            }
            
            case 2: {
                // Test torch::arange(start, end, step)
                double start = extract_double(Data, Size, offset);
                double end = extract_double(Data, Size, offset);
                double step = extract_double(Data, Size, offset);
                
                // Clamp values and ensure step is not zero
                start = std::clamp(start, -10000.0, 10000.0);
                end = std::clamp(end, -10000.0, 10000.0);
                step = std::clamp(step, -1000.0, 1000.0);
                if (std::abs(step) < 1e-10) {
                    step = 1.0; // Avoid zero step
                }
                
                auto result = torch::arange(start, end, step);
                result.sum();
                break;
            }
            
            case 3: {
                // Test with different dtypes
                int64_t end = extract_int64(Data, Size, offset);
                end = std::clamp(end, -10000LL, 10000LL);
                
                uint8_t dtype_choice = extract_uint8(Data, Size, offset) % 6;
                torch::Dtype dtype;
                switch (dtype_choice) {
                    case 0: dtype = torch::kInt32; break;
                    case 1: dtype = torch::kInt64; break;
                    case 2: dtype = torch::kFloat32; break;
                    case 3: dtype = torch::kFloat64; break;
                    case 4: dtype = torch::kInt8; break;
                    case 5: dtype = torch::kInt16; break;
                }
                
                auto options = torch::TensorOptions().dtype(dtype);
                auto result = torch::arange(end, options);
                result.sum();
                break;
            }
            
            case 4: {
                // Test with device options (CPU/CUDA if available)
                float start = extract_float(Data, Size, offset);
                float end = extract_float(Data, Size, offset);
                
                start = std::clamp(start, -1000.0f, 1000.0f);
                end = std::clamp(end, -1000.0f, 1000.0f);
                
                auto options = torch::TensorOptions().dtype(torch::kFloat32);
                
                // Test CPU device
                auto result_cpu = torch::arange(start, end, options.device(torch::kCPU));
                result_cpu.sum();
                
                // Test CUDA if available
                if (torch::cuda::is_available()) {
                    auto result_cuda = torch::arange(start, end, options.device(torch::kCUDA));
                    result_cuda.sum();
                }
                break;
            }
            
            case 5: {
                // Test edge cases with integer types
                int32_t start = extract_int32(Data, Size, offset);
                int32_t end = extract_int32(Data, Size, offset);
                int32_t step = extract_int32(Data, Size, offset);
                
                // Clamp to avoid excessive memory usage
                start = std::clamp(start, -10000, 10000);
                end = std::clamp(end, -10000, 10000);
                step = std::clamp(step, -1000, 1000);
                if (step == 0) step = 1; // Avoid zero step
                
                auto result = torch::arange(start, end, step, torch::TensorOptions().dtype(torch::kInt32));
                result.sum();
                break;
            }
        }
        
        // Additional edge case testing
        if (offset < Size - 8) {
            uint8_t edge_case = extract_uint8(Data, Size, offset) % 4;
            
            switch (edge_case) {
                case 0: {
                    // Test with very small positive step
                    auto result = torch::arange(0.0, 1.0, 0.1);
                    result.sum();
                    break;
                }
                case 1: {
                    // Test with negative step
                    auto result = torch::arange(10.0, 0.0, -1.0);
                    result.sum();
                    break;
                }
                case 2: {
                    // Test empty range
                    auto result = torch::arange(5.0, 5.0);
                    result.sum();
                    break;
                }
                case 3: {
                    // Test single element
                    auto result = torch::arange(1);
                    result.sum();
                    break;
                }
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}