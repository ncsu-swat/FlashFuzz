#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least enough data for window_length and optional parameters
        if (Size < sizeof(int64_t)) {
            return 0;
        }

        // Extract window_length (required parameter)
        int64_t window_length = extract_int64_t(Data, Size, offset);
        
        // Clamp window_length to reasonable range to avoid memory issues
        window_length = std::max(int64_t(0), std::min(window_length, int64_t(10000)));

        // Test basic hann_window with just window_length
        auto result1 = torch::hann_window(window_length);

        // If we have more data, test with periodic parameter
        if (offset + sizeof(bool) <= Size) {
            bool periodic = extract_bool(Data, Size, offset);
            auto result2 = torch::hann_window(window_length, periodic);
        }

        // If we have even more data, test with dtype parameter
        if (offset + sizeof(int32_t) <= Size) {
            int32_t dtype_int = extract_int32_t(Data, Size, offset);
            
            // Map to valid torch dtypes
            std::vector<torch::Dtype> valid_dtypes = {
                torch::kFloat32, torch::kFloat64, torch::kFloat16
            };
            
            torch::Dtype dtype = valid_dtypes[std::abs(dtype_int) % valid_dtypes.size()];
            
            auto result3 = torch::hann_window(window_length, torch::TensorOptions().dtype(dtype));
            
            // Test with both periodic and dtype
            if (offset + sizeof(bool) <= Size) {
                bool periodic2 = extract_bool(Data, Size, offset);
                auto result4 = torch::hann_window(window_length, periodic2, torch::TensorOptions().dtype(dtype));
            }
        }

        // If we have more data, test with layout parameter
        if (offset + sizeof(int32_t) <= Size) {
            int32_t layout_int = extract_int32_t(Data, Size, offset);
            
            // Map to valid layouts
            std::vector<torch::Layout> valid_layouts = {
                torch::kStrided, torch::kSparse
            };
            
            torch::Layout layout = valid_layouts[std::abs(layout_int) % valid_layouts.size()];
            
            auto result5 = torch::hann_window(window_length, torch::TensorOptions().layout(layout));
        }

        // If we have more data, test with device parameter
        if (offset + sizeof(int32_t) <= Size) {
            int32_t device_int = extract_int32_t(Data, Size, offset);
            
            // Test with CPU device (most common and available)
            torch::Device device = torch::kCPU;
            
            auto result6 = torch::hann_window(window_length, torch::TensorOptions().device(device));
        }

        // Test with requires_grad parameter
        if (offset + sizeof(bool) <= Size) {
            bool requires_grad = extract_bool(Data, Size, offset);
            
            auto result7 = torch::hann_window(window_length, torch::TensorOptions().requires_grad(requires_grad));
        }

        // Test edge cases
        if (window_length == 0) {
            auto empty_result = torch::hann_window(0);
        }
        
        if (window_length == 1) {
            auto single_result = torch::hann_window(1);
        }

        // Test with combination of all parameters
        if (offset + sizeof(bool) + sizeof(int32_t) * 2 <= Size) {
            bool periodic_final = extract_bool(Data, Size, offset);
            int32_t dtype_final = extract_int32_t(Data, Size, offset);
            int32_t device_final = extract_int32_t(Data, Size, offset);
            
            std::vector<torch::Dtype> dtypes = {torch::kFloat32, torch::kFloat64};
            torch::Dtype final_dtype = dtypes[std::abs(dtype_final) % dtypes.size()];
            
            auto final_result = torch::hann_window(
                window_length, 
                periodic_final,
                torch::TensorOptions()
                    .dtype(final_dtype)
                    .device(torch::kCPU)
                    .requires_grad(false)
            );
            
            // Verify the result has expected properties
            if (final_result.defined()) {
                auto sizes = final_result.sizes();
                auto dtype_check = final_result.dtype();
                auto device_check = final_result.device();
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