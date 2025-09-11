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
        
        // Need at least 1 byte for window_length
        if (Size < 1) {
            return 0;
        }
        
        // Parse window_length from input data
        int64_t window_length = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&window_length, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        } else {
            // If not enough data, use a single byte
            window_length = static_cast<int64_t>(Data[offset++]);
        }
        
        // Parse periodic flag (true/false)
        bool periodic = false;
        if (offset < Size) {
            periodic = static_cast<bool>(Data[offset++] & 0x01);
        }
        
        // Parse dtype
        torch::ScalarType dtype = torch::kFloat;
        if (offset < Size) {
            dtype = fuzzer_utils::parseDataType(Data[offset++]);
        }
        
        // Parse layout
        torch::Layout layout = torch::kStrided;
        if (offset < Size) {
            layout = (Data[offset++] % 2 == 0) ? torch::kStrided : torch::kSparse;
        }
        
        // Parse device
        torch::Device device = torch::kCPU;
        
        // Parse requires_grad
        bool requires_grad = false;
        if (offset < Size) {
            requires_grad = static_cast<bool>(Data[offset++] & 0x01);
        }
        
        // Create options
        auto options = torch::TensorOptions()
            .dtype(dtype)
            .layout(layout)
            .device(device)
            .requires_grad(requires_grad);
        
        // Call hann_window with different combinations of parameters
        try {
            // Basic call with just window_length
            torch::Tensor result1 = torch::hann_window(window_length);
            
            // Call with window_length and periodic flag
            torch::Tensor result2 = torch::hann_window(window_length, periodic);
            
            // Call with window_length, periodic flag, and options
            torch::Tensor result3 = torch::hann_window(window_length, periodic, options);
        } catch (const c10::Error &e) {
            // PyTorch-specific exceptions are expected and handled
        }
        
        // Try with a tensor input for window_length - convert to scalar
        try {
            if (offset < Size) {
                torch::Tensor window_length_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Only proceed if the tensor can be interpreted as a scalar
                if (window_length_tensor.numel() == 1) {
                    int64_t scalar_window_length = window_length_tensor.item<int64_t>();
                    
                    torch::Tensor result4 = torch::hann_window(scalar_window_length);
                    
                    // With periodic flag
                    torch::Tensor result5 = torch::hann_window(scalar_window_length, periodic);
                    
                    // With options
                    torch::Tensor result6 = torch::hann_window(scalar_window_length, periodic, options);
                }
            }
        } catch (const c10::Error &e) {
            // PyTorch-specific exceptions are expected and handled
        }
        
        // Try with different dtypes
        try {
            auto float_options = torch::TensorOptions().dtype(torch::kFloat);
            torch::Tensor result7 = torch::hann_window(window_length, periodic, float_options);
            
            auto double_options = torch::TensorOptions().dtype(torch::kDouble);
            torch::Tensor result8 = torch::hann_window(window_length, periodic, double_options);
            
            auto half_options = torch::TensorOptions().dtype(torch::kHalf);
            torch::Tensor result9 = torch::hann_window(window_length, periodic, half_options);
            
            auto complex_options = torch::TensorOptions().dtype(torch::kComplexFloat);
            torch::Tensor result10 = torch::hann_window(window_length, periodic, complex_options);
        } catch (const c10::Error &e) {
            // PyTorch-specific exceptions are expected and handled
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
