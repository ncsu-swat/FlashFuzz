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
        
        // Need at least 1 byte for n and 1 byte for d
        if (Size < 2) {
            return 0;
        }
        
        // Parse n (number of points)
        int64_t n = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&n, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        } else {
            // Not enough data for int64_t, use what's available
            for (size_t i = 0; i < std::min(Size - offset, sizeof(int64_t)); i++) {
                n = (n << 8) | Data[offset++];
            }
        }
        
        // Parse d (sample spacing)
        double d = 1.0;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&d, Data + offset, sizeof(double));
            offset += sizeof(double);
        } else if (offset < Size) {
            // Use remaining bytes to construct a double
            uint64_t d_bits = 0;
            for (size_t i = 0; i < std::min(Size - offset, sizeof(double)); i++) {
                d_bits = (d_bits << 8) | Data[offset++];
            }
            std::memcpy(&d, &d_bits, sizeof(double));
        }
        
        // Try different variants of rfftfreq
        
        // Variant 1: Using scalar n with default options
        torch::Tensor result1 = torch::fft::rfftfreq(n, torch::TensorOptions());
        
        // Variant 2: Using scalar n with d parameter
        torch::Tensor result2 = torch::fft::rfftfreq(n, d, torch::TensorOptions());
        
        // Try with different dtypes for options
        if (offset < Size) {
            auto dtype_selector = Data[offset++];
            auto dtype = fuzzer_utils::parseDataType(dtype_selector);
            
            auto options = torch::TensorOptions().dtype(dtype);
            
            // Variant 3: With dtype options
            torch::Tensor result3 = torch::fft::rfftfreq(n, d, options);
            
            // Variant 4: With just n and dtype options
            torch::Tensor result4 = torch::fft::rfftfreq(n, options);
        }
        
        // Try with different device options if there's more data
        if (offset < Size) {
            bool use_cuda = Data[offset++] % 2 == 0;
            
            if (use_cuda && torch::cuda::is_available()) {
                auto device_options = torch::TensorOptions().device(torch::kCUDA);
                
                // Variant 5: With CUDA device
                torch::Tensor result5 = torch::fft::rfftfreq(n, d, device_options);
                
                // Variant 6: With just n and CUDA device
                torch::Tensor result6 = torch::fft::rfftfreq(n, device_options);
            }
        }
        
        // Try with negative n (should throw exception)
        if (offset < Size && Data[offset++] % 2 == 0) {
            try {
                torch::Tensor result_neg = torch::fft::rfftfreq(-std::abs(n), torch::TensorOptions());
            } catch (const c10::Error &e) {
                // Expected exception for negative n
            }
        }
        
        // Try with zero n
        if (offset < Size && Data[offset++] % 2 == 0) {
            try {
                torch::Tensor result_zero = torch::fft::rfftfreq(0, torch::TensorOptions());
            } catch (const c10::Error &e) {
                // May throw exception for n=0
            }
        }
        
        // Try with very large n
        if (offset < Size && Data[offset++] % 2 == 0) {
            try {
                int64_t large_n = std::numeric_limits<int32_t>::max();
                torch::Tensor result_large = torch::fft::rfftfreq(large_n, torch::TensorOptions());
            } catch (const std::exception &e) {
                // May throw for very large n
            }
        }
        
        // Try with NaN or Inf for d
        if (offset < Size && Data[offset++] % 3 == 0) {
            try {
                torch::Tensor result_nan = torch::fft::rfftfreq(n, std::numeric_limits<double>::quiet_NaN(), torch::TensorOptions());
            } catch (const std::exception &e) {
                // May throw for NaN d
            }
        } else if (offset < Size && Data[offset++] % 3 == 1) {
            try {
                torch::Tensor result_inf = torch::fft::rfftfreq(n, std::numeric_limits<double>::infinity(), torch::TensorOptions());
            } catch (const std::exception &e) {
                // May throw for Inf d
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
