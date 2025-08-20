#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for basic parameters
        if (Size < 4) {
            return 0;
        }
        
        // Create a tensor for n (number of points)
        torch::Tensor n_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract a scalar value for n
        int64_t n = 0;
        if (n_tensor.numel() > 0) {
            if (n_tensor.dtype() == torch::kFloat || n_tensor.dtype() == torch::kDouble) {
                n = static_cast<int64_t>(n_tensor.item<float>());
            } else if (n_tensor.dtype() == torch::kInt || n_tensor.dtype() == torch::kLong) {
                n = n_tensor.item<int64_t>();
            } else {
                // For other types, just use the first byte as a small integer
                n = n_tensor.data_ptr<uint8_t>()[0];
            }
        }
        
        // Create a tensor for d (sample spacing)
        torch::Tensor d_tensor;
        if (offset < Size) {
            d_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            d_tensor = torch::tensor(1.0);
        }
        
        // Extract a scalar value for d
        double d = 1.0;
        if (d_tensor.numel() > 0) {
            if (d_tensor.dtype() == torch::kFloat || d_tensor.dtype() == torch::kDouble) {
                d = d_tensor.item<float>();
            } else if (d_tensor.dtype() == torch::kInt || d_tensor.dtype() == torch::kLong) {
                d = static_cast<double>(d_tensor.item<int64_t>());
            } else {
                // For other types, just use a default value
                d = 1.0;
            }
        }
        
        // Try different tensor options
        std::vector<torch::TensorOptions> options = {
            torch::TensorOptions().dtype(torch::kFloat),
            torch::TensorOptions().dtype(torch::kDouble),
            torch::TensorOptions().dtype(torch::kInt),
            torch::TensorOptions().dtype(torch::kLong)
        };
        
        // Try with different options
        for (const auto& opt : options) {
            if (offset >= Size) break;
            
            // Get a byte to determine if we should use normalized form
            bool normalized = false;
            if (offset < Size) {
                normalized = (Data[offset++] % 2 == 0);
            }
            
            try {
                // Call fftfreq with different parameters
                torch::Tensor result;
                
                if (normalized) {
                    result = torch::fft::fftfreq(n, d, opt);
                } else {
                    result = torch::fft::fftfreq(n, opt);
                }
                
                // Try to access elements to ensure it's valid
                if (result.numel() > 0) {
                    auto first_element = result[0];
                }
            } catch (const std::exception& e) {
                // Catch and continue to the next option
            }
        }
        
        // Try with extreme values for n and d
        if (offset < Size) {
            try {
                // Use a potentially large value for n
                int64_t extreme_n = 0;
                if (offset + sizeof(int64_t) <= Size) {
                    std::memcpy(&extreme_n, Data + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                }
                
                // Use a potentially extreme value for d
                double extreme_d = 0.0;
                if (offset + sizeof(double) <= Size) {
                    std::memcpy(&extreme_d, Data + offset, sizeof(double));
                    offset += sizeof(double);
                }
                
                // Call fftfreq with extreme values
                auto result = torch::fft::fftfreq(extreme_n, extreme_d);
            } catch (const std::exception& e) {
                // Expected to potentially throw for extreme values
            }
        }
        
        // Try with negative n (should throw an error)
        if (offset < Size) {
            try {
                int64_t negative_n = -1;
                if (offset < Size) {
                    negative_n = -std::abs(static_cast<int64_t>(Data[offset++]));
                }
                
                auto result = torch::fft::fftfreq(negative_n);
            } catch (const std::exception& e) {
                // Expected to throw for negative n
            }
        }
        
        // Try with zero d (should throw an error)
        if (offset < Size) {
            try {
                auto result = torch::fft::fftfreq(10, 0.0);
            } catch (const std::exception& e) {
                // Expected to throw for d=0
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