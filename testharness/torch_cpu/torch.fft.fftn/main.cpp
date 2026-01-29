#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <cstring>        // For std::memcpy

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // FFT requires at least 1D tensor
        if (input_tensor.dim() == 0) {
            return 0;
        }
        
        // Convert to float/complex type if needed (FFT requires floating point)
        if (!input_tensor.is_floating_point() && !input_tensor.is_complex()) {
            input_tensor = input_tensor.to(torch::kFloat32);
        }
        
        // Parse FFT parameters from the remaining data
        std::vector<int64_t> s;
        std::vector<int64_t> dim;
        std::string norm = "backward";
        
        // Parse 's' parameter (shape of output) - must be positive values
        if (offset + 1 < Size) {
            uint8_t s_rank = Data[offset++] % 4; // Limit to reasonable rank
            s_rank = std::min(s_rank, static_cast<uint8_t>(input_tensor.dim()));
            
            for (uint8_t i = 0; i < s_rank && offset < Size; i++) {
                // Use single byte to get reasonable positive sizes
                int64_t dim_size = (Data[offset++] % 64) + 1; // 1 to 64
                s.push_back(dim_size);
            }
        }
        
        // Parse 'dim' parameter (dimensions to transform)
        if (offset + 1 < Size) {
            uint8_t dim_count = Data[offset++] % (input_tensor.dim() + 1);
            dim_count = std::min(dim_count, static_cast<uint8_t>(input_tensor.dim()));
            
            for (uint8_t i = 0; i < dim_count && offset < Size; i++) {
                // Map to valid dimension indices
                int64_t dim_val = Data[offset++] % input_tensor.dim();
                // Avoid duplicates
                bool duplicate = false;
                for (auto d : dim) {
                    if (d == dim_val) {
                        duplicate = true;
                        break;
                    }
                }
                if (!duplicate) {
                    dim.push_back(dim_val);
                }
            }
        }
        
        // Parse 'norm' parameter
        if (offset < Size) {
            uint8_t norm_selector = Data[offset++] % 3;
            switch (norm_selector) {
                case 0: norm = "backward"; break;
                case 1: norm = "ortho"; break;
                case 2: norm = "forward"; break;
            }
        }
        
        torch::Tensor result;
        
        // Case 1: Basic fftn call
        try {
            result = torch::fft::fftn(input_tensor);
        } catch (...) {
            // Expected failures for edge cases
        }
        
        // Case 2: With s parameter
        if (!s.empty()) {
            try {
                result = torch::fft::fftn(input_tensor, s);
            } catch (...) {
                // Expected failures for invalid s values
            }
        }
        
        // Case 3: With dim parameter
        if (!dim.empty()) {
            try {
                result = torch::fft::fftn(input_tensor, c10::nullopt, dim);
            } catch (...) {
                // Expected failures for invalid dim values
            }
        }
        
        // Case 4: With norm parameter
        try {
            result = torch::fft::fftn(input_tensor, c10::nullopt, c10::nullopt, norm);
        } catch (...) {
            // Expected failures
        }
        
        // Case 5: With s and dim parameters (ensure matching sizes)
        if (!s.empty() && !dim.empty()) {
            // Adjust s to match dim length
            std::vector<int64_t> s_adjusted;
            for (size_t i = 0; i < dim.size() && i < s.size(); i++) {
                s_adjusted.push_back(s[i]);
            }
            if (!s_adjusted.empty()) {
                try {
                    result = torch::fft::fftn(input_tensor, s_adjusted, dim, norm);
                } catch (...) {
                    // Expected failures
                }
            }
        }
        
        // Ensure the result is used to prevent optimization
        if (result.defined()) {
            try {
                volatile float sum = result.abs().sum().item<float>();
                (void)sum;
            } catch (...) {
                // Complex sum might fail, ignore
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