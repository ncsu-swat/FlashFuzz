#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <cmath>          // For isfinite

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
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor y
        torch::Tensor y = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure y is a floating point type for trapz
        if (!y.is_floating_point()) {
            y = y.to(torch::kFloat32);
        }
        
        // If we have enough data, create a second tensor x for the coordinates
        torch::Tensor x;
        bool has_x = false;
        if (offset + 4 < Size) {
            x = fuzzer_utils::createTensor(Data, Size, offset);
            if (!x.is_floating_point()) {
                x = x.to(torch::kFloat32);
            }
            has_x = true;
        }
        
        // Get a dimension to integrate over
        int64_t dim = 0;
        if (offset + sizeof(int8_t) <= Size) {
            int8_t dim_byte;
            std::memcpy(&dim_byte, Data + offset, sizeof(int8_t));
            offset += sizeof(int8_t);
            dim = static_cast<int64_t>(dim_byte);
        }
        
        // Get dx value from fuzzer data
        double dx = 1.0;
        if (offset + sizeof(float) <= Size) {
            float dx_float;
            std::memcpy(&dx_float, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Sanitize dx to avoid extreme values
            if (std::isfinite(dx_float) && dx_float != 0.0f) {
                dx = static_cast<double>(dx_float);
            }
        }
        
        // Try different variants of trapz
        try {
            // Basic trapz with default parameters (integrates over last dimension)
            if (y.dim() > 0 && y.numel() > 0) {
                torch::Tensor result1 = torch::trapz(y);
            }
        } catch (const c10::Error& e) {
            // Expected for some tensor configurations
        }
        
        try {
            // Trapz with dimension specified
            if (y.dim() > 0 && y.numel() > 0) {
                int64_t safe_dim = dim % y.dim();
                if (safe_dim < 0) {
                    safe_dim += y.dim();
                }
                torch::Tensor result2 = torch::trapz(y, safe_dim);
            }
        } catch (const c10::Error& e) {
            // Expected for some tensor configurations
        }
        
        try {
            // Trapz with dx spacing and dimension
            if (y.dim() > 0 && y.numel() > 0) {
                int64_t safe_dim = dim % y.dim();
                if (safe_dim < 0) {
                    safe_dim += y.dim();
                }
                torch::Tensor result3 = torch::trapz(y, dx, safe_dim);
            }
        } catch (const c10::Error& e) {
            // Expected for some tensor configurations
        }
        
        try {
            // Trapz with x coordinates and dimension
            if (has_x && x.defined() && y.dim() > 0 && y.numel() > 0) {
                int64_t safe_dim = dim % y.dim();
                if (safe_dim < 0) {
                    safe_dim += y.dim();
                }
                
                // Make x 1D and match the size of y along the integration dimension
                int64_t y_dim_size = y.size(safe_dim);
                if (x.numel() >= y_dim_size && y_dim_size > 0) {
                    torch::Tensor x_1d = x.flatten().slice(0, 0, y_dim_size);
                    if (!x_1d.is_floating_point()) {
                        x_1d = x_1d.to(torch::kFloat32);
                    }
                    torch::Tensor result4 = torch::trapz(y, x_1d, safe_dim);
                }
            }
        } catch (const c10::Error& e) {
            // Expected for some tensor configurations
        }
        
        try {
            // Test with negative dimension indexing
            if (y.dim() > 0 && y.numel() > 0) {
                torch::Tensor result_neg = torch::trapz(y, -1);
            }
        } catch (const c10::Error& e) {
            // Expected for some configurations
        }
        
        try {
            // Test with different dx values
            if (y.dim() > 0 && y.numel() > 0) {
                torch::Tensor result_dx_small = torch::trapz(y, 0.001, -1);
                torch::Tensor result_dx_large = torch::trapz(y, 100.0, -1);
            }
        } catch (const c10::Error& e) {
            // Expected for some configurations
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}