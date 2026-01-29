#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr, cout
#include <cstring>        // For std::memcpy
#include <cmath>          // For std::isnan, std::isinf

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
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // cumulative_trapezoid requires at least 1D tensor with size >= 1 along the integration dim
        if (input.dim() == 0) {
            return 0;
        }
        
        // Get a dimension to use for the operation
        int64_t dim = -1;
        if (offset + sizeof(int64_t) <= Size) {
            int64_t raw_dim = 0;
            std::memcpy(&raw_dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Normalize dim to valid range [0, ndim-1]
            int64_t ndim = input.dim();
            dim = ((raw_dim % ndim) + ndim) % ndim;
        } else {
            dim = input.dim() - 1;  // Default to last dimension
        }
        
        // Check that the dimension size is at least 1 for trapezoid integration
        if (input.size(dim) < 1) {
            return 0;
        }
        
        // Get dx scalar value
        double dx = 1.0;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&dx, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Sanitize dx to avoid problematic values
            if (std::isnan(dx) || std::isinf(dx) || dx == 0.0) {
                dx = 1.0;
            }
        }
        
        // Decide whether to use x tensor variant
        bool use_x = false;
        if (offset < Size) {
            use_x = (Data[offset++] % 2 == 0);
        }
        
        // Create x tensor for coordinate-based integration
        torch::Tensor x;
        if (use_x && offset + 4 < Size) {
            try {
                // x should be 1D with same size as input along dim, or same shape as input
                int64_t x_size = input.size(dim);
                
                // Create 1D x coordinates
                x = torch::linspace(0.0, 1.0, x_size, input.options().dtype(torch::kFloat64));
                
                // Add some noise from fuzzer data to make x more interesting
                if (offset + sizeof(float) <= Size) {
                    float scale;
                    std::memcpy(&scale, Data + offset, sizeof(float));
                    offset += sizeof(float);
                    if (!std::isnan(scale) && !std::isinf(scale) && scale != 0.0f) {
                        x = x * static_cast<double>(std::abs(scale));
                    }
                }
            } catch (const std::exception&) {
                use_x = false;
            }
        } else {
            use_x = false;
        }
        
        // Determine which variant to test
        int variant = 0;
        if (offset < Size) {
            variant = Data[offset++] % 3;
        }
        
        try {
            torch::Tensor result;
            
            if (variant == 0) {
                // Variant 1: cumulative_trapezoid with dx scalar and dim
                result = torch::cumulative_trapezoid(input, dx, dim);
            } 
            else if (variant == 1 && use_x) {
                // Variant 2: cumulative_trapezoid with x tensor and dim
                result = torch::cumulative_trapezoid(input, x, dim);
            }
            else {
                // Variant 3: cumulative_trapezoid with default dx=1.0
                result = torch::cumulative_trapezoid(input, 1.0, dim);
            }
            
            // Ensure result is used
            if (result.numel() > 0) {
                auto sum = result.sum();
                (void)sum;
            }
        }
        catch (const c10::Error &e) {
            // Expected failures (shape mismatches, etc.) - silently ignore
        }
        
        // Test with different dtypes if we have more data
        if (offset < Size) {
            try {
                torch::ScalarType dtype = fuzzer_utils::parseDataType(Data[offset++]);
                torch::Tensor typed_input = input.to(dtype);
                
                // Only run if the conversion was successful and tensor is valid
                if (typed_input.defined() && typed_input.dim() > 0) {
                    int64_t valid_dim = dim % typed_input.dim();
                    if (valid_dim < 0) valid_dim += typed_input.dim();
                    
                    torch::Tensor result = torch::cumulative_trapezoid(typed_input, 1.0, valid_dim);
                    (void)result.sum();
                }
            }
            catch (const c10::Error &e) {
                // Expected failures for dtype conversion - silently ignore
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}