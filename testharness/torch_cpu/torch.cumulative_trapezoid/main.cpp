#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get a dimension to use for the operation
        int64_t dim = -1;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // If tensor has dimensions, ensure dim is within valid range
        if (input.dim() > 0) {
            dim = dim % input.dim();
            if (dim < 0) {
                dim += input.dim();
            }
        }
        
        // Create a second tensor for x coordinates (optional parameter)
        torch::Tensor x;
        bool use_x = false;
        
        if (offset < Size) {
            use_x = (Data[offset++] % 2 == 0);
            
            if (use_x && offset < Size) {
                try {
                    x = fuzzer_utils::createTensor(Data, Size, offset);
                } catch (const std::exception&) {
                    use_x = false;
                }
            }
        }
        
        // Try different variants of cumulative_trapezoid
        if (input.dim() > 0) {
            try {
                // Variant 1: Basic cumulative_trapezoid with dimension
                torch::Tensor result1 = torch::cumulative_trapezoid(input, 1.0, dim);
                
                // Variant 2: With x coordinates if available
                if (use_x) {
                    torch::Tensor result2 = torch::cumulative_trapezoid(input, x, dim);
                }
                
                // Variant 3: With different dx scalar
                if (offset < Size) {
                    double dx = 1.0;
                    if (offset + sizeof(double) <= Size) {
                        std::memcpy(&dx, Data + offset, sizeof(double));
                        offset += sizeof(double);
                        if (std::isnan(dx) || std::isinf(dx) || dx == 0.0) {
                            dx = 1.0;
                        }
                    }
                    
                    torch::Tensor result3 = torch::cumulative_trapezoid(input, dx, dim);
                }
            } catch (const std::exception&) {
                // Allow exceptions from the operation itself
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