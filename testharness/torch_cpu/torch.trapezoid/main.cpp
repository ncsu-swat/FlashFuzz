#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <cmath>          // For isnan, isinf

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
        
        // Create input tensor y
        torch::Tensor y = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Need at least 1D tensor for trapezoid
        if (y.dim() == 0) {
            return 0;
        }
        
        // Get dim parameter if we have data left
        int64_t dim = 0;
        if (offset + sizeof(int8_t) <= Size) {
            int8_t dim_byte = static_cast<int8_t>(Data[offset]);
            offset += sizeof(int8_t);
            // Clamp dim to valid range for the tensor
            dim = dim_byte % y.dim();
            if (dim < 0) {
                dim += y.dim();
            }
        }
        
        // Get dx parameter if we have data left
        double dx = 1.0;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&dx, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Sanitize dx to avoid NaN/Inf issues
            if (std::isnan(dx) || std::isinf(dx) || dx == 0.0) {
                dx = 1.0;
            }
        }
        
        // Determine which variant to use based on remaining data
        uint8_t variant = 0;
        if (offset < Size) {
            variant = Data[offset] % 5;
            offset++;
        }
        
        try {
            switch (variant) {
                case 0: {
                    // Variant: trapezoid with just y
                    torch::Tensor result = torch::trapezoid(y);
                    break;
                }
                case 1: {
                    // Variant: trapezoid with y and dim
                    torch::Tensor result = torch::trapezoid(y, dim);
                    break;
                }
                case 2: {
                    // Variant: trapezoid with y and dx
                    torch::Tensor result = torch::trapezoid(y, dx);
                    break;
                }
                case 3: {
                    // Variant: trapezoid with y, dx, and dim
                    torch::Tensor result = torch::trapezoid(y, dx, dim);
                    break;
                }
                case 4: {
                    // Variant: trapezoid with x tensor
                    // x should have same size as y along the integration dim
                    if (offset + 4 < Size) {
                        // Create x tensor with same shape as y
                        torch::Tensor x = fuzzer_utils::createTensor(Data, Size, offset);
                        
                        // x must be 1D with size matching y.size(dim)
                        // or have same shape as y
                        if (x.numel() > 0) {
                            // Reshape x to be 1D matching y's size along dim
                            int64_t dim_size = y.size(dim);
                            if (x.numel() >= dim_size && dim_size > 0) {
                                x = x.flatten().slice(0, 0, dim_size);
                                torch::Tensor result = torch::trapezoid(y, x, dim);
                            }
                        }
                    }
                    break;
                }
            }
        } catch (const std::exception &e) {
            // Inner catch for expected shape/type mismatches - silently ignore
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}