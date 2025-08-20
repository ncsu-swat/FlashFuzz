#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse q (quantile value between 0 and 1)
        double q = 0.5;
        if (offset + sizeof(double) <= Size) {
            double raw_q;
            std::memcpy(&raw_q, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Ensure q is between 0 and 1
            q = std::abs(raw_q);
            q = q - std::floor(q);
        }
        
        // Parse dim
        int64_t dim = 0;
        if (offset + sizeof(int64_t) <= Size) {
            int64_t raw_dim;
            std::memcpy(&raw_dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Allow negative dimensions for testing edge cases
            if (input.dim() > 0) {
                dim = raw_dim % input.dim();
                if (dim < 0) {
                    dim += input.dim();
                }
            }
        }
        
        // Parse keepdim
        bool keepdim = false;
        if (offset < Size) {
            keepdim = Data[offset++] & 0x1;
        }
        
        // Parse interpolation
        std::string interpolation = "linear";
        if (offset < Size) {
            uint8_t interp_selector = Data[offset++] % 4;
            switch (interp_selector) {
                case 0: interpolation = "linear"; break;
                case 1: interpolation = "lower"; break;
                case 2: interpolation = "higher"; break;
                case 3: interpolation = "midpoint"; break;
            }
        }
        
        // Try different variants of nanquantile
        if (offset < Size) {
            uint8_t variant = Data[offset++] % 4;
            
            switch (variant) {
                case 0: {
                    // Basic nanquantile with just q
                    torch::Tensor result = torch::nanquantile(input, q);
                    break;
                }
                case 1: {
                    // nanquantile with q and dim
                    torch::Tensor result = torch::nanquantile(input, q, dim);
                    break;
                }
                case 2: {
                    // nanquantile with q, dim, and keepdim
                    torch::Tensor result = torch::nanquantile(input, q, dim, keepdim);
                    break;
                }
                case 3: {
                    // Full nanquantile with all parameters
                    torch::Tensor result = torch::nanquantile(input, q, dim, keepdim, interpolation);
                    break;
                }
            }
        } else {
            // Default case if we don't have enough data
            torch::Tensor result = torch::nanquantile(input, q);
        }
        
        // Try with q as a tensor
        if (offset + 1 < Size) {
            // Create a tensor with q values
            std::vector<double> q_values;
            uint8_t num_q = Data[offset++] % 5 + 1; // 1 to 5 q values
            
            for (uint8_t i = 0; i < num_q && offset + sizeof(double) <= Size; i++) {
                double raw_q;
                std::memcpy(&raw_q, Data + offset, sizeof(double));
                offset += sizeof(double);
                
                // Ensure q is between 0 and 1
                double q_val = std::abs(raw_q);
                q_val = q_val - std::floor(q_val);
                q_values.push_back(q_val);
            }
            
            if (!q_values.empty()) {
                torch::Tensor q_tensor = torch::tensor(q_values);
                
                // Try different variants with q as tensor
                if (offset < Size) {
                    uint8_t tensor_variant = Data[offset++] % 4;
                    
                    switch (tensor_variant) {
                        case 0: {
                            // Basic nanquantile with q tensor
                            torch::Tensor result = torch::nanquantile(input, q_tensor);
                            break;
                        }
                        case 1: {
                            // nanquantile with q tensor and dim
                            torch::Tensor result = torch::nanquantile(input, q_tensor, dim);
                            break;
                        }
                        case 2: {
                            // nanquantile with q tensor, dim, and keepdim
                            torch::Tensor result = torch::nanquantile(input, q_tensor, dim, keepdim);
                            break;
                        }
                        case 3: {
                            // Full nanquantile with all parameters and q tensor
                            torch::Tensor result = torch::nanquantile(input, q_tensor, dim, keepdim, interpolation);
                            break;
                        }
                    }
                } else {
                    // Default case with q tensor
                    torch::Tensor result = torch::nanquantile(input, q_tensor);
                }
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