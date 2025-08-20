#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse norm parameters from remaining data
        torch::Scalar ord;
        std::vector<int64_t> dim;
        bool keepdim = false;
        
        // Parse ord parameter (if we have data left)
        if (offset + sizeof(float) <= Size) {
            float ord_value;
            std::memcpy(&ord_value, Data + offset, sizeof(float));
            offset += sizeof(float);
            ord = torch::Scalar(ord_value);
        } else {
            // Default to Frobenius norm
            ord = torch::Scalar(2.0);
        }
        
        // Parse dim parameter (if we have data left)
        if (offset < Size) {
            uint8_t dim_count = Data[offset++] % 4; // Up to 3 dimensions
            
            for (uint8_t i = 0; i < dim_count && offset < Size; i++) {
                int8_t dim_value = static_cast<int8_t>(Data[offset++]);
                dim.push_back(dim_value);
            }
        }
        
        // Parse keepdim parameter (if we have data left)
        if (offset < Size) {
            keepdim = Data[offset++] & 1; // Use lowest bit to determine boolean
        }
        
        // Apply torch.norm with different parameter combinations
        torch::Tensor result;
        
        // Try different combinations of parameters
        if (dim.empty()) {
            // Case 1: Just tensor and ord
            result = torch::norm(input, ord);
        } else {
            // Case 2: tensor, ord, dim, keepdim
            result = torch::norm(input, ord, dim, keepdim);
        }
        
        // Try alternative overloads if we have enough data left
        if (offset < Size) {
            uint8_t variant = Data[offset++] % 3;
            
            switch (variant) {
                case 0:
                    // "fro" norm
                    result = torch::norm(input, "fro");
                    break;
                case 1:
                    // "nuc" norm
                    result = torch::norm(input, "nuc");
                    break;
                case 2:
                    // Try inf norm
                    result = torch::norm(input, std::numeric_limits<double>::infinity());
                    break;
            }
        }
        
        // Try vector norm if we have enough data left
        if (offset < Size && input.dim() == 1) {
            float p_value;
            if (offset + sizeof(float) <= Size) {
                std::memcpy(&p_value, Data + offset, sizeof(float));
                offset += sizeof(float);
                result = torch::norm(input, p_value);
            }
        }
        
        // Try matrix norm if we have enough data left
        if (offset < Size && input.dim() == 2) {
            uint8_t norm_type = Data[offset++] % 3;
            switch (norm_type) {
                case 0:
                    result = torch::norm(input, 1);
                    break;
                case 1:
                    result = torch::norm(input, std::numeric_limits<double>::infinity());
                    break;
                case 2:
                    result = torch::norm(input, "fro");
                    break;
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