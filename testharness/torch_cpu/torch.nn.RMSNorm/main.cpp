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
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for RMSNorm
        // Get normalized_shape from the remaining data
        std::vector<int64_t> normalized_shape;
        if (offset + 1 < Size) {
            uint8_t dim_count = Data[offset++] % 4 + 1; // 1-4 dimensions
            
            for (uint8_t i = 0; i < dim_count && offset < Size; i++) {
                if (offset + sizeof(int64_t) <= Size) {
                    int64_t dim_size;
                    std::memcpy(&dim_size, Data + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                    
                    // Allow any dimension size including 0
                    dim_size = std::abs(dim_size) % 16;
                    normalized_shape.push_back(dim_size);
                } else {
                    normalized_shape.push_back(1); // Default if not enough data
                    offset = Size; // Prevent further reading
                }
            }
        } else {
            // Default if not enough data
            normalized_shape.push_back(1);
        }
        
        // Get epsilon parameter
        double epsilon = 1e-5; // Default value
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&epsilon, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Ensure epsilon is positive and not too small
            if (epsilon <= 0 || !std::isfinite(epsilon)) {
                epsilon = 1e-5;
            }
        }
        
        // Get weight and bias parameters
        bool use_weight = false;
        
        if (offset < Size) {
            uint8_t options = Data[offset++];
            use_weight = (options & 1);
        }
        
        // Create weight tensor if needed
        torch::Tensor weight;
        if (use_weight && offset < Size) {
            weight = fuzzer_utils::createTensor(Data, Size, offset);
        }
        
        // Apply RMSNorm using functional API
        torch::Tensor output = torch::nn::functional::rms_norm(
            input, 
            normalized_shape, 
            use_weight ? torch::optional<torch::Tensor>(weight) : torch::nullopt,
            epsilon
        );
        
        // Ensure computation is performed
        output.sum().item<float>();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}