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
        
        // Need at least a few bytes to create meaningful tensors
        if (Size < 4) {
            return 0;
        }
        
        // Parse number of tensors to create (1-10)
        uint8_t num_tensors = (Data[offset++] % 10) + 1;
        if (offset >= Size) {
            return 0;
        }
        
        // Parse norm type
        double norm_type = 2.0; // Default to L2 norm
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&norm_type, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        // Create a vector of tensors
        std::vector<torch::Tensor> parameters;
        
        // Create tensors with various shapes and types
        for (uint8_t i = 0; i < num_tensors && offset < Size; ++i) {
            try {
                torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
                parameters.push_back(tensor);
            } catch (const std::exception& e) {
                // If one tensor creation fails, continue with the next
                continue;
            }
        }
        
        // Skip if no valid tensors were created
        if (parameters.empty()) {
            return 0;
        }
        
        // Apply get_total_norm operation
        torch::Tensor total_norm;
        
        // Test with different norm types
        if (offset < Size) {
            uint8_t norm_selector = Data[offset++];
            
            // Choose between different norm types
            if (norm_selector % 3 == 0) {
                // Use default L2 norm (norm_type = 2.0)
                total_norm = torch::nn::utils::clip_grad_norm_(parameters, std::numeric_limits<double>::infinity());
            } else if (norm_selector % 3 == 1) {
                // Use custom norm_type
                total_norm = torch::nn::utils::clip_grad_norm_(parameters, std::numeric_limits<double>::infinity(), norm_type);
            } else {
                // Use infinity norm
                total_norm = torch::nn::utils::clip_grad_norm_(parameters, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity());
            }
        } else {
            // Default case
            total_norm = torch::nn::utils::clip_grad_norm_(parameters, std::numeric_limits<double>::infinity());
        }
        
        // Access the result to ensure computation is performed
        float norm_value = total_norm.item<float>();
        
        // Test the clip_grad_norm_ function with actual clipping
        if (!parameters.empty() && offset < Size) {
            double max_norm = std::abs(*reinterpret_cast<const double*>(Data + offset % (Size - sizeof(double))));
            torch::Tensor clip_result = torch::nn::utils::clip_grad_norm_(parameters, max_norm, norm_type);
            float clip_value = clip_result.item<float>();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
