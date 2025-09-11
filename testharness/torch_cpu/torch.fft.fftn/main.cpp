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
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse FFT parameters from the remaining data
        std::vector<int64_t> s;
        std::vector<int64_t> dim;
        std::string norm = "backward";
        
        // Parse 's' parameter (shape of output)
        if (offset + 1 < Size) {
            uint8_t s_rank = Data[offset++] % 4; // Limit to reasonable rank
            
            for (uint8_t i = 0; i < s_rank && offset + sizeof(int64_t) <= Size; i++) {
                int64_t dim_size;
                std::memcpy(&dim_size, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                s.push_back(dim_size);
            }
        }
        
        // Parse 'dim' parameter (dimensions to transform)
        if (offset + 1 < Size) {
            uint8_t dim_count = Data[offset++] % (input_tensor.dim() + 1);
            
            for (uint8_t i = 0; i < dim_count && offset + sizeof(int64_t) <= Size; i++) {
                int64_t dim_val;
                std::memcpy(&dim_val, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                dim.push_back(dim_val);
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
        
        // Apply FFT operation with different parameter combinations
        torch::Tensor result;
        
        // Case 1: Basic fftn call
        result = torch::fft::fftn(input_tensor);
        
        // Case 2: With s parameter
        if (!s.empty()) {
            result = torch::fft::fftn(input_tensor, s);
        }
        
        // Case 3: With dim parameter
        if (!dim.empty()) {
            result = torch::fft::fftn(input_tensor, c10::nullopt, dim);
        }
        
        // Case 4: With norm parameter
        result = torch::fft::fftn(input_tensor, c10::nullopt, c10::nullopt, norm);
        
        // Case 5: With all parameters
        if (!s.empty() && !dim.empty()) {
            result = torch::fft::fftn(input_tensor, s, dim, norm);
        }
        
        // Ensure the result is used to prevent optimization
        if (result.defined()) {
            volatile float sum = result.sum().item<float>();
            (void)sum;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
