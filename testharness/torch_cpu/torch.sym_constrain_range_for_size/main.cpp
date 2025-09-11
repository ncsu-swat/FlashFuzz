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
        
        // Need at least a few bytes for the tensor and range parameters
        if (Size < 4) {
            return 0;
        }
        
        // Create a tensor to use for the operation
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract min and max values from the remaining data
        int64_t min_val = 0;
        int64_t max_val = 0;
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&min_val, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&max_val, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Ensure max_val is at least min_val to avoid invalid range
        if (max_val < min_val) {
            std::swap(min_val, max_val);
        }
        
        // Get a dimension to constrain
        int64_t dim = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // If tensor has dimensions, make dim valid by taking modulo
            if (tensor.dim() > 0) {
                dim = std::abs(dim) % tensor.dim();
            }
        }
        
        // Get the size of the specified dimension as a scalar
        at::Scalar size_scalar;
        if (tensor.dim() > 0 && dim < tensor.dim()) {
            size_scalar = tensor.size(dim);
        } else {
            size_scalar = at::Scalar(1);
        }
        
        // Apply the sym_constrain_range_for_size operation
        torch::sym_constrain_range_for_size(size_scalar, min_val, max_val);
        
        // Try with negative dimension (should wrap around)
        if (tensor.dim() > 0) {
            int64_t neg_dim = -1;
            at::Scalar neg_size_scalar = tensor.size(neg_dim);
            torch::sym_constrain_range_for_size(neg_size_scalar, min_val, max_val);
        }
        
        // Try with extreme values
        torch::sym_constrain_range_for_size(size_scalar, std::numeric_limits<int64_t>::min(), std::numeric_limits<int64_t>::max());
        
        // Try with equal min and max
        int64_t equal_val = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&equal_val, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        torch::sym_constrain_range_for_size(size_scalar, equal_val, equal_val);
        
        // Try with inverted range (should be handled by the API)
        torch::sym_constrain_range_for_size(size_scalar, max_val, min_val);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
