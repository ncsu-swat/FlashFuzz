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
        
        // Need at least some data to create tensors
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensors for torch.xlogy
        torch::Tensor x = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create second tensor if we have more data
        torch::Tensor y;
        if (offset < Size) {
            y = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If not enough data for second tensor, use a copy of the first
            y = x.clone();
        }
        
        // Apply torch.xlogy operation
        // xlogy(x, y) = x * log(y) if y > 0, 0 if x == 0, nan otherwise
        torch::Tensor result = torch::xlogy(x, y);
        
        // Try scalar version if we have more data
        if (offset < Size) {
            // Extract a scalar from the remaining data
            double scalar_value = 0.0;
            if (offset + sizeof(double) <= Size) {
                std::memcpy(&scalar_value, Data + offset, sizeof(double));
                offset += sizeof(double);
            }
            
            // Try x * log(scalar)
            torch::Tensor result_scalar_y = torch::xlogy(x, scalar_value);
            
            // Try scalar * log(y)
            torch::Tensor result_scalar_x = torch::xlogy(scalar_value, y);
        }
        
        // Try with different tensor shapes if we have more data
        if (offset + 4 < Size) {
            // Create a tensor with different shape
            torch::Tensor z = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Try broadcasting if possible
            try {
                torch::Tensor result_broadcast = torch::xlogy(x, z);
            } catch (const std::exception&) {
                // Broadcasting might fail due to incompatible shapes, which is expected
            }
        }
        
        // Try with special values that might cause issues
        try {
            // Create tensors with special values
            auto options = torch::TensorOptions().dtype(x.dtype());
            
            // Test with zeros in x
            torch::Tensor zeros = torch::zeros_like(x);
            torch::Tensor result_zeros_x = torch::xlogy(zeros, y);
            
            // Test with zeros in y (should produce NaN where x != 0)
            torch::Tensor zeros_y = torch::zeros_like(y);
            torch::Tensor result_zeros_y = torch::xlogy(x, zeros_y);
            
            // Test with negative values in y (should produce NaN)
            torch::Tensor neg_y = -torch::ones_like(y);
            torch::Tensor result_neg_y = torch::xlogy(x, neg_y);
            
            // Test with infinities and NaNs if we have floating point tensors
            if (x.is_floating_point()) {
                torch::Tensor inf_x = torch::full_like(x, std::numeric_limits<float>::infinity());
                torch::Tensor nan_x = torch::full_like(x, std::numeric_limits<float>::quiet_NaN());
                
                torch::Tensor result_inf_x = torch::xlogy(inf_x, y);
                torch::Tensor result_nan_x = torch::xlogy(nan_x, y);
                
                torch::Tensor inf_y = torch::full_like(y, std::numeric_limits<float>::infinity());
                torch::Tensor nan_y = torch::full_like(y, std::numeric_limits<float>::quiet_NaN());
                
                torch::Tensor result_inf_y = torch::xlogy(x, inf_y);
                torch::Tensor result_nan_y = torch::xlogy(x, nan_y);
            }
        } catch (const std::exception&) {
            // Some operations might fail with certain dtypes, which is expected
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
