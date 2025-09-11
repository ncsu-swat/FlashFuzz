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
        
        // Need at least 2 bytes for the input tensor and 1 byte for n
        if (Size < 3) {
            return 0;
        }
        
        // Create input tensor x
        torch::Tensor x = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse n value (degree of the polynomial)
        int64_t n = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&n, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            // Limit n to a reasonable range to avoid excessive computation
            n = std::abs(n) % 100;
        }
        
        // Apply the shifted_chebyshev_polynomial_w operation
        torch::Tensor result = torch::special::shifted_chebyshev_polynomial_w(x, n);
        
        // Try different n values to increase coverage
        if (offset + 1 < Size) {
            uint8_t alt_n_byte = Data[offset++];
            int64_t alt_n = alt_n_byte % 10; // Use a smaller range for alternative n
            
            torch::Tensor alt_result = torch::special::shifted_chebyshev_polynomial_w(x, alt_n);
        }
        
        // Try with n=0 and n=1 which are special cases
        torch::Tensor result_n0 = torch::special::shifted_chebyshev_polynomial_w(x, 0);
        torch::Tensor result_n1 = torch::special::shifted_chebyshev_polynomial_w(x, 1);
        
        // Try with negative n (should handle according to definition)
        if (offset < Size) {
            int64_t neg_n = -static_cast<int64_t>(Data[offset] % 10) - 1;
            torch::Tensor result_neg_n = torch::special::shifted_chebyshev_polynomial_w(x, neg_n);
        }
        
        // Try with empty tensor if we have enough data
        if (offset < Size) {
            std::vector<int64_t> empty_shape = {0};
            torch::Tensor empty_tensor = torch::empty(empty_shape, x.options());
            torch::Tensor result_empty = torch::special::shifted_chebyshev_polynomial_w(empty_tensor, n);
        }
        
        // Try with scalar tensor
        if (offset < Size) {
            torch::Tensor scalar_tensor = torch::tensor(static_cast<float>(Data[offset]));
            torch::Tensor result_scalar = torch::special::shifted_chebyshev_polynomial_w(scalar_tensor, n);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
