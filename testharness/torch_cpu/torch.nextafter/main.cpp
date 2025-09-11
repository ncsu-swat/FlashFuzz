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
        
        // Create first tensor (x)
        torch::Tensor x = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create second tensor (other)
        torch::Tensor other;
        if (offset < Size) {
            other = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If we don't have enough data for a second tensor, use the same tensor
            other = x;
        }
        
        // Try different variants of nextafter
        
        // 1. Basic nextafter
        torch::Tensor result1 = torch::nextafter(x, other);
        
        // 2. Out variant
        torch::Tensor out = torch::empty_like(x);
        torch::nextafter_out(out, x, other);
        
        // 3. In-place variant (if supported)
        if (x.is_floating_point()) {
            torch::Tensor x_copy = x.clone();
            x_copy.nextafter_(other);
        }
        
        // 4. Try scalar variants
        if (x.numel() > 0) {
            // Extract a scalar value from the tensor
            double scalar_value = 0.0;
            if (x.is_floating_point()) {
                scalar_value = x.item<double>();
            } else if (x.is_complex()) {
                scalar_value = x.item<c10::complex<double>>().real();
            } else if (x.dtype().isIntegralType(false)) {
                scalar_value = static_cast<double>(x.item<int64_t>());
            }
            
            // Test nextafter with scalar as tensor
            torch::Tensor scalar_tensor = torch::tensor(scalar_value);
            torch::Tensor result_scalar = torch::nextafter(x, scalar_tensor);
            
            // Test nextafter with scalar as first argument
            if (other.numel() > 0 && other.dim() == 0) {
                torch::Tensor result_scalar_first = torch::nextafter(scalar_tensor, other);
            }
        }
        
        // 5. Try with different dtypes
        if (x.is_floating_point() && offset + 1 < Size) {
            uint8_t dtype_selector = Data[offset++];
            auto dtype = fuzzer_utils::parseDataType(dtype_selector);
            
            // Only proceed if the target dtype is floating point
            if (dtype == torch::kFloat || dtype == torch::kDouble || 
                dtype == torch::kHalf || dtype == torch::kBFloat16) {
                torch::Tensor x_cast = x.to(dtype);
                torch::Tensor other_cast = other.to(dtype);
                torch::Tensor result_cast = torch::nextafter(x_cast, other_cast);
            }
        }
        
        // 6. Try with empty tensors
        torch::Tensor empty_tensor = torch::empty({0});
        if (x.numel() > 0) {
            try {
                torch::Tensor result_empty = torch::nextafter(x, empty_tensor);
            } catch (...) {
                // Expected to fail in some cases
            }
            
            try {
                torch::Tensor result_empty2 = torch::nextafter(empty_tensor, x);
            } catch (...) {
                // Expected to fail in some cases
            }
        }
        
        // 7. Try with tensors of different shapes
        if (offset + 2 < Size) {
            uint8_t rank_byte = Data[offset++];
            uint8_t rank = fuzzer_utils::parseRank(rank_byte);
            std::vector<int64_t> new_shape = fuzzer_utils::parseShape(Data, offset, Size, rank);
            
            if (!new_shape.empty()) {
                try {
                    torch::Tensor reshaped = x.reshape(new_shape);
                    torch::Tensor result_reshaped = torch::nextafter(reshaped, other);
                } catch (...) {
                    // Reshape might fail, that's expected
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
