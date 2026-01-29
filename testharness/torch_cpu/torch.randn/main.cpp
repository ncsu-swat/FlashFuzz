#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Need at least 2 bytes for basic parameters
        if (Size < 2) {
            return 0;
        }
        
        // Parse number of dimensions
        uint8_t rank_byte = Data[offset++];
        uint8_t rank = fuzzer_utils::parseRank(rank_byte);
        
        // Parse shape for randn
        std::vector<int64_t> shape = fuzzer_utils::parseShape(Data, offset, Size, rank);
        
        // Parse dtype for the tensor - randn only supports floating point types
        torch::ScalarType dtype = torch::kFloat;
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset++];
            // randn only supports float, double, half, bfloat16, complex float/double
            switch (dtype_selector % 4) {
                case 0:
                    dtype = torch::kFloat;
                    break;
                case 1:
                    dtype = torch::kDouble;
                    break;
                case 2:
                    dtype = torch::kFloat16;
                    break;
                case 3:
                    dtype = torch::kBFloat16;
                    break;
            }
        }
        
        // Create options with the selected dtype
        auto options = torch::TensorOptions().dtype(dtype);
        
        // Call torch::randn with the shape and options
        torch::Tensor result = torch::randn(shape, options);
        
        // Test additional operations based on the selector
        if (offset < Size) {
            uint8_t op_selector = Data[offset++];
            
            // Perform different operations based on the selector
            switch (op_selector % 6) {
                case 0:
                    // Test mean and std (need at least one element and float type)
                    if (result.numel() > 0) {
                        try {
                            auto mean_result = result.to(torch::kFloat).mean();
                            auto std_result = result.to(torch::kFloat).std();
                        } catch (...) {
                            // May fail for some configurations
                        }
                    }
                    break;
                case 1:
                    // Test mathematical operations
                    result = result * 2.0;
                    result = result + 1.0;
                    break;
                case 2:
                    // Test reshaping if possible
                    if (result.numel() > 0) {
                        try {
                            result = result.reshape({-1});
                        } catch (...) {
                            // Reshape might fail
                        }
                    }
                    break;
                case 3:
                    {
                        // Test cloning and other operations
                        torch::Tensor cloned = result.clone();
                        cloned = torch::abs(cloned);
                        break;
                    }
                case 4:
                    // Test type conversion
                    try {
                        result = result.to(torch::kFloat);
                    } catch (...) {
                        // Conversion might fail
                    }
                    break;
                case 5:
                    // Test randn_like
                    {
                        torch::Tensor randn_like_result = torch::randn_like(result);
                    }
                    break;
            }
        }
        
        // Test randn with generator if we have more data
        if (offset + 1 < Size) {
            uint64_t seed = static_cast<uint64_t>(Data[offset++]) << 8 | Data[offset++];
            auto gen = torch::make_generator<torch::CPUGeneratorImpl>(seed);
            torch::Tensor seeded_result = torch::randn(shape, gen, options);
        }
        
        // Test randn with different shape configurations
        if (offset < Size) {
            uint8_t shape_config = Data[offset++];
            std::vector<int64_t> test_shape;
            
            switch (shape_config % 5) {
                case 0:
                    // Scalar (0-d tensor)
                    test_shape = {};
                    break;
                case 1:
                    // 1-D tensor
                    test_shape = {static_cast<int64_t>((shape_config % 16) + 1)};
                    break;
                case 2:
                    // 2-D tensor
                    test_shape = {static_cast<int64_t>((shape_config % 8) + 1), 
                                  static_cast<int64_t>((shape_config % 4) + 1)};
                    break;
                case 3:
                    // 3-D tensor
                    test_shape = {2, 3, 4};
                    break;
                case 4:
                    // Empty tensor
                    test_shape = {0};
                    break;
            }
            
            torch::Tensor config_result = torch::randn(test_shape);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}