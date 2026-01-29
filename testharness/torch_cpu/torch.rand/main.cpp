#include "fuzzer_utils.h"
#include <iostream>

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
        
        // Need at least 2 bytes for rank and dimensions
        if (Size < 2) {
            return 0;
        }
        
        // Parse rank from the first byte
        uint8_t rank_byte = Data[offset++];
        uint8_t rank = fuzzer_utils::parseRank(rank_byte);
        
        // Parse shape for the tensor
        std::vector<int64_t> shape = fuzzer_utils::parseShape(Data, offset, Size, rank);
        
        // torch::rand only supports floating point types
        // Select from float types based on fuzzer data
        torch::ScalarType dtype = torch::kFloat;
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset++] % 4;
            switch (dtype_selector) {
                case 0: dtype = torch::kFloat; break;
                case 1: dtype = torch::kDouble; break;
                case 2: dtype = torch::kHalf; break;
                case 3: dtype = torch::kBFloat16; break;
            }
        }
        
        // Create options with the parsed dtype
        auto options = torch::TensorOptions().dtype(dtype);
        
        // Test torch.rand with the parsed shape and options
        // Inner try-catch for expected failures (invalid shapes, etc.)
        try {
            torch::Tensor rand_tensor = torch::rand(shape, options);
            
            // Access the tensor to ensure it's computed
            if (rand_tensor.numel() > 0) {
                // For half/bfloat16, convert to float for range checking
                torch::Tensor check_tensor = rand_tensor;
                if (dtype == torch::kHalf || dtype == torch::kBFloat16) {
                    check_tensor = rand_tensor.to(torch::kFloat);
                }
                
                // Verify values are in range [0, 1) - just access, don't throw
                auto min_val = check_tensor.min().item<float>();
                auto max_val = check_tensor.max().item<float>();
                (void)min_val;
                (void)max_val;
            }
        } catch (const std::exception& e) {
            // Expected for invalid shapes (negative dims, etc.)
        }
        
        // Test with different generator seeds if we have more data
        if (offset + 1 < Size) {
            try {
                // Test with explicit generator
                auto gen = torch::make_generator<torch::CPUGeneratorImpl>();
                gen.set_current_seed(Data[offset++]);
                torch::Tensor seeded_tensor = torch::rand(shape, gen, options);
                (void)seeded_tensor;
            } catch (const std::exception& e) {
                // Expected for some configurations
            }
        }
        
        // Test rand_like if we have a valid tensor
        if (offset < Size) {
            try {
                // Create a small reference tensor
                std::vector<int64_t> small_shape;
                for (size_t i = 0; i < shape.size() && i < 3; i++) {
                    small_shape.push_back(std::min(shape[i], (int64_t)10));
                }
                if (small_shape.empty()) {
                    small_shape.push_back(1);
                }
                
                torch::Tensor ref = torch::rand(small_shape, options);
                torch::Tensor like_tensor = torch::rand_like(ref);
                (void)like_tensor;
            } catch (const std::exception& e) {
                // Expected for some dtypes
            }
        }
        
        // Test with specific shapes
        if (offset + 2 < Size) {
            try {
                int64_t dim1 = (Data[offset++] % 64) + 1;
                int64_t dim2 = (Data[offset++] % 64) + 1;
                torch::Tensor matrix = torch::rand({dim1, dim2}, options);
                (void)matrix;
            } catch (const std::exception& e) {
                // Silently handle
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}