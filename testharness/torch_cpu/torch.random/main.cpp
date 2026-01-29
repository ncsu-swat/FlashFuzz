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
        
        // Need at least 4 bytes for basic parameters
        if (Size < 4) {
            return 0;
        }
        
        // Parse parameters for torch::random
        int64_t from = 0;
        int64_t to = 100;
        
        // Extract 'from' parameter if we have enough data
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&from, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            // Clamp to reasonable range to avoid overflow issues
            from = from % 10000;
        }
        
        // Extract 'to' parameter if we have enough data
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&to, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            // Clamp to reasonable range
            to = to % 10000;
        }
        
        // Ensure to > from
        if (to <= from) {
            to = from + 100;
        }
        
        // Create a tensor to hold the shape for the random tensor
        torch::Tensor shape_tensor;
        if (offset < Size) {
            shape_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        }
        
        // Test different variants of torch::random_ (in-place method on tensors)
        
        // 1. Create a tensor first, then apply random_() - fills with uniform random values
        torch::Tensor tensor1 = torch::empty({3, 3});
        tensor1.random_();
        
        // 2. Random with from and to parameters on existing tensor
        // random_(from, to) fills with values in [from, to)
        try {
            torch::Tensor tensor2 = torch::empty({2, 2}, torch::kInt64);
            tensor2.random_(from, to);
        } catch (...) {
            // Silently handle invalid from/to combinations
        }
        
        // 3. Random with shape tensor if available
        if (shape_tensor.defined() && shape_tensor.numel() > 0) {
            try {
                // random_() works on integer and floating point tensors
                if (shape_tensor.is_floating_point()) {
                    torch::Tensor result3 = shape_tensor.clone();
                    result3.random_();
                } else {
                    // For non-floating point, create a float clone
                    torch::Tensor result3 = shape_tensor.to(torch::kFloat32);
                    result3.random_();
                }
            } catch (...) {
                // Silently handle failures
            }
            
            try {
                // Try random with from and to on integer tensor
                torch::Tensor result4 = torch::empty_like(shape_tensor, torch::kInt64);
                result4.random_(from, to);
            } catch (...) {
                // Silently handle failures
            }
        }
        
        // 4. Random with explicit shape vector
        std::vector<int64_t> explicit_shape;
        
        // Extract up to 4 dimensions from remaining data
        for (int i = 0; i < 4 && offset + sizeof(int64_t) <= Size; i++) {
            int64_t dim;
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Use absolute value to avoid negative dimensions, keep reasonable
            explicit_shape.push_back(std::abs(dim) % 10 + 1);
        }
        
        if (!explicit_shape.empty()) {
            // Create tensor with explicit shape and apply random
            torch::Tensor result6 = torch::empty(explicit_shape);
            result6.random_();
            
            // Random with from and to on integer tensor
            try {
                torch::Tensor result7 = torch::empty(explicit_shape, torch::kInt64);
                result7.random_(from, to);
            } catch (...) {
                // Silently handle failures
            }
        }
        
        // 5. Random with generator
        torch::Generator gen = torch::make_generator<torch::CPUGeneratorImpl>();
        
        // Set seed if we have more data
        if (offset + sizeof(uint64_t) <= Size) {
            uint64_t seed;
            std::memcpy(&seed, Data + offset, sizeof(uint64_t));
            offset += sizeof(uint64_t);
            gen.set_current_seed(seed);
        }
        
        // Random with from, to, and generator (the proper signature)
        try {
            torch::Tensor result9 = torch::empty({2, 2}, torch::kInt64);
            result9.random_(from, to, gen);
        } catch (...) {
            // Silently handle failures
        }
        
        // Random with shape vector, from, to, and generator
        if (!explicit_shape.empty()) {
            try {
                torch::Tensor result11 = torch::empty(explicit_shape, torch::kInt64);
                result11.random_(from, to, gen);
            } catch (...) {
                // Silently handle failures
            }
        }
        
        // 6. Random with different dtypes
        std::vector<torch::ScalarType> dtypes = {
            torch::kFloat32,
            torch::kFloat64,
            torch::kInt32,
            torch::kInt64,
            torch::kInt16,
            torch::kInt8
        };
        
        // Select dtype based on remaining data
        size_t dtype_idx = 0;
        if (offset < Size) {
            dtype_idx = Data[offset++] % dtypes.size();
        }
        
        torch::ScalarType selected_dtype = dtypes[dtype_idx];
        torch::TensorOptions options = torch::TensorOptions().dtype(selected_dtype);
        
        // Random with options
        torch::Tensor result12 = torch::empty({3, 3}, options);
        result12.random_();
        
        // Random with from, to, and options (only for integer types)
        if (!at::isFloatingType(selected_dtype)) {
            try {
                torch::Tensor result13 = torch::empty({3, 3}, options);
                result13.random_(from, to);
            } catch (...) {
                // Silently handle failures
            }
        }
        
        // Random with shape vector and options
        if (!explicit_shape.empty()) {
            torch::Tensor result14 = torch::empty(explicit_shape, options);
            result14.random_();
            
            // Random with from, to, and options (only for integer types)
            if (!at::isFloatingType(selected_dtype)) {
                try {
                    torch::Tensor result15 = torch::empty(explicit_shape, options);
                    result15.random_(from, to);
                } catch (...) {
                    // Silently handle failures
                }
            }
        }
        
        // 7. Test edge cases for random range
        try {
            torch::Tensor edge1 = torch::empty({2, 2}, torch::kInt64);
            edge1.random_(0, 1);  // Single value range
        } catch (...) {
            // Silently handle failures
        }
        
        try {
            torch::Tensor edge2 = torch::empty({2, 2}, torch::kInt64);
            edge2.random_(-100, 100);  // Negative to positive range
        } catch (...) {
            // Silently handle failures
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0;  // Keep the input
}