#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
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
        }
        
        // Extract 'to' parameter if we have enough data
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&to, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
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
        
        // Test different variants of torch::random
        
        // 1. Create a tensor first, then apply random
        torch::Tensor tensor1 = torch::empty({3, 3});
        tensor1.random_();
        
        // 2. Random with from and to parameters on existing tensor
        torch::Tensor tensor2 = torch::empty({2, 2});
        tensor2.random_(from, to);
        
        // 3. Random with shape tensor if available
        if (shape_tensor.defined()) {
            try {
                // Apply random to the shape tensor itself
                torch::Tensor result3 = shape_tensor.clone();
                result3.random_();
            } catch (...) {
                // If that fails, try to create a new tensor with similar shape
                if (shape_tensor.dim() > 0 && shape_tensor.numel() > 0) {
                    // Create tensor with same shape as shape_tensor
                    torch::Tensor result4 = torch::empty_like(shape_tensor);
                    result4.random_();
                    
                    // Try random with from and to
                    torch::Tensor result5 = torch::empty_like(shape_tensor);
                    result5.random_(from, to);
                }
            }
        }
        
        // 4. Random with explicit shape vector
        std::vector<int64_t> explicit_shape;
        
        // Extract up to 4 dimensions from remaining data
        for (int i = 0; i < 4 && offset + sizeof(int64_t) <= Size; i++) {
            int64_t dim;
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Use absolute value to avoid negative dimensions
            explicit_shape.push_back(std::abs(dim) % 10 + 1); // Keep dimensions reasonable and positive
        }
        
        if (!explicit_shape.empty()) {
            // Create tensor with explicit shape and apply random
            torch::Tensor result6 = torch::empty(explicit_shape);
            result6.random_();
            
            // Random with from and to
            torch::Tensor result7 = torch::empty(explicit_shape);
            result7.random_(from, to);
        }
        
        // 5. Random with generator
        torch::Generator gen = torch::default_generator();
        
        // Set seed if we have more data
        if (offset + sizeof(uint64_t) <= Size) {
            uint64_t seed;
            std::memcpy(&seed, Data + offset, sizeof(uint64_t));
            offset += sizeof(uint64_t);
            gen.set_current_seed(seed);
        }
        
        // Random with generator
        torch::Tensor result8 = torch::empty({2, 2});
        result8.random_(gen);
        
        // Random with from, to, and generator
        torch::Tensor result9 = torch::empty({2, 2});
        result9.random_(from, to, gen);
        
        // Random with shape vector and generator
        if (!explicit_shape.empty()) {
            torch::Tensor result10 = torch::empty(explicit_shape);
            result10.random_(gen);
            
            // Random with from, to, and generator
            torch::Tensor result11 = torch::empty(explicit_shape);
            result11.random_(from, to, gen);
        }
        
        // 6. Random with options
        torch::TensorOptions options = torch::TensorOptions();
        
        // Set dtype based on remaining data
        if (offset < Size) {
            options = options.dtype(fuzzer_utils::parseDataType(Data[offset++]));
        }
        
        // Random with options
        torch::Tensor result12 = torch::empty({3, 3}, options);
        result12.random_();
        
        // Random with from, to, and options
        torch::Tensor result13 = torch::empty({3, 3}, options);
        result13.random_(from, to);
        
        // Random with shape vector and options
        if (!explicit_shape.empty()) {
            torch::Tensor result14 = torch::empty(explicit_shape, options);
            result14.random_();
            
            // Random with from, to, and options
            torch::Tensor result15 = torch::empty(explicit_shape, options);
            result15.random_(from, to);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}