#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <vector>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create meaningful input
        if (Size < 4) {
            return 0;
        }
        
        // Create a tensor from the input data
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Build options from remaining bytes
        torch::TensorOptions options;
        
        if (offset + 1 < Size) {
            uint8_t dtype_selector = Data[offset++];
            options = options.dtype(fuzzer_utils::parseDataType(dtype_selector));
        }
        
        if (offset + 1 < Size) {
            uint8_t requires_grad = Data[offset++] % 2;
            // requires_grad only valid for floating point types
            if (options.dtype() == torch::kFloat32 || 
                options.dtype() == torch::kFloat64 ||
                options.dtype() == torch::kFloat16) {
                options = options.requires_grad(requires_grad == 1);
            }
        }
        
        // Test 1: Create tensor from scalar double
        {
            double scalar_val = 3.14;
            if (offset + sizeof(double) <= Size) {
                memcpy(&scalar_val, Data + offset, sizeof(double));
                offset += sizeof(double);
                // Ensure it's a valid finite number
                if (!std::isfinite(scalar_val)) {
                    scalar_val = 0.0;
                }
            }
            torch::Tensor scalar_tensor = torch::tensor(scalar_val, options);
            (void)scalar_tensor.numel(); // Force evaluation
        }
        
        // Test 2: Create tensor from int scalar
        {
            int int_val = 42;
            if (offset + sizeof(int) <= Size) {
                memcpy(&int_val, Data + offset, sizeof(int));
                offset += sizeof(int);
            }
            torch::Tensor int_tensor = torch::tensor(int_val);
            (void)int_tensor.numel();
        }
        
        // Test 3: Create tensor from vector of floats
        {
            size_t vec_size = 1;
            if (offset + 1 < Size) {
                vec_size = (Data[offset++] % 16) + 1; // 1-16 elements
            }
            std::vector<float> vec_data;
            for (size_t i = 0; i < vec_size && offset + sizeof(float) <= Size; i++) {
                float val;
                memcpy(&val, Data + offset, sizeof(float));
                offset += sizeof(float);
                if (!std::isfinite(val)) {
                    val = 0.0f;
                }
                vec_data.push_back(val);
            }
            if (vec_data.empty()) {
                vec_data.push_back(1.0f);
            }
            torch::Tensor vec_tensor = torch::tensor(vec_data, options);
            (void)vec_tensor.numel();
        }
        
        // Test 4: Create tensor from vector of ints
        {
            std::vector<int64_t> int_vec;
            size_t int_vec_size = 1;
            if (offset + 1 < Size) {
                int_vec_size = (Data[offset++] % 8) + 1;
            }
            for (size_t i = 0; i < int_vec_size && offset + sizeof(int64_t) <= Size; i++) {
                int64_t val;
                memcpy(&val, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                int_vec.push_back(val);
            }
            if (int_vec.empty()) {
                int_vec.push_back(1);
            }
            torch::Tensor int_vec_tensor = torch::tensor(int_vec);
            (void)int_vec_tensor.numel();
        }
        
        // Test 5: Create tensor from nested initializer list (2D)
        {
            torch::Tensor nested_tensor = torch::tensor({{1.0, 2.0}, {3.0, 4.0}});
            (void)nested_tensor.numel();
        }
        
        // Test 6: Create tensor from boolean
        {
            bool bool_val = false;
            if (offset + 1 < Size) {
                bool_val = (Data[offset++] % 2) == 1;
            }
            torch::Tensor bool_tensor = torch::tensor(bool_val);
            (void)bool_tensor.numel();
        }
        
        // Test 7: Empty vector
        {
            std::vector<float> empty_vec;
            torch::Tensor empty_tensor = torch::tensor(empty_vec);
            (void)empty_tensor.numel();
        }
        
        // Test 8: Create tensor from input_tensor's item (if scalar)
        if (input_tensor.numel() == 1) {
            try {
                torch::Scalar scalar = input_tensor.item();
                torch::Tensor from_item = torch::tensor(scalar.toDouble());
                (void)from_item.numel();
            } catch (const std::exception&) {
                // item() can fail for certain dtypes, silently ignore
            }
        }
        
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}