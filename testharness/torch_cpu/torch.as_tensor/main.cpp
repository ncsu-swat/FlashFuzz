#include "fuzzer_utils.h"
#include <iostream>
#include <vector>

// Target API: torch.as_tensor

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
        
        // Skip if we don't have enough data
        if (Size < 4) {
            return 0;
        }

        // Test 1: torch::as_tensor from vector<float>
        try {
            size_t vec_size = std::min(static_cast<size_t>(Data[offset++] % 16 + 1), Size - offset);
            std::vector<float> float_vec;
            float_vec.reserve(vec_size);
            for (size_t i = 0; i < vec_size && offset < Size; i++) {
                float_vec.push_back(static_cast<float>(Data[offset++]) / 255.0f);
            }
            if (!float_vec.empty()) {
                torch::Tensor result1 = torch::tensor(float_vec);
                // Verify the tensor was created correctly
                (void)result1.numel();
            }
        } catch (const c10::Error &) {
            // Expected for invalid inputs
        }

        // Test 2: torch::as_tensor from vector<int64_t>
        try {
            if (offset + 4 < Size) {
                size_t vec_size = std::min(static_cast<size_t>(Data[offset++] % 16 + 1), Size - offset);
                std::vector<int64_t> int_vec;
                int_vec.reserve(vec_size);
                for (size_t i = 0; i < vec_size && offset < Size; i++) {
                    int_vec.push_back(static_cast<int64_t>(Data[offset++]));
                }
                if (!int_vec.empty()) {
                    torch::Tensor result2 = torch::tensor(int_vec);
                    (void)result2.numel();
                }
            }
        } catch (const c10::Error &) {
        }

        // Test 3: torch::as_tensor from vector<double> with dtype option
        try {
            if (offset + 4 < Size) {
                size_t vec_size = std::min(static_cast<size_t>(Data[offset++] % 12 + 1), Size - offset);
                std::vector<double> double_vec;
                double_vec.reserve(vec_size);
                for (size_t i = 0; i < vec_size && offset < Size; i++) {
                    double_vec.push_back(static_cast<double>(Data[offset++]));
                }
                if (!double_vec.empty()) {
                    auto dtype = fuzzer_utils::parseDataType(Data[offset++]);
                    torch::Tensor result3 = torch::tensor(double_vec, torch::TensorOptions().dtype(dtype));
                    (void)result3.numel();
                }
            }
        } catch (const c10::Error &) {
        }

        // Test 4: Create tensor from raw data using from_blob (simulates as_tensor behavior)
        try {
            if (offset + 8 < Size) {
                size_t num_elements = std::min(static_cast<size_t>(Data[offset++] % 8 + 1), (Size - offset) / sizeof(float));
                if (num_elements > 0) {
                    // Allocate aligned buffer for from_blob
                    std::vector<float> buffer(num_elements);
                    for (size_t i = 0; i < num_elements && offset + sizeof(float) <= Size; i++) {
                        float val;
                        std::memcpy(&val, Data + offset, sizeof(float));
                        buffer[i] = val;
                        offset += sizeof(float);
                    }
                    // from_blob creates a tensor that shares memory (like as_tensor)
                    torch::Tensor result4 = torch::from_blob(buffer.data(), {static_cast<int64_t>(num_elements)}, torch::kFloat);
                    // Clone to own the data before buffer goes out of scope
                    torch::Tensor owned = result4.clone();
                    (void)owned.sum();
                }
            }
        } catch (const c10::Error &) {
        }

        // Test 5: Scalar to tensor
        try {
            if (offset < Size) {
                int scalar_val = static_cast<int>(Data[offset++]);
                torch::Tensor result5 = torch::tensor(scalar_val);
                (void)result5.item<int>();
            }
        } catch (const c10::Error &) {
        }

        // Test 6: Float scalar with dtype
        try {
            if (offset + 1 < Size) {
                float scalar_float = static_cast<float>(Data[offset++]) / 128.0f;
                auto dtype = fuzzer_utils::parseDataType(Data[offset++]);
                torch::Tensor result6 = torch::tensor(scalar_float, torch::TensorOptions().dtype(dtype));
                (void)result6.numel();
            }
        } catch (const c10::Error &) {
        }

        // Test 7: Empty vector
        try {
            std::vector<float> empty_vec;
            torch::Tensor result7 = torch::tensor(empty_vec);
            (void)result7.numel();
        } catch (const c10::Error &) {
        }

        // Test 8: 2D tensor from flattened data
        try {
            if (offset + 4 < Size) {
                size_t rows = Data[offset++] % 4 + 1;
                size_t cols = Data[offset++] % 4 + 1;
                size_t total = rows * cols;
                
                std::vector<float> flat_data;
                flat_data.reserve(total);
                for (size_t i = 0; i < total && offset < Size; i++) {
                    flat_data.push_back(static_cast<float>(Data[offset++]));
                }
                
                if (flat_data.size() == total) {
                    torch::Tensor flat_tensor = torch::tensor(flat_data);
                    torch::Tensor result8 = flat_tensor.view({static_cast<int64_t>(rows), static_cast<int64_t>(cols)});
                    (void)result8.sizes();
                }
            }
        } catch (const c10::Error &) {
        }

        // Test 9: Boolean tensor
        try {
            if (offset + 2 < Size) {
                size_t vec_size = std::min(static_cast<size_t>(Data[offset++] % 8 + 1), Size - offset);
                std::vector<int64_t> bool_as_int;
                bool_as_int.reserve(vec_size);
                for (size_t i = 0; i < vec_size && offset < Size; i++) {
                    bool_as_int.push_back(Data[offset++] % 2);
                }
                if (!bool_as_int.empty()) {
                    torch::Tensor result9 = torch::tensor(bool_as_int, torch::kBool);
                    (void)result9.numel();
                }
            }
        } catch (const c10::Error &) {
        }

        // Test 10: Complex number simulation via pair of floats
        try {
            if (offset + 4 < Size) {
                size_t num_complex = std::min(static_cast<size_t>(Data[offset++] % 4 + 1), (Size - offset) / 2);
                std::vector<float> real_imag;
                real_imag.reserve(num_complex * 2);
                for (size_t i = 0; i < num_complex * 2 && offset < Size; i++) {
                    real_imag.push_back(static_cast<float>(Data[offset++]) / 255.0f);
                }
                if (real_imag.size() >= 2) {
                    torch::Tensor ri_tensor = torch::tensor(real_imag);
                    torch::Tensor reshaped = ri_tensor.view({-1, 2});
                    torch::Tensor result10 = torch::view_as_complex(reshaped);
                    (void)result10.numel();
                }
            }
        } catch (const c10::Error &) {
        }

        // Test 11: tensor with requires_grad option
        try {
            if (offset + 2 < Size) {
                size_t vec_size = std::min(static_cast<size_t>(Data[offset++] % 8 + 1), Size - offset);
                std::vector<float> grad_vec;
                grad_vec.reserve(vec_size);
                for (size_t i = 0; i < vec_size && offset < Size; i++) {
                    grad_vec.push_back(static_cast<float>(Data[offset++]));
                }
                if (!grad_vec.empty()) {
                    bool requires_grad = (offset < Size) && (Data[offset++] % 2 == 0);
                    torch::Tensor result11 = torch::tensor(grad_vec, torch::TensorOptions().dtype(torch::kFloat).requires_grad(requires_grad));
                    (void)result11.numel();
                }
            }
        } catch (const c10::Error &) {
        }

    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}