#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstring>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Minimum size check - we need at least some bytes for basic tensor creation
    if (Size < 4) {
        return 0; // Too small to do anything meaningful, but keep it
    }

    try
    {
        size_t offset = 0;
        
        // Create Tanhshrink module
        torch::nn::Tanhshrink tanhshrink_module;
        
        // Test 1: Basic tensor with fuzzer-controlled properties
        try {
            auto tensor1 = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Apply Tanhshrink
            auto result1 = tanhshrink_module->forward(tensor1);
            
            // Verify output shape matches input
            if (result1.sizes() != tensor1.sizes()) {
                std::cerr << "Shape mismatch after Tanhshrink!" << std::endl;
            }
            
            // Test in-place operation if we have enough data
            if (offset < Size && Data[offset++] % 2 == 0) {
                auto tensor1_copy = tensor1.clone();
                tensor1_copy = tanhshrink_module->forward(tensor1_copy);
            }
        } catch (const c10::Error& e) {
            // PyTorch-specific errors - continue fuzzing
        } catch (const std::runtime_error& e) {
            // Tensor creation errors - continue fuzzing
        }
        
        // Test 2: Multiple tensors with different properties if we have more data
        if (offset + 4 < Size) {
            try {
                auto tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Test with requires_grad if byte indicates
                if (offset < Size && Data[offset++] % 3 == 0) {
                    if (tensor2.dtype() == torch::kFloat || tensor2.dtype() == torch::kDouble ||
                        tensor2.dtype() == torch::kHalf || tensor2.dtype() == torch::kBFloat16) {
                        tensor2.requires_grad_(true);
                    }
                }
                
                auto result2 = tanhshrink_module->forward(tensor2);
                
                // Test backward pass if gradient is enabled
                if (tensor2.requires_grad()) {
                    try {
                        auto grad_output = torch::ones_like(result2);
                        result2.backward(grad_output);
                    } catch (const c10::Error& e) {
                        // Backward pass errors - continue
                    }
                }
            } catch (const c10::Error& e) {
                // Continue fuzzing
            } catch (const std::runtime_error& e) {
                // Continue fuzzing
            }
        }
        
        // Test 3: Edge cases with special values
        if (offset + 2 < Size) {
            try {
                uint8_t edge_case_selector = (offset < Size) ? Data[offset++] : 0;
                
                // Create tensors with special values based on selector
                torch::Tensor edge_tensor;
                auto dtype = (Data[offset % Size] % 2 == 0) ? torch::kFloat : torch::kDouble;
                
                switch (edge_case_selector % 8) {
                    case 0: // Zeros
                        edge_tensor = torch::zeros({2, 3}, torch::TensorOptions().dtype(dtype));
                        break;
                    case 1: // Ones
                        edge_tensor = torch::ones({3, 2}, torch::TensorOptions().dtype(dtype));
                        break;
                    case 2: // Negative values
                        edge_tensor = torch::full({4, 4}, -1.5, torch::TensorOptions().dtype(dtype));
                        break;
                    case 3: // Large values
                        edge_tensor = torch::full({2, 2}, 1e6, torch::TensorOptions().dtype(dtype));
                        break;
                    case 4: // Small values
                        edge_tensor = torch::full({3, 3}, 1e-6, torch::TensorOptions().dtype(dtype));
                        break;
                    case 5: // NaN values (if floating point)
                        edge_tensor = torch::full({2, 2}, std::numeric_limits<float>::quiet_NaN(), 
                                                torch::TensorOptions().dtype(dtype));
                        break;
                    case 6: // Infinity
                        edge_tensor = torch::full({2, 2}, std::numeric_limits<float>::infinity(), 
                                                torch::TensorOptions().dtype(dtype));
                        break;
                    case 7: // Mixed values
                        edge_tensor = torch::randn({3, 3}, torch::TensorOptions().dtype(dtype));
                        break;
                }
                
                auto edge_result = tanhshrink_module->forward(edge_tensor);
                
                // Verify mathematical property: Tanhshrink(x) = x - tanh(x)
                auto expected = edge_tensor - torch::tanh(edge_tensor);
                if (!torch::allclose(edge_result, expected, 1e-5, 1e-8)) {
                    // Allow some tolerance for numerical precision
                    auto max_diff = torch::max(torch::abs(edge_result - expected)).item<double>();
                    if (max_diff > 1e-3) {
                        std::cerr << "Large deviation from expected Tanhshrink formula: " << max_diff << std::endl;
                    }
                }
                
            } catch (const c10::Error& e) {
                // Continue fuzzing
            } catch (const std::exception& e) {
                // Continue fuzzing
            }
        }
        
        // Test 4: Non-contiguous tensors
        if (offset + 4 < Size) {
            try {
                auto base_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Make non-contiguous by transposing or slicing
                if (base_tensor.dim() >= 2) {
                    auto non_contiguous = base_tensor.transpose(0, 1);
                    auto result_nc = tanhshrink_module->forward(non_contiguous);
                    
                    // Result should have same shape as input
                    if (result_nc.sizes() != non_contiguous.sizes()) {
                        std::cerr << "Shape mismatch for non-contiguous tensor!" << std::endl;
                    }
                }
                
                // Test with sliced tensor
                if (base_tensor.size(0) > 1) {
                    auto sliced = base_tensor.slice(0, 0, base_tensor.size(0), 2); // stride of 2
                    if (sliced.numel() > 0) {
                        auto result_sliced = tanhshrink_module->forward(sliced);
                    }
                }
            } catch (const c10::Error& e) {
                // Continue fuzzing
            } catch (const std::runtime_error& e) {
                // Continue fuzzing
            }
        }
        
        // Test 5: Batch processing
        if (offset + 8 < Size) {
            try {
                // Create a batch of tensors
                uint8_t batch_size = (offset < Size) ? (Data[offset++] % 8 + 1) : 4;
                std::vector<torch::Tensor> batch;
                
                for (int i = 0; i < batch_size && offset < Size; ++i) {
                    try {
                        auto t = fuzzer_utils::createTensor(Data, Size, offset);
                        if (t.numel() > 0) {
                            batch.push_back(t);
                        }
                    } catch (...) {
                        break; // Stop creating batch on error
                    }
                }
                
                // Process batch
                for (auto& t : batch) {
                    auto result = tanhshrink_module->forward(t);
                }
                
                // Try stacking if all tensors have same shape
                if (batch.size() > 1) {
                    bool same_shape = true;
                    for (size_t i = 1; i < batch.size(); ++i) {
                        if (batch[i].sizes() != batch[0].sizes()) {
                            same_shape = false;
                            break;
                        }
                    }
                    
                    if (same_shape) {
                        auto stacked = torch::stack(batch);
                        auto batch_result = tanhshrink_module->forward(stacked);
                    }
                }
            } catch (const c10::Error& e) {
                // Continue fuzzing
            } catch (const std::exception& e) {
                // Continue fuzzing
            }
        }
        
        // Test 6: Device transfers (if CUDA available)
        #ifdef USE_GPU
        if (torch::cuda::is_available() && offset < Size && Data[offset++] % 4 == 0) {
            try {
                auto cpu_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                auto cuda_tensor = cpu_tensor.to(torch::kCUDA);
                auto cuda_result = tanhshrink_module->forward(cuda_tensor);
                auto cpu_result = cuda_result.to(torch::kCPU);
                
                // Compare CPU and GPU results
                auto cpu_direct = tanhshrink_module->forward(cpu_tensor);
                if (!torch::allclose(cpu_result, cpu_direct, 1e-5, 1e-8)) {
                    std::cerr << "CPU/GPU result mismatch!" << std::endl;
                }
            } catch (const c10::Error& e) {
                // CUDA errors - continue
            } catch (const std::exception& e) {
                // Continue fuzzing
            }
        }
        #endif
        
    }
    catch (const std::bad_alloc& e)
    {
        // Memory allocation failure - might want to track but continue fuzzing
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1; // Unexpected exception - discard input
    }
    
    return 0; // Successfully processed input
}