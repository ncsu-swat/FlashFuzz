#include "fuzzer_utils.h"    // General fuzzing utilities
#include <ATen/ExpandUtils.h> // For at::infer_size
#include <iostream>          // For cerr, cout

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
        
        // Determine number of shapes to broadcast (need at least 1 byte)
        if (Size < 1) return 0;
        uint8_t num_shapes = (Data[offset++] % 5) + 1; // 1 to 5 shapes
        
        // Create vector to hold shapes
        std::vector<std::vector<int64_t>> shapes;
        
        // Parse shapes
        for (uint8_t i = 0; i < num_shapes && offset < Size; ++i) {
            // Get rank for this shape
            if (offset >= Size) break;
            uint8_t rank_byte = Data[offset++];
            uint8_t rank = fuzzer_utils::parseRank(rank_byte);
            
            // Parse shape dimensions
            auto shape = fuzzer_utils::parseShape(Data, offset, Size, rank);
            shapes.push_back(shape);
        }
        
        // Need at least one shape
        if (shapes.empty()) {
            return 0;
        }
        
        // Call broadcast_shapes using at::infer_size
        try {
            // Convert vector of shapes to vector of IntArrayRef
            std::vector<c10::IntArrayRef> shape_refs;
            for (const auto& shape : shapes) {
                shape_refs.push_back(c10::IntArrayRef(shape));
            }
            
            // Start with first shape as the result
            std::vector<int64_t> result = shapes[0];
            
            // Iteratively broadcast with each subsequent shape
            for (size_t i = 1; i < shape_refs.size(); ++i) {
                result = at::infer_size(result, shape_refs[i]);
            }
            
            // Verify result by creating tensors with the shapes and checking if they broadcast
            std::vector<torch::Tensor> tensors;
            for (const auto& shape : shapes) {
                tensors.push_back(torch::ones(shape, torch::kFloat32));
            }
            
            // Try to broadcast the tensors
            if (tensors.size() >= 2) {
                torch::Tensor broadcasted = tensors[0];
                for (size_t i = 1; i < tensors.size(); ++i) {
                    broadcasted = broadcasted + tensors[i];
                }
                
                // Check if the result shape matches what broadcast_shapes returned
                auto tensor_shape = broadcasted.sizes().vec();
                
                if (tensor_shape != result) {
                    throw std::runtime_error("Broadcast shape mismatch between infer_size and actual tensor broadcast");
                }
            }
            
            // Also test with expand_as to exercise more broadcast code paths
            if (tensors.size() >= 2) {
                auto target = torch::ones(result, torch::kFloat32);
                for (size_t i = 0; i < tensors.size(); ++i) {
                    auto expanded = tensors[i].expand(result);
                    (void)expanded; // Use the result
                }
            }
            
        } catch (const c10::Error& e) {
            // PyTorch-specific exceptions are expected for incompatible shapes
            // Silently ignore as this is expected behavior
        } catch (const std::runtime_error& e) {
            // Shape mismatch would be a real bug - rethrow
            if (std::string(e.what()).find("mismatch") != std::string::npos) {
                throw;
            }
            // Other runtime errors from invalid shapes are expected
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}