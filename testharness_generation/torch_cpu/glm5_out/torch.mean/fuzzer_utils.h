
#ifndef FUZZER_UTILS_H
#define FUZZER_UTILS_H

#include <torch/torch.h>
#include <cstdint>
#include <vector>
#include <string>
#include <stdexcept>

// Macro definitions (Consider moving to CMake/build system defines if possible)
#define MIN_RANK 0
#define MAX_RANK 4
#define MAX_TENSOR_SHAPE_DIMS 16
#define MIN_TENSOR_SHAPE_DIMS 0
// Define USE_RANDOM_TENSOR and USE_GPU via build system flags (e.g., -DUSE_RANDOM_TENSOR=1)
// #define USE_RANDOM_TENSOR 0 // Default to input-based
// #define USE_GPU 0           // Default to CPU
#define DEBUG_FUZZ 1 // Keep for debugging convenience

namespace fuzzer_utils
{

    // --- Logging ---
    void logErrorMessage(const std::string &msg);
    void saveErrorInput(const uint8_t *data, size_t size);
    void saveDiffInput(const uint8_t *data, size_t size, const std::string &timestamp);
    std::string currentTimestamp();
    std::string sanitizedTimestamp();
    bool ensure_log_directory_exists(const std::string &dir = ".");

    // --- Tensor Parsing ---
    torch::ScalarType parseDataType(uint8_t selector);
    uint8_t parseRank(uint8_t byte);
    std::vector<int64_t> parseShape(const uint8_t *data, size_t &offset, size_t size, uint8_t rank);
    std::vector<uint8_t> parseTensorData(const uint8_t *data, size_t &offset, size_t size,
                                         int64_t numElements, size_t dtypeSize);
    torch::Tensor createTensor(const uint8_t *Data, size_t Size, size_t &offset);

    // --- Comparison  ---
    void compareTensors(const torch::Tensor &t1, const torch::Tensor &t2, const uint8_t *data, size_t size, double rtol = 1e-5, double atol = 1e-8);

} // namespace fuzzer_utils

#endif // FUZZER_UTILS_H
