#include "fuzzer_utils.h"
#include <iostream>
#include <limits>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  try {
    if (Size < 3) return 0;
    size_t offset = 0;

    // Create two input tensors with varied shapes/dtypes
    torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
    torch::Tensor other = fuzzer_utils::createTensor(Data, Size, offset);

    // Parse alpha (scalar multiplier)
    float alpha = 1.0f;
    if (offset < Size) {
      uint8_t alpha_byte = Data[offset++];
      // Map to range [-10.0, 10.0] with special values
      if (alpha_byte == 0) alpha = 0.0f;
      else if (alpha_byte == 255) alpha = std::numeric_limits<float>::infinity();
      else if (alpha_byte == 254) alpha = -std::numeric_limits<float>::infinity();
      else if (alpha_byte == 253) alpha = std::numeric_limits<float>::quiet_NaN();
      else {
        float normalized = static_cast<float>(alpha_byte) / 127.5f - 1.0f;
        alpha = normalized * 10.0f;
      }
    }

    // Test with out= parameter (optional tensor)
    bool use_out = false;
    if (offset < Size) {
      use_out = (Data[offset++] % 2 == 0);
    }

    // Execute the add operation
    if (use_out) {
      // Pre-allocate output tensor and use add_out
      torch::Tensor out = torch::empty_like(input);
      torch::add_out(out, input, other, alpha);
    } else {
      torch::Tensor result = torch::add(input, other, alpha);
      (void)result; // Suppress unused variable warning
    }

  } catch (const c10::Error &e) {
    // Catch PyTorch-specific errors (broadcasting, type promotion, etc.)
    std::cerr << "c10::Error: " << e.what_without_backtrace() << std::endl;
    return 0; // Keep input for potential further exploration
  } catch (const std::exception &e) {
    std::cerr << "Exception caught: " << e.what() << std::endl;
    return -1; // Discard input
  }
  return 0;
}