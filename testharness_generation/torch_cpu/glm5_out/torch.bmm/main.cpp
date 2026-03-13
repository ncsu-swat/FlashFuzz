#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  try {
    size_t offset = 0;

    // Consume a byte to decide if we should test the out_dtype parameter
    bool use_out_dtype = false;
    if (offset < Size) {
      use_out_dtype = (Data[offset++] % 2) == 0;
    }

    // Create the first input tensor (input)
    // bmm expects a 3D tensor (b x n x m)
    torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);

    // Create the second input tensor (mat2)
    // bmm expects a 3D tensor (b x m x p)
    torch::Tensor mat2 = fuzzer_utils::createTensor(Data, Size, offset);

    // Determine if we should provide an output tensor
    bool provide_out = false;
    if (offset < Size) {
      provide_out = (Data[offset++] % 2) == 0;
    }

    torch::Tensor out_tensor;
    if (provide_out) {
      // Try to create a compatible output tensor, or let it be empty/incorrect
      // to test error handling.
      try {
        // Heuristic: if shapes are somewhat compatible, create a matching out tensor.
        // Otherwise, create a random one to test shape mismatches.
        if (input.dim() == 3 && mat2.dim() == 3 && input.size(0) == mat2.size(0) && input.size(2) == mat2.size(1)) {
             out_tensor = torch::empty({input.size(0), input.size(1), mat2.size(2)}, input.options());
        } else {
             // Create a tensor with arbitrary shape to test mismatch handling
             out_tensor = torch::empty({2, 2, 2}, input.options());
        }
      } catch (...) {
        // If we fail to create a valid 'out' tensor, proceed without it.
        provide_out = false;
      }
    }

    // Perform the bmm operation
    torch::Tensor result;

    if (provide_out) {
        result = torch::bmm_out(out_tensor, input, mat2);
    } else {
        result = torch::bmm(input, mat2);
    }

  } catch (const c10::Error &e) {
    // Catch PyTorch errors specifically to avoid catching our own logic errors
    // std::cerr << "C10 Error: " << e.what() << std::endl;
    return 0; // Keep input, it's a valid API error discovery
  } catch (const std::exception &e) {
    std::cout << "Exception caught: " << e.what() << std::endl;
    return -1; // Discard input for unexpected errors
  }
  return 0;
}