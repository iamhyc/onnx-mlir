// RUN: onnx-mlir %s -O3 --mtriple=x86_64-pc-linux-gnu --shape-inference --convert-onnx-to-krnl='ops-for-call=Conv' --convert-krnl-to-affine --convert-krnl-to-llvm -o /tmp/e2e_conv_call.o 2>&1 | FileCheck %s

// End-to-end test: onnx.Conv with dynamic shape -> ops-for-call=Conv -> LLVM
// Verifies the complete pipeline for Conv with array attributes (pads/strides/dilations)

func.func private @test_e2e_conv_call_with_arrays(%arg0 : tensor<?x?x?x?xf32>, %arg1 : tensor<5x2x3x3xf32>, %arg2 : tensor<5xf32>) -> tensor<*xf32> {
  %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {
    auto_pad = "NOTSET", 
    dilations = [1, 1], 
    group = 1 : si64, 
    pads = [1, 1, 1, 1], 
    strides = [1, 1]
  } : (tensor<?x?x?x?xf32>, tensor<5x2x3x3xf32>, tensor<5xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// CHECK: llvm.call @Conv
