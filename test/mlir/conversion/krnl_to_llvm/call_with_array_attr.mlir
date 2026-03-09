// RUN: onnx-mlir-opt --convert-krnl-to-llvm %s -split-input-file | FileCheck %s

func.func private @test_krnl_call_with_array_attr(%out: memref<1x5x8x8xf32>, %x: memref<1x2x10x10xf32>, %w: memref<5x2x3x3xf32>, %b: memref<5xf32>) {
  "krnl.call"(%out, %x, %w, %b) <{funcName = "Conv", numOfOutput = 1 : si64}> {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, pads = [1, 1, 1, 1], strides = [1, 1]} : (memref<1x5x8x8xf32>, memref<1x2x10x10xf32>, memref<5x2x3x3xf32>, memref<5xf32>) -> ()
  return
}

// CHECK: llvm.mlir.global internal constant @constant_{{[0-9]+}}(dense<1> : tensor<2xi64>)
// CHECK: llvm.mlir.global internal constant @constant_{{[0-9]+}}(dense<1> : tensor<4xi64>)
// CHECK-LABEL: llvm.func @test_krnl_call_with_array_attr
// CHECK: llvm.call @omTensorCreateUntyped
// CHECK: llvm.call @Conv(
