// Original Location at:
// /Data/devesh/llvm-project/mlir/test/Integration/GPU/CUDA/gpu-addition.mlir

module attributes {gpu.container_module} {
    gpu.module @kernels {
        gpu.func @add_arrays (%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>) kernel {
            %idx = gpu.thread_id x
            
            %val0 = memref.load %arg0[%idx] : memref<?xf32>
            %val1 = memref.load %arg1[%idx] : memref<?xf32>

            %result = arith.addf %val0, %val1 : f32
            memref.store %result, %arg2[%idx] : memref<?xf32>
    
            gpu.return
        }
    }



    func.func @main() {

        %size = arith.constant 4 : index

        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c3 = arith.constant 3 : index
        

        %constant_1 = arith.constant 1.0 : f32
        %constant_2 = arith.constant 2.0 : f32

        %arg0 = memref.alloc(%size) : memref<?xf32>
        %arg1 = memref.alloc(%size) : memref<?xf32>
        %arg2 = memref.alloc(%size) : memref<?xf32>

        affine.for %i = 0 to %size {
            memref.store %constant_1, %arg0[%i] : memref<?xf32>
            memref.store %constant_2, %arg1[%i] : memref<?xf32>
        }

        %printval = memref.cast %arg2 : memref<?xf32> to memref<*xf32>
        
        gpu.launch_func @kernels::@add_arrays
            blocks in (%c1, %c1, %c1) 
            threads in (%size, %c1, %c1)
            args(%arg0 : memref<?xf32> , %arg1 : memref<?xf32>, %arg2 : memref<?xf32>)

        call @printMemrefF32(%printval) : (memref<*xf32>) -> ()
        //CHECK: [3, 3, 3, 3]
        return
    }
    func.func private @printMemrefF32(memref<*xf32>)
}

