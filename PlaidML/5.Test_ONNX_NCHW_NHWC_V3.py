import numpy as np
import onnx
import onnx_plaidml.backend 
import plaidml

_ctx = onnx_plaidml.backend.PlaidMLBackend.ctx
_device     = plaidml.Device(_ctx, None)

    
def get_value(x):
     
    func = plaidml.tile.compose(_ctx, _device, [], [('out', x)])
    invoker = plaidml.Invoker(_ctx, func)
    shape = invoker.get_output_shape('out')
    tensor = plaidml.Tensor(_device, shape)
    invoker.set_output('out', tensor)
    invoker.invoke()
    array = np.ndarray(x.shape.dims, dtype=plaidml.tile.PLAIDML_DTYPE_TO_NUMPY[x.shape.dtype])
    with tensor.mmap_current() as view:
        view.copy_to_ndarray(array)
    return array
    

class NHWCtoNCHW(plaidml.tile.Operation):
    def __init__(self, x):
        code = """function (A[N,H,W,C]) -> (R) {
                      R[n,c,h,w:N,C,H,W] = =(A[n,h,w,c]);
                  }"""
        idims = x.shape.dims
        outshape = plaidml.tile.Shape(x.shape.dtype, (idims[0], idims[3], idims[1], idims[2]))
        super(NHWCtoNCHW, self).__init__(code, [('A', x)], [('R', outshape)])


#Generate input data
dimensions = (1, 2, 2, 3) 

A = (np.random.rand(*dimensions) + 1.) / 52.  

#Process data using PlaidML
input = plaidml.tile.Value.from_python_value(A)
result = NHWCtoNCHW.function(input)

# Report results        
print(A)
print(get_value(result)) 
 