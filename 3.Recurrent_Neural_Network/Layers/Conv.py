import numpy as np
from scipy import signal
from Layers import Base, Initializers

class Conv(Base.BaseLayer):

    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__() # Because it inherits the Base Layer
        self.trainable = True
        self.stride_shape = stride_shape # Can be single value or tuple 
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels # How many filters I have
        self.weights = np.random.random([num_kernels, *self.convolution_shape]) 
        self.bias = np.random.random(num_kernels)
        self._optimizer = None # Weight optimizer
        self._bias_optimizer = None # Bias optimizer
    
    def forward(self, input_tensor):
        #1D Order: batches, channels, y (width)
        #2D Order: batches, channel, y (width) , x (height)
        batches = input_tensor.shape[0]
        self.input_tensor = input_tensor
        
        channels = input_tensor.shape[1] #Channels of the image (if it's RGB or not)
        one_dim_conv = len(input_tensor.shape) == 3 # Flag that is true if it is a 1D Convolution
        output_tensor = np.zeros([batches, self.num_kernels, *input_tensor.shape[2:]])
        
        for b in range(batches):            # iterate over each tensor (e.g. image) in the batch
            for k in range(self.num_kernels): # iterate over each kernel
                for c in range(channels):  # iterate over each channel to sum them up in the end to get 3D convolution (feature map)
                    output_tensor[b, k] += signal.correlate(input_tensor[b, c], self.weights[k, c], mode = 'same')
        
                output_tensor[b,k] += self.bias[k] # add bias to each feature map

        if one_dim_conv:
            output_tensor = output_tensor[:, :, ::self.stride_shape[0]]
        else:
            output_tensor = output_tensor[:, :, ::self.stride_shape[0], ::self.stride_shape[1]]
        
        self.output_tensor = output_tensor
        return output_tensor

    def backward(self, error_tensor):
        out_error_tensor = np.zeros(self.input_tensor.shape)
        grad_weights = np.zeros(self.weights.shape)
        self.gradient_tensor = np.zeros_like(self.weights)
        self.bias_gradient = np.zeros_like(self.bias)

        batches = error_tensor.shape[0]
        channels = self.weights.shape[1]

        # Run for each image in the batch
        for b in range(batches):
            
            err_strided_tensor = np.zeros((self.num_kernels, *self.input_tensor.shape[2:] ))
            conv_2d_err = len(error_tensor.shape)==4
            
            for k in range(error_tensor.shape[1]):
                err_b_k = error_tensor[b, k, :] # Error array for the b^th image and k^th kernel
                if conv_2d_err: err_strided_tensor[k,::self.stride_shape[0], ::self.stride_shape[1]] = err_b_k
                else: err_strided_tensor[k,::self.stride_shape[0]] = err_b_k

            #Gradient with respect to the input
            for c in range(channels):
                err = signal.convolve(err_strided_tensor, np.flip(self.weights, 0)[:,c,:], mode='same') # The output is the same size as in1, centered with respect to the ‘full’ output.

                midchannel = int(err.shape[0] / 2)
                out_error_tensor[b,c,:] = err[midchannel,:]

            # Gradient with respect to the weights
            for k in range(self.num_kernels):
                self.bias_gradient[k] += np.sum(error_tensor[b, k, :]) # Formula slide 43 

                for c in range(self.input_tensor.shape[1]):
                    input_image = self.input_tensor[b,c,:]

                    if conv_2d_err:
                        pad_x = self.convolution_shape[1]/2 # Pad X dimension with half the kernel's width
                        pad_y = self.convolution_shape[2]/2 # Pad Y dimension with half the kernel's width
                        px = (int(np.floor(pad_x)),int(np.floor(pad_x-0.5)))
                        py = (int(np.floor(pad_y)),int(np.floor(pad_y-0.5)))
                        padded_image = np.pad(input_image, (px,py))
                    else:
                        pad_x = self.convolution_shape[1]/2
                        px = (int(np.floor(pad_x)),int(np.floor(pad_x-0.5)))
                        padded_image = np.pad(input_image, px)

                    grad_weights[k,c,:] = signal.correlate(padded_image, err_strided_tensor[k,:], mode="valid")
                    # The output consists only of those elements that do not rely on the zero-padding. 
                    # In ‘valid’ mode, either in1 or in2 must be at least as large as the other in every dimension.

            self.gradient_tensor += grad_weights # For every image

        # Update weights
        if self._optimizer is not None:
            self.weights = self._optimizer.calculate_update(self.weights, self.gradient_tensor)
        if self._bias_optimizer is not None:
            self.bias = self._bias_optimizer.calculate_update(self.bias, self.bias_gradient)

        return out_error_tensor

    def initialize(self, weights_initializer, bias_initializer):
        fan_in    = np.prod(self.convolution_shape)
        fan_out   = np.prod(self.convolution_shape[1:]) * self.num_kernels
        self.weights = weights_initializer.initialize(self.weights.shape, fan_in, fan_out)
        self.bias = bias_initializer.initialize(self.bias.shape, fan_in, fan_out)

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    @property
    def optimizer_b(self):
        return self._bias_optimizer

    @optimizer_b.setter
    def optimizer_b(self, optimizer_b):
        self._bias_optimizer = optimizer_b

    @property
    def gradient_weights(self):
        return self.gradient_tensor

    @property
    def gradient_bias(self):
        return self.bias_gradient