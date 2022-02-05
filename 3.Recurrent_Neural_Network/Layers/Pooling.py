import numpy as np
from Layers import Base, Initializers

class Pooling(Base.BaseLayer):
    ''' Does dimensionality reduction.
    Here, max pooling is implemented'''

    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape

    def forward(self, input_tensor):
        self.input_tensor = input_tensor #Save for future use in backward function
        batches = input_tensor.shape[0]
        channels = input_tensor.shape[1]
        output_tensor = list()

        for b in range(batches):
            locations = list()
            for c in range(channels): 
                image_part = input_tensor[b,c]

                if self.pooling_shape[0]%2 == 0: w = int(image_part.shape[0]-np.ceil((self.pooling_shape[0]-1)/2))
                if self.pooling_shape[1]%2 == 0: h = int(image_part.shape[1]-np.ceil((self.pooling_shape[1]-1)/2))
                if self.pooling_shape[0]%2 != 0: w = int(image_part.shape[0]-self.pooling_shape[0]//2)
                if self.pooling_shape[1]%2 != 0: h = int(image_part.shape[1]-self.pooling_shape[1]//2)

                out = np.zeros([w,h])

                for (y,x), _ in np.ndenumerate(image_part):
                    if y<=image_part.shape[0]-self.pooling_shape[0] and x<=image_part.shape[1]-self.pooling_shape[1]:
                        maxpool = image_part[y:y+self.pooling_shape[0], x:x+self.pooling_shape[1]].max()
                        out[y,x] = maxpool

                fin = np.zeros([out.shape[0]//self.stride_shape[0], out.shape[1]//self.stride_shape[1]])
                fin = out[::self.stride_shape[0],::self.stride_shape[1]]

                locations.append(fin)
            output_tensor.append(locations)

        return np.array(output_tensor)


    def up_image(self, stride_shape, pooling_shape, input_part, error_part):
        '''Helper function for backward'''
        up_img = np.zeros_like(input_part)

        for (y, x), v in np.ndenumerate(error_part):
            ax = x*stride_shape[1]
            ay = y*stride_shape[0]
  
            locations = input_part[ay:ay+pooling_shape[0],ax:ax+pooling_shape[1]]
            i,j = np.where(locations == locations.max()) # Get locations of maxima
            up_img[ay+i, ax+j] += v

        return up_img


    def backward(self, error_tensor):
        self.error_tensor = error_tensor
        batch_num = error_tensor.shape[0]
        channel_num = error_tensor.shape[1]
        output_tensor = list()

        for b in range(batch_num):
            locations = list()
            for c in range(channel_num):
                locations.append(self.up_image(self.stride_shape, self.pooling_shape, self.input_tensor[b,c], error_tensor[b,c]))
            output_tensor.append(locations)
        
        return np.array(output_tensor)
