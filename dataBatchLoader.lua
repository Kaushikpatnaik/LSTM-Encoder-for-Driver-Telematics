--[[ 
loader for tensor datasets in lua. Splits the data into train, validation
and testing. Also returns miniBatches through function call.

The script assumes that sequences have already been randomly shuffled
so that different classes are inputed to the network in one batch
--]]

require 'torch'
require 'math'

local dataBatchLoader = {}
dataBatchLoader.__index = dataBatchLoader

function dataBatchLoader.create(tensor_datafile, tensor_classfile, batch_size, seq_length, split_fraction)
    
    local self = {}
    setmetatable(self, dataBatchLoader)

    -- construct tensor
    print('loading data files...')
    local data = torch.load(tensor_datafile)
    local class = torch.load(tensor_classfile)
    
    -- ensure the overall data length is a multiple of batch_size and seq_length
    local len = data:size(1)
    if len % (batch_size * seq_length) ~= 0 then
        data = data:sub(1, batch_size * seq_length 
                    * math.floor(len / (batch_size * seq_length)))
    end
    
    -- self.batches is a table of tensors
    print('reshaping tensor...')
    self.batch_size = batch_size
    self.seq_length = seq_length

    -- modified for use when the sequences
    self.x_batches = data:split(seq_length*batch_size, 1)
    self.nbatches = #self.x_batches
    print('size of x_batches', #self.x_batches)
    print('number of batches %d', self.nbatches)
    self.class_batches = class:split(seq_length*batch_size, 1)
    assert(#self.x_batches == #self.class_batches)

    -- Split into train, validation and testing
    self.ntrain = math.floor(self.nbatches * split_fraction[1])
    self.nval = math.floor(self.nbatches * split_fraction[2])
    self.ntest = self.nbatches - self.ntrain - self.nval
    
    self.split_batches = {self.ntrain, self.nval, self.ntest}
    
     -- counter for tracking iterations
    self.current_batch = {0, 0, 0}
    self.evaluated_batches = {0, 0, 0}  -- number of times next_batch() called

    print('data loading done.The split of batches is train: %d, val: %d, test: %d', self.ntrain, self.nval, self.ntest)
    collectgarbage()
    return self
end

function dataBatchLoader:reset_counter(idx)
    self.current_batch[idx] = 0
end

function dataBatchLoader:nextBatch(idx)
    -- idx 1 is training, idx 2 is val, idx 3 is testing
    
    self.current_batch[idx] = (self.current_batch[idx] % self.split_batches[idx]) + 1
    local start_idx = self.current_batch[idx]
    
    if idx == 2 then start_idx = self.ntrain + start_idx end
    if idx == 3 then start_idx = self.ntrain + self.nval + start_idx end
    
    self.evaluated_batches[idx] = self.evaluated_batches[idx] + 1
    return self.x_batches[start_idx], self.class_batches[start_idx]
end

return dataBatchLoader

