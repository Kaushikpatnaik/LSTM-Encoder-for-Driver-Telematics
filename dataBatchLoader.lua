--[[ 
loader for tensor datasets in lua. Splits the data into train, validation
and testing. Also returns miniBatches through function call.

The script assumes that sequences have already been randomly shuffled
so that different classes are inputed to the network in one batch
--]]

require 'torch'
require 'math'
--local csvFileReader = require 'csvFileReader'

local dataBatchLoader = {}
dataBatchLoader.__index = dataBatchLoader

-- Split string
function string:split(sep)
  local sep, fields = sep, {}
  local pattern = string.format("([^%s]+)", sep)
  self:gsub(pattern, function(substr) fields[#fields + 1] = substr end)
  return fields
end

function dataBatchLoader.loadCSV(csv_ipfile)

    local filePath = csv_ipfile

    -- Count number of rows and columns in file
    local i = 0
    for line in io.lines(filePath) do
      if i == 0 then
        COLS = #line:split(',')
      end
      i = i + 1
    end

    local ROWS = i

    -- Read data from CSV to tensor
    local csvFile = io.open(filePath, 'r')
    local header = csvFile:read()

    local data = torch.Tensor(ROWS, COLS)

    local i = 0
    for line in csvFile:lines('*l') do
      i = i + 1
      local l = line:split(',')
      for key, val in ipairs(l) do
        data[i][key] = val
      end
    end

    csvFile:close()
    
    return data

    -- Serialize tensor
    --local outputFilePath = opt.opfilepath
    --torch.save(outputFilePath, data)
end

function dataBatchLoader.create(csv_datafile, csv_classfile, batch_size, seq_length, split_fraction)
    
    local self = {}
    setmetatable(self, dataBatchLoader)

    -- construct tensor
    print('loading data files...')
    local data = dataBatchLoader.loadCSV(csv_datafile)
    local class =  dataBatchLoader.loadCSV(csv_classfile)
    
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
    -- Split into 10 fold cross validation batches
    self.cTrain = math.floor(self.nbatches * 0.2)
    self.ntrain = math.floor(self.cTrain * 4)
    --self.nval = math.floor(self.cTrain)
    --self.ntest = self.nbatches - self.ntrain - self.nval
    self.ntest = self.nbatches - self.ntrain
    
    --self.split_batches = {self.ntrain, self.nval, self.ntest}
    self.split_batches = {self.ntrain, self.ntest}
    
     -- counter for tracking iterations
    self.current_batch = {0, 0}

    print('data loading done.The split of batches is train: %d, test: %d', self.ntrain, self.ntest)
    collectgarbage()
    return self
end

function dataBatchLoader:reset_counter(idx)
    self.current_batch[idx] = 0
end

function dataBatchLoader:nextBatchCrossVal(cross_idx, idx)
    -- cross_idx: is the k fold cross validation step we are in
    -- idx: is train, val, test index
    -- we know the start position and number of batches for train, val and test
    
    local start_idx = {((cross_idx)%5)*self.cTrain+1,(cross_idx-1)*self.cTrain+1} 

    -- idx 1 is training, idx 2 is val, idx 3 is testing
    -- current idx under each fold
    self.current_batch[idx] = (self.current_batch[idx] % self.split_batches[idx]) + 1
    local k_start_idx = (start_idx[idx] + self.current_batch[idx]) % self.nbatches + 1
    
    --print(start_idx)
    --print(k_start_idx)

    self.current_batch[idx] = self.current_batch[idx] + 1
    return self.x_batches[k_start_idx], self.class_batches[k_start_idx]
end
 
function dataBatchLoader:nextBatch(idx)
    -- idx 1 is training, idx 2 is val, idx 3 is testing
    
    self.current_batch[idx] = (self.current_batch[idx] % self.split_batches[idx]) + 1
    local start_idx = self.current_batch[idx]
    
    if idx == 2 then start_idx = self.ntrain + start_idx end
    if idx == 3 then start_idx = self.ntrain + self.nval + start_idx end
    
    self.current_batch[idx] = self.current_batch[idx] + 1
    return self.x_batches[start_idx], self.class_batches[start_idx]
end

return dataBatchLoader

