require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'csvigo'
local dataBatchLoader = require 'dataBatchLoader'
local LSTM = require 'LSTM'             -- LSTM timestep and utilities
local lstm_utils=require 'lstm_utils'


local cmd = torch.CmdLine()
cmd:text()
cmd:text('Training a two-layered encoder LSTM model for sequence classification')
cmd:text()
cmd:text('Options')
cmd:option('-classfile','./torch_data/classfile2.th7','filename of the drivers table')
cmd:option('-datafile','./torch_data/datafile2.th7','filename of the serialized torch ByteTensor to load')
cmd:option('-opfile','./torch_data/datafile2.th7','filename of the serialized torch ByteTensor to load')
cmd:option('-train_split',0.8,'Fraction of data into training')
cmd:option('-val_split',0.1,'Fraction of data into validation')
cmd:option('-batch_size',1,'number of sequences to train on in parallel')
cmd:option('-seq_length',50,'number of timesteps to unroll to')
cmd:option('-input_size',15,'number of dimensions of input')
cmd:option('-rnn_size',128,'size of LSTM internal state')
cmd:option('-depth',2,'Number of LSTM layers stacked on top of each other')
cmd:option('-dropout',0,'Droput Probability')
cmd:option('-max_epochs',10,'number of full passes through the training data')
cmd:option('-savefile','model_autosave','filename to autosave the model (protos) to, appended with the,param,string.t7')
cmd:option('-save_every',10000,'save every 1000 steps, overwriting the existing file')
cmd:option('-print_every',1000,'how many steps/minibatches between printing out the loss')
cmd:option('-eval_every',1000,'evaluate the holdout set every 100 steps')
cmd:option('-seed',123,'torch manual random number generator seed')
cmd:option('-gpuid',-1,'which gpu to use. -1 == CPU usage')
cmd:text()

-- parse input params
local opt = cmd:parse(arg)

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(4)

local test_split = 1 - opt.train_split - opt.val_split
split_fraction = {opt.train_split, opt.val_split, test_split}

-- preparation stuff:
torch.manualSeed(opt.seed)
opt.savefile = cmd:string(opt.savefile, opt,
    {save_every=true, print_every=true, savefile=true, vocabfile=true, datafile=true})
    .. '.t7'

-- initialize cunn/cutorch for training on the GPU and fall back to CPU gracefully
if opt.gpuid >= 0 then
    local ok, cunn = pcall(require, 'cunn')
    local ok2, cutorch = pcall(require, 'cutorch')
    if not ok then print('package cunn not found!') end
    if not ok2 then print('package cutorch not found!') end
    if ok and ok2 then
        print('using CUDA on GPU ' .. opt.gpuid .. '...')
        cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
        cutorch.manualSeed(opt.seed)
    else
        print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
        print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
        print('Falling back on CPU mode')
        opt.gpuid = -1 -- overwrite user setting
    end
end

local loader = dataBatchLoader.create(
        opt.datafile, opt.classfile, opt.batch_size, opt.seq_length, split_fraction)

-- define model prototypes for ONE timestep, then clone them
local protos = {}
-- lstm timestep's input: {x, prev_c, prev_h}, output: {next_c, next_h}
protos.lstm = LSTM.deepLstm(opt.input_size, opt.rnn_size, opt.depth, opt.dropout)
-- The softmax and criterion layers will be added at the end of the sequence
softmax = nn.Sequential():add(nn.Linear(opt.rnn_size, 2)):add(nn.LogSoftMax())
criterion = nn.CrossEntropyCriterion()

--ship the model to GPU
if opt.gpuid >= 0 then
    for k,v in pairs(protos) do v:cuda() end
    softmax:cuda()
    criterion:cuda()
end

-- put the above things into one flattened parameters tensor
local params, grad_params = lstm_utils.combine_all_parameters(protos.lstm, softmax)
params:uniform(-0.08, 0.08)

-- make a bunch of clones, AFTER flattening, as that reallocates memory
local clones = {}
for name,proto in pairs(protos) do
    print('cloning '..name)
    clones[name] = lstm_utils.clone_model(proto, opt.seq_length, not proto.parameters)
end

-- LSTM initial state (zero initially, but final state gets sent to initial state when we do BPTT)
init_state_global = {}
for l=1,opt.depth do
    local initstate_c = torch.zeros(opt.batch_size, opt.rnn_size)
    if opt.gpuid >=0 then initstate_c = initstate_c:cuda() end
    table.insert(init_state_global, initstate_c:clone())
    table.insert(init_state_global, initstate_c:clone())
end
local init_state = clone_list(init_state_global)

-- LSTM final state's backward message (dloss/dfinalstate) is 0, since it doesn't influence predictions
local dfinal_state = clone_list(init_state_global, true)

-- evaluate the loss over entire validation or test set
function eval_split(k, split_index)
    -- split index 2 is validation set
    -- split index 3 is test set
    
    print('evaluating loss over Data Split' .. split_index)
    local n = loader.split_batches[split_index]

    loader:reset_counter(split_index) -- move batch iteration pointer for this split to front
    local loss = 0
    local count_acc = 0
    local predicted_label = torch.Tensor(n):fill(0)
    local output = {}
    local lstm_state = {[0]=init_state} -- internal cell states of LSTM
    
    for i = 1,n do -- iterate over batches in the split
        ------------------ get minibatch -------------------
        local x, y = loader:nextBatchCrossVal(k, split_index)
        local x1 = x:t()
        if opt.gpuid >= 0 then -- ship the input arrays to GPU
            -- have to convert to float because integers can't be cuda()'d
            x = x1:float():cuda()
            y = y:float():cuda()
        end

        ------------------- forward pass -------------------
        local predictions = {}              -- softmax outputs
        local counter                       -- local counter index for internal states of deep LSTMs
        
        for t=1,opt.seq_length do
            -- dimension 1 should always be batch_size
            local data_view = x1[{{}, t}]:view(opt.batch_size,-1)
            local temp_state = clones.lstm[t]:forward{data_view, unpack(lstm_state[t-1])}
            -- insert parameters sequentially
            lstm_state[t] = {}
            for k=1,#init_state_global do
                table.insert(lstm_state[t], temp_state[k])
            end
        end
        
        --softmax and loss are only at the last time step
        t = opt.seq_length
        -- get last element in time step t
        predictions = softmax:forward(lstm_state[t][#init_state_global])
        loss = loss + criterion:forward(predictions, y[1])
        
        -- taking argmax
        _, predicted_label[i] = predictions:max(2)
        --print(predicted_label[i])
        -- adding another additional third index ? 
        if predicted_label[i] == y[1][1] then 
            count_acc = count_acc + 1 
        end
        
        -- transfer final state to initial state (BPTT)
        lstm_state[0] = lstm_state[#lstm_state]
        
    end
    --print(count_acc)

    loss = loss / opt.seq_length / n
    accuracy = count_acc / opt.batch_size / n
    --print(type(predicted_label))
    local prediction = torch.Tensor(predicted_label)
    
    table.insert(output, loss)
    table.insert(output, accuracy)
    table.insert(output, prediction)
    
    return output
end

local kfold = 1
-- do fwd/bwd and return loss, grad_params
function feval(params_)
    if params_ ~= params then
        params:copy(params_)
    end
    grad_params:zero()
    
    --print(kfold)
    ------------------ get minibatch -------------------
    local x, y = loader:nextBatchCrossVal(kfold, 1)
    local x1 = x:t()
    if opt.gpuid >= 0 then -- ship the input arrays to GPU
        -- have to convert to float because integers can't be cuda()'d
        x = x1:float():cuda()
        y = y:float():cuda()
    end

    ------------------- forward pass -------------------
    local lstm_state = {[0]=init_state} -- internal cell states of LSTM
    local predictions = {}              -- softmax outputs
    local loss = 0
    local counter                       -- local counter index for internal states of deep LSTMs

    for t=1,opt.seq_length do
        -- dimension 1 should always be batch_size
        local data_view = x1[{{}, t}]:view(opt.batch_size,-1)
        local temp_state = clones.lstm[t]:forward{data_view, unpack(lstm_state[t-1])}
        -- insert parameters sequentially
        lstm_state[t] = {}
        for k=1,#init_state_global do
            table.insert(lstm_state[t], temp_state[k])
        end
    end
    
    --print('Done with complete FWD pass')
    
    --softmax and loss are only at the last time step
    t = opt.seq_length
    -- get last element in time step t
    predictions = softmax:forward(lstm_state[t][#init_state_global])
    loss = loss + criterion:forward(predictions, y[1])
    
    ------------------ backward pass -------------------
    -- complete reverse order of the above
    local dlstm_state = {[opt.seq_length]=dfinal_state}    -- internal cell states of LSTM
    
    -- Backprop through softmax and crossentropy only at t=opt.seq_length
    t = opt.seq_length
    -- backprop through loss, and softmax/linear
    local doutput_t = criterion:backward(predictions, y[1])
    local dsoftmax_t = softmax:backward(lstm_state[t][#init_state_global], doutput_t)
    dlstm_state[t][#init_state_global] =  dsoftmax_t
    
    -- backprop through LSTM timestep
    for t=opt.seq_length,1,-1 do
        local data_view = x1[{{}, t}]:view(opt.batch_size,-1)
        local dtemp_state = clones.lstm[t]:backward({data_view, unpack(lstm_state[t-1])}, dlstm_state[t])
        dlstm_state[t-1] = {}
        for k,v in pairs(dtemp_state) do
            if k > 1 then
                dlstm_state[t-1][k-1] = v
            end
        end
    end
    
    ------------------------ misc ----------------------
    -- transfer final state to initial state (BPTT)
    init_state = lstm_state[#lstm_state]

    -- clip gradient element-wise
    grad_params:clamp(-5, 5)

    return loss, grad_params
end

-- optimization stuff
local losses = {}
local val_losses = {}
local optim_state = {learningRate = 2e-5}
local iterations = opt.max_epochs * loader.ntrain
local final_pred = {}
while kfold <= 5 do
    print(kfold)
	for i = 1, iterations do
	    --print('In iteration ',i)

	    local _, loss = optim.rmsprop(feval, params, optim_state)
	    losses[#losses + 1] = loss[1]
	    
	    -- tune the learning rate every 3 epochs
	    if i % (3*loader.nbatches) == 0 then
		optim_state.learningRate = optim_state.learningRate * 0.95
	    end
	    
	    --[[ Print statistics in valuation set
	    if i % opt.eval_every == 0 then
		local val_loss = eval_split(kfold,2)
		val_losses[i] = val_loss
		print(string.format("iteration %4d, accuracy = %6.8f, loss = %6.8f, loss/seq_len = %6.8f, gradnorm = %6.4e", i, val_loss[2], val_loss[1] * opt.seq_length, val_loss[1], grad_params:norm()))
	    end
        ]]--

	    if i % opt.save_every == 0 then
		torch.save(opt.savefile, protos)
	    end

	    if i % opt.print_every == 0 then
		print(string.format("iteration %4d, loss = %6.8f, loss/seq_len = %6.8f, gradnorm = %6.4e", i, loss[1], loss[1] / opt.seq_length, grad_params:norm()))
	    end
	    
	    collectgarbage()
	end

	-- run prediction on testing
	local test_loss = eval_split(kfold,2)
	print(string.format("accuracy = %f, loss = %6.8f, loss/seq_len = %6.8f", test_loss[2], test_loss[1] * opt.seq_length, test_loss[1]))

	-- save predictions in file
    final_pred[kfold] = torch.totable(test_loss[3])
    
    kfold = kfold + 1
end

csvigo.save(opt.opfile,final_pred,',','raw')
