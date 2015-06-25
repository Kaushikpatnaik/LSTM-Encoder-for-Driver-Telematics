require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
local dataBatchLoader = require 'dataBatchLoader'
local LSTM = require 'LSTM'             -- LSTM timestep and utilities
local lstm_utils=require 'lstm_utils'


local cmd = torch.CmdLine()
cmd:text()
cmd:text('Training a two-layered encoder LSTM model for sequence classification')
cmd:text()
cmd:text('Options')
cmd:option('-classfile','classfile2.th7','filename of the drivers table')
cmd:option('-datafile','datafile2.th7','filename of the serialized torch ByteTensor to load')
cmd:option('-train_split',0.8,'Fraction of data into training')
cmd:option('-val_split',0.1,'Fraction of data into validation')
cmd:option('-batch_size',1,'number of sequences to train on in parallel')
cmd:option('-seq_length',50,'number of timesteps to unroll to')
cmd:option('-input_size',15,'number of dimensions of input')
cmd:option('-rnn_size',128,'size of LSTM internal state')
cmd:option('-depth',2,'Number of LSTM layers stacked on top of each other')
cmd:option('-dropout',0.5,'Droput Probability')
cmd:option('-max_epochs',10,'number of full passes through the training data')
cmd:option('-savefile','model_autosave','filename to autosave the model (protos) to, appended with the,param,string.t7')
cmd:option('-save_every',1000,'save every 1000 steps, overwriting the existing file')
cmd:option('-print_every',100,'how many steps/minibatches between printing out the loss')
cmd:option('-eval_every',1000,'evaluate the holdout set every 100 steps')
cmd:option('-seed',123,'torch manual random number generator seed')
cmd:text()

-- parse input params
local opt = cmd:parse(arg)

torch.setdefaulttensortype('torch.DoubleTensor')

local test_split = 1 - opt.train_split - opt.val_split
split_fraction = {opt.train_split, opt.val_split, test_split}

-- preparation stuff:
torch.manualSeed(opt.seed)
opt.savefile = cmd:string(opt.savefile, opt,
    {save_every=true, print_every=true, savefile=true, vocabfile=true, datafile=true})
    .. '.t7'

local loader = dataBatchLoader.create(
        opt.datafile, opt.classfile, opt.batch_size, opt.seq_length, split_fraction)

-- define model prototypes for ONE timestep, then clone them
local protos = {}
-- lstm timestep's input: {x, prev_c, prev_h}, output: {next_c, next_h}
protos.lstm = LSTM.deepLstm(opt.input_size, opt.rnn_size, opt.depth, opt.dropout)
-- The softmax and criterion layers will be added at the end of the sequence
softmax = nn.Sequential():add(nn.Linear(opt.rnn_size, 5)):add(nn.LogSoftMax())
criterion = nn.CrossEntropyCriterion()

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
    table.insert(init_state_global, initstate_c:clone())
    table.insert(init_state_global, initstate_c:clone())
end
local init_state = clone_list(init_state_global)

-- LSTM final state's backward message (dloss/dfinalstate) is 0, since it doesn't influence predictions
local dfinal_state = clone_list(init_state_global, true)

-- evaluate the loss over entire validation or test set
function eval_split(split_index)
    -- split index 2 is validation set
    -- split index 3 is test set
    
    print('evaluating loss over Data Split' .. split_index)
    local n = loader.split_batches[split_index]

    loader:reset_counter(split_index) -- move batch iteration pointer for this split to front
    local loss = 0
    local predicted_label = {}
    local count_acc = 0
    local output = {}
    
    for i = 1,n do -- iterate over batches in the split
        ------------------ get minibatch -------------------
        local x, y = loader:nextBatch(split_index)
        local x1 = x:t()

        ------------------- forward pass -------------------
        local lstm_state = {[0]=init_state} -- internal cell states of LSTM
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
        loss = loss + criterion:forward(predictions, y[t])
        
        -- taking argmax
        _, predicted_label[i] = predictions:max(1)
        if predicted_label[i] == y[t] then count_acc = count_acc + 1 end
        
        -- transfer final state to initial state (BPTT)
        init_state = lstm_state[#lstm_state]
        
    end

    loss = loss / opt.seq_length / n
    accuracy = count_acc / opt.batch_size / n
    
    table.insert(output, loss)
    table.insert(output, accuracy)
    
    return output
end


-- do fwd/bwd and return loss, grad_params
function feval(params_)
    if params_ ~= params then
        params:copy(params_)
    end
    grad_params:zero()
    
    ------------------ get minibatch -------------------
    local x, y = loader:nextBatch(1)
    local x1 = x:t()

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
    loss = loss + criterion:forward(predictions, y[t])
    
    ------------------ backward pass -------------------
    -- complete reverse order of the above
    local dlstm_state = {[opt.seq_length]=dfinal_state}    -- internal cell states of LSTM
    
    -- Backprop through softmax and crossentropy only at t=opt.seq_length
    t = opt.seq_length
    -- backprop through loss, and softmax/linear
    local doutput_t = criterion:backward(predictions, y[t])
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
local optim_state = {learningRate = 1e-1}
local iterations = opt.max_epochs * loader.nbatches
for i = 1, iterations do
    -- print('In iteration ',i)

    local _, loss = optim.adagrad(feval, params, optim_state)
    losses[#losses + 1] = loss[1]
    
    -- tune the learning rate every 5 epochs
    if i % (5*loader.nbatches) == 0 then
        optim_state.learningRate = optim_state.learningRate * 0.95
    end
    
    -- Print statistics in valuation set
    if i % opt.eval_every == 0 then
        local val_loss = eval_split(2)
        val_losses[i] = val_loss
        print(string.format("iteration %4d, accuracy = %6.8f, loss = %6.8f, loss/seq_len = %6.8f, gradnorm = %6.4e", i, val_loss[2], val_loss[1] * opt.seq_length, val_loss[1], grad_params:norm()))
    end

    if i % opt.save_every == 0 then
        torch.save(opt.savefile, protos)
    end

    if i % opt.print_every == 0 then
        print(string.format("iteration %4d, loss = %6.8f, loss/seq_len = %6.8f, gradnorm = %6.4e", i, loss[1], loss[1] / opt.seq_length, grad_params:norm()))
    end
    
    collectgarbage()
end

-- run prediction on testing
local test_loss = eval_split(3)
print(string.format("iteration %4d, accuracy = %6.8f, loss = %6.8f, loss/seq_len = %6.8f, gradnorm = %6.4e", i, val_loss[2], val_loss[1] * opt.seq_length, val_loss[1], grad_params:norm()))

