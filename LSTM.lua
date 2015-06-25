--[[
This file creates one timestep of the LSTM model. The model is then
unrolled in time using the "clone_model" function in lstm_utils file

For details on the LSTM model please see
http://web.eecs.utk.edu/~itamar/courses/ECE-692/Bobby_paper1.pdf
--]]

local LSTM = {}

function LSTM.lstm(opt)
    -- Three inputs
    -- 1.current data: x
    -- 2.previous hidden unit: prev_h
    -- 3.previous memory cell: prev_c
    -- Two Outputs
    -- 1. next hidden unit: next_h
    -- 2. next memory cell: next_c
    
    local x = nn.Identity()()
    local prev_h = nn.Identity()()
    local prev_c = nn.Identity()()
    
    -- ip2hidden and hidden2hidden are the weights based on input and rnn size
    function sum_inputs()
        local ip2hidden = nn.Linear(opt.input_size, opt.rnn_size)(x)
        local hidden2hidden = nn.Linear(opt.rnn_size, opt.rnn_size)(prev_h)
        return nn.CAddTable()({ip2hidden, hidden2hidden})
    end
    
    -- compute the four gates
    local ip_gate = nn.Sigmoid()(sum_inputs())
    local ip_transform = nn.Tanh()(sum_inputs())
    local forget_gate = nn.Sigmoid()(sum_inputs())
    local out_gate = nn.Sigmoid()(sum_inputs())
    
    -- next_c is determined through element wise multiplication of above terms and prev_c
    local next_c = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({ip_gate, ip_transform})
    })
    
    -- next_h is determined by Tanh(next_c) and out_gate
    local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
    
    if opt.dropout > 0 then next_h = nn.Dropout(opt.dropout)(next_h) end
    
    -- gModule is a nngraph extension to Module in nn package.
    -- it allows us to create complex models and visualize the gradient flow
    return nn.gModule({x, prev_c, prev_h}, {next_c, next_h})
end

function LSTM.deepLstm(input_size, rnn_size, depth, dropout)
    -- for gModule we need to keep the inputs and ouputs in a single table

    -- For each layer there will be 2 inputs, prev_h and ouput from
    -- bottom layer. Thus there will be 2*n+1 inputs due to data also

    local inputs = {}
    table.insert(inputs, nn.Identity()())   -- x
    for l = 1,depth do
        table.insert(inputs, nn.Identity()()) -- prev_c[l]
        table.insert(inputs, nn.Identity()()) -- prev_h[l]
    end

    local x, ip_size_track
    local outputs = {}
    for l=1,depth do
        -- For each layer we will obtain the prev_c and prev_h, and calculate
        -- the new c and h values
        local prev_c = inputs[2*l]  -- prev_c always in even position
        local prev_h = inputs[2*l+1] -- prev_h follows prev_c

        -- input
        if l == 1 then
            x = inputs[1]
            ip_size_track = input_size
        else
            x = outputs[(l-1)*2]  -- insert h after c    
            ip_size_track  = rnn_size
            -- simple dropout between layers
            if dropout > 0 then x = nn.Dropout(dropout)(x) end  
        end

        -- evaluate sums at the same time
        local ip2hidden = nn.Linear(ip_size_track, 4*rnn_size)(x)
        local hidden2hidden = nn.Linear(rnn_size, 4*rnn_size)(prev_h)
        local sums = nn.CAddTable()({ip2hidden, hidden2hidden})
        
        -- extract idv gates
        local sigmoid_split = nn.Narrow(2, 1, 3*rnn_size)(sums)
        local tanh_split = nn.Narrow(2,3*rnn_size+1,rnn_size)(sums)

        local sigmoid_op = nn.Sigmoid()(sigmoid_split)

        local ip_gate = nn.Narrow(2, 1, rnn_size)(sigmoid_op)
        local forget_gate = nn.Narrow(2,rnn_size+1,rnn_size)(sigmoid_op)
        local out_gate = nn.Narrow(2,2*rnn_size+1,rnn_size)(sigmoid_op)
        local ip_transform = nn.Tanh()(tanh_split)

        -- next_c is determined through element wise multiplication of above terms and prev_c
        local next_c = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({ip_gate, ip_transform})
        })

        -- next_h is determined by Tanh(next_c) and out_gate
        local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

        table.insert(outputs, next_c)
        table.insert(outputs, next_h)
    end
    
    local top_hidden = outputs[#outputs]
    if dropout > 0 then top_hidden = nn.Dropout(dropout)(top_hidden) end
    
    return nn.gModule(inputs, outputs)
    
end

return LSTM    

