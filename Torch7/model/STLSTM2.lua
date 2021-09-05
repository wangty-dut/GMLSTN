--lstm model for the seconed level
local STLSTM2 = {}
function STLSTM2.lstm(input_size, output_size, rnn_size, n, dropout) -- n: num_layers
  dropout = dropout or 0 

  -- there will be 4*n+1 inputs
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x
  table.insert(inputs, nn.Identity()()) -- control gate
  for L = 1, n do
    table.insert(inputs, nn.Identity()()) -- prev_ct[L]
    table.insert(inputs, nn.Identity()()) -- prev_ht[L]
  end
  for L = 1, n do
    table.insert(inputs, nn.Identity()()) -- prev_cj[L]
    table.insert(inputs, nn.Identity()()) -- prev_hj[L]
  end

  local x, input_size_L 
  local outputs = {}

  for L = 1, n do
    -- c,h from previos steps
    local prev_ct = inputs[L*3]
    local prev_ht = inputs[L*3+1]

    local prev_cj = inputs[n*2+L*3]
    local prev_hj = inputs[n*2+L*3+1]
    
    local control_gate = inputs[2]

    -- the input to this layer
    if (L == 1) then
      x = inputs[1]
      input_size_L = input_size
    else
      x = outputs[(L-1)*2]
      if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
      input_size_L = rnn_size
    end

    -- evaluate the input sums at once for efficiency
    -- for temporal component
    local i2h_t  = nn.Linear(input_size_L, 3 * rnn_size)(x)
    local h2ht = nn.Linear(rnn_size,     3 * rnn_size)(prev_ht)
    local all_input_sums_t = nn.CAddTable()({i2h_t, h2ht})

    local reshaped_t = nn.Reshape(3, rnn_size)(all_input_sums_t)
    local n1_t, n2_t, n3_t = nn.SplitTable(2)(reshaped_t):split(3)

    local in_gate_t = nn.Sigmoid()(n1_t)
    local forget_gate_t = nn.Sigmoid()(n2_t)
    local in_transform_t = nn.Tanh()(n3_t)

    -- for spatial component
    local i2h_s  = nn.Linear(input_size_L, 3 * rnn_size)(x):annotate{      name = 'i2h_1'  .. L}
    local h2hj = nn.Linear(rnn_size,     3 * rnn_size)(prev_hj):annotate{name = 'h2hj_' .. L}
    local all_input_sums_s = nn.CAddTable()({i2h_s, h2hj})

    local reshaped_s = nn.Reshape(3, rnn_size)(all_input_sums_s)
    local n1_s, n2_s, n3_s = nn.SplitTable(2)(reshaped_s):split(3)

    local in_gate_s = nn.Sigmoid()(n1_s)
    local forget_gate_s = nn.Sigmoid()(n2_s)
    local in_transform_s = nn.Tanh()(n3_s)

    -- perform the STLSTM update
    local next_c_t = nn.CAddTable()({
        nn.CMulTable()({forget_gate_t, prev_ct}),
        nn.CMulTable()({in_gate_t,  in_transform_t})})

    local next_c_j = nn.CAddTable()({
        nn.CMulTable()({forget_gate_s, prev_cj}),
        nn.CMulTable()({in_gate_s,  in_transform_s})})
    
    local next_c  = nn.CAddTable()({next_c_t,  nn.CMulTable()({control_gate, next_c_j})})
    local prev_htj = nn.CAddTable()({prev_ht, nn.CMulTable()({control_gate, prev_hj})})
    
    -- define individual output
    local i2h_in = nn.Linear(input_size_L, rnn_size)(x)
    local h2h_in = nn.Linear(rnn_size, rnn_size)(prev_htj)
    local n4 = nn.CAddTable()({i2h_in, h2h_in})
    local out_gate = nn.Sigmoid()(n4)
    
    -- gated cells form the output
    local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

    table.insert(outputs, next_c)
    table.insert(outputs, next_h)
  end

  -- set up the decoder
  local top_h = outputs[#outputs]
  if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end
  local proj = nn.Linear(rnn_size, rnn_size)(top_h):annotate{name='decoder'}
  local relu1 = nn.ReLU()(proj)
  local fc1 = nn.Linear(rnn_size, output_size)(relu1)
  table.insert(outputs, fc1)

  return nn.gModule(inputs, outputs)
end

return STLSTM2

