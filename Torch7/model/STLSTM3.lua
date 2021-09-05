--lstm model for the third level
local STLSTM3 = {}
function STLSTM3.lstm(input_size, output_size, rnn_size, n, dropout, tao1) -- n: num_layers
  dropout = dropout or 0 

  -- there will be 4*n+1 inputs
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x
  table.insert(inputs, nn.Identity()()) -- control gate1
  table.insert(inputs, nn.Identity()()) -- control gate2
  for L = 1, n do
    table.insert(inputs, nn.Identity()()) -- prev_cj1[L]
    table.insert(inputs, nn.Identity()()) -- prev_hj1[L]
  end
  
  for L = 1, n do
    table.insert(inputs, nn.Identity()()) -- prev_cj2[L]
    table.insert(inputs, nn.Identity()()) -- prev_ht2[L]
  end
  
  for L = 1, n do
    table.insert(inputs, nn.Identity()()) -- prev_ct[L]
    table.insert(inputs, nn.Identity()()) -- prev_ht[L]
  end

  local x, input_size_L 
  local outputs = {}

  for L = 1, n do
    -- c,h from previos steps
    local prev_cj1 = inputs[L*4]
    local prev_hj1 = inputs[L*4+1]

    local prev_cj2 = inputs[n*2+L*4]
    local prev_hj2 = inputs[n*2+L*4+1]
    
    local prev_ct = inputs[n*4+L*4]
    local prev_ht = inputs[n*4+L*4+1]
    
    local control_gate1 = inputs[2]   
    local control_gate2 = inputs[3]

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
    -- for spatial module 1
    local i2h_1  = nn.Linear(input_size_L, 4 * rnn_size)(x):annotate{      name = 'i2h_'  .. L}
    local h2ht_1 = nn.Linear(rnn_size,     4 * rnn_size)(prev_ht):annotate{name = 'h2ht_' .. L}
    local h2hj_1 = nn.Linear(rnn_size,     4 * rnn_size)(prev_hj1):annotate{name = 'h2ht_' .. L}
    local all_input_sums_1 = nn.CAddTable()({i2h_1, h2ht_1, h2hj_1})

    local reshaped_1 = nn.Reshape(4, rnn_size)(all_input_sums_1)
    local n1_1, n2_1, n3_1, n4_1 = nn.SplitTable(2)(reshaped_1):split(4)

    local in_gate_t1 = nn.Sigmoid()(n1_1)
    local forget_gate_t1 = nn.Sigmoid()(n2_1)
    local forget_gate_j1 = nn.Sigmoid()(n3_1)
    local in_transform_t1 = nn.Tanh()(n4_1)

    -- for spatial module 2
    local i2h_2  = nn.Linear(input_size_L, 4 * rnn_size)(x):annotate{      name = 'i2h_'  .. L}
    local h2ht_2 = nn.Linear(rnn_size,     4 * rnn_size)(prev_ht):annotate{name = 'h2ht_' .. L}
    local h2hj_2 = nn.Linear(rnn_size,     4 * rnn_size)(prev_hj1):annotate{name = 'h2ht_' .. L}
    local all_input_sums_2 = nn.CAddTable()({i2h_2, h2ht_2, h2hj_2})

    local reshaped_2 = nn.Reshape(4, rnn_size)(all_input_sums_2)
    local n1_2, n2_2, n3_2, n4_2 = nn.SplitTable(2)(reshaped_2):split(4)

    local in_gate_t2 = nn.Sigmoid()(n1_2)
    local forget_gate_t2 = nn.Sigmoid()(n2_2)
    local forget_gate_j2 = nn.Sigmoid()(n3_2)
    local in_transform_t2 = nn.Tanh()(n4_2)

    -- perform the STLSTM update
    local next_c_t1 = nn.CAddTable()({
        nn.CMulTable()({forget_gate_t1, prev_ct}),
        nn.CMulTable()({forget_gate_j1, prev_cj1}),
        nn.CMulTable()({in_gate_t1, in_transform_t1})})

    local next_c_t2 = nn.CAddTable()({
        nn.CMulTable()({forget_gate_t2, prev_ct}),
        nn.CMulTable()({forget_gate_j2, prev_cj2}),
        nn.CMulTable()({in_gate_t2, in_transform_t2})})
    
    local next_c = nn.CAddTable()({nn.CMulTable()({control_gate1, next_c_t1}), nn.CMulTable()({control_gate2, next_c_t2})})
    local prev_htj = nn.CAddTable()({nn.CMulTable()({control_gate1, prev_hj1}), nn.CMulTable()({control_gate2, prev_hj2})})
    
    -- define individual output
    local i2h_in = nn.Linear(input_size_L, rnn_size)(x)
    local h2h_in_jk = nn.Linear(rnn_size, rnn_size)(prev_htj)
    local h2h_in_i = nn.Linear(rnn_size, rnn_size)(prev_ht)
    local n4 = nn.CAddTable()({i2h_in, h2h_in_jk, h2h_in_i})
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

return STLSTM3

