--require('mobdebug').start()
--recurrent assign init_state
--cuda
require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'
require 'gnuplot'

require 'util.misc'

local model_utils = require 'util.model_utils'
local STLSTM1 = require 'model.STLSTM1'
local STLSTM2 = require 'model.STLSTM2'
local STLSTM3 = require 'model.STLSTM3'

cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Options')
-- model params
cmd:option('-rnn_size', 128, 'size of LSTM internal state')
cmd:option('-num_layers', 1, 'number of layers in the LSTM') --
--cmd:option('-model', 'stlstm3', 'lstm, gru or rnn')
cmd:option('-model', 'stlstm1', 'stlstm2, or stlstm3')
-- optimization
cmd:option('-learning_rate', 2e-5, 'learning rate')  --2e-5
cmd:option('-learning_rate_decay', 0.998, 'learning rate decay')
cmd:option('-learning_rate_decay_after', 50, 'in number of epochs, when to start decaying the learning rate')
cmd:option('-decay_rate', 0.99, 'decay rate for rmsprop')
cmd:option('-dropout', 0.0, 'dropout for regularization, used after each RNN hidden layer. 0 = no dropout') --0.0
cmd:option('-seq_length', 1, 'number of timesteps to unroll for')   
cmd:option('-batch_size', 20, 'number of sequences to train on in parallel') 
cmd:option('-max_epochs', 2000, 'number of full passes through the training data')  --2000
cmd:option('-grad_clip', 8, 'clip gradients at this value')
cmd:option('-init_from', '', 'initialize network parameters from checkpoint at this path')
-- bookkeeping
cmd:option('-seed', 123, 'torch manual random number generator seed')
cmd:option('-print_every', 1, 'how many steps/minibatches between printing out the loss')
cmd:option('-eval_val_every', 1, 'every how many epochs should we evaluate on validation data?')
cmd:option('-savefile', 'stlstm3', 'filename to autosave the checkpont to. Will be inside checkpoint_dir/')
cmd:option('-accurate_gpu_timing', 0, 'set this flag to 1 to get precise timings when using GPU. Might make code bit slower but reports accurate timings.')
-- GPU/CPU
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')
cmd:option('-opencl', 0, 'use OpenCL (instead of CUDA)')

-- Evaluation Criteria
cmd:option('-crosssub', 1, 'cross-subject = 1 or cross-view = 0')
cmd:text()

-- parse input params
opt = cmd:parse(arg)
torch.manualSeed(opt.seed)

-- initialize cunn/cutorch for training on the GPU and fall back to CPU gracefully
if opt.gpuid >= 0 and opt.opencl == 0 then
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

------------------------------- Read data from CSV to tensor ---------------------------------------------

local csvFile = io.open('input1.csv', 'r')
local header = csvFile:read()

local ROWS = 850
local COLS = 15
local data = torch.CudaTensor(ROWS, COLS)

local i = 0
for line in csvFile:lines('*l') do
  i = i + 1
  local l = line:split(',')
  for key, val in ipairs(l) do
    data[i][key] = val
  end
end

csvFile:close()

---------------------------------------------------- define the input and output dimension ----------------------------------------------------------------
local input_size_all = 9
local input_size = 2
local output_size = 1

---eliminate abnormal points
for i = 1, data:size()[1] do
  if data[{i,4}] > 160 then
    data[{i,4}] = data[{i-1,4}]
  end
  if data[{i,9}] > 160 then
    data[{i,9}] = data[{i-1,9}]
  end
  if data[{i,14}] >160 then
    data[{i,14}] = data[{i-1,14}]
  end
end

local data_norm1 = torch.cat(data[{{},{1,2}}],data[{{},{4}}], 2)
local data_norm2 = torch.cat(data[{{},{6,7}}],data[{{},{9}}], 2)
local data_norm3 = torch.cat(data[{{},{11,12}}],data[{{},{14}}], 2)
local data_norm12 = torch.cat(data_norm1, data_norm2, 2)
local data_norm = torch.cat(data_norm12, data_norm3, 2)
local mean_v = torch.mean(data_norm)
local std_v = torch.std(data_norm)
local data_std = (data_norm - mean_v)/std_v
local input_std1 = torch.cat(data_std[{{},{1,2}}], data[{{},{5}}],2)
local input_std2 = torch.cat(data_std[{{},{4,5}}], data[{{},{10}}],2)
local input_std3 = torch.cat(data_std[{{},{7,8}}], data[{{},{15}}],2)
local input_std11 = torch.cat(input_std1, input_std2, 2)
local input_std = torch.cat(input_std11, input_std3, 2)
local output_std1 = torch.cat(data_std[{{},{3}}], data_std[{{},{6}}],2)
local output_std = torch.cat(output_std1, data_std[{{},{9}}],2)

--local output_std = data_norm[{{},{3}}]

--define train_set and test_set
local train_size = 600
local trainX = torch.CudaTensor(train_size, opt.seq_length, input_size_all):zero()
for i = 1, train_size do
  for t = 1, opt.seq_length do
    for k = 1, input_size_all do
      trainX[{{i},{t},{k}}] = input_std[{{i+t-1},{k}}]
    end
  end
end
local trainy = output_std[{{1+opt.seq_length-1,train_size+opt.seq_length-1},{}}]
--local trainy = output_std[{{1+opt.seq_length,train_size+opt.seq_length},{}}]

local test_size = 200
local testX = torch.CudaTensor(test_size, opt.seq_length, input_size_all):zero()
for i = 1, test_size do
  for t = 1, opt.seq_length do
    for k = 1, input_size_all do
      testX[{{i},{t},{k}}] = input_std[{{train_size+i+t-1},{k}}]
    end
  end
end
local testy = output_std[{{train_size+1+opt.seq_length-1,train_size+test_size+opt.seq_length-1},{}}]

-------------------------------------------- define LSTM network -------------------------------------------------------------
protos1 = {}
protos2 = {}
protos3 = {}
protos1.rnn = STLSTM1.lstm(input_size, output_size, opt.rnn_size, opt.num_layers, opt.dropout)
protos2.rnn = STLSTM2.lstm(input_size, output_size, opt.rnn_size, opt.num_layers, opt.dropout)
protos3.rnn = STLSTM3.lstm(input_size, output_size, opt.rnn_size, opt.num_layers, opt.dropout)

protos1.criterion = nn.MSECriterion()
protos2.criterion = nn.MSECriterion()
protos3.criterion = nn.MSECriterion()

-- the initial state of the cell/hidden states
init_state1 = {}   --initial state of level1
for L = 1, opt.num_layers do
  local h_init = torch.zeros(opt.batch_size, opt.rnn_size)  
  if opt.gpuid >=0 and opt.opencl == 0 then h_init = h_init:cuda() end
  table.insert(init_state1, h_init:clone())  --prev_ct
  table.insert(init_state1, h_init:clone())   --prev_ht
end

init_state2 = {}  --initial state of level2
for L = 1, opt.num_layers do
  local h_init = torch.zeros(opt.batch_size, opt.rnn_size)
  if opt.gpuid >=0 and opt.opencl == 0 then h_init = h_init:cuda() end
  table.insert(init_state2, h_init:clone())  --prev_ct
  table.insert(init_state2, h_init:clone())  --prev_ht
end

init_state3 = {}  --initial state of level3
for L = 1, opt.num_layers do
  local h_init = torch.zeros(opt.batch_size, opt.rnn_size)
  if opt.gpuid >=0 and opt.opencl == 0 then h_init = h_init:cuda() end
  table.insert(init_state3, h_init:clone())   --prev_ct
  table.insert(init_state3, h_init:clone())   --prev_ht
end

-- ship the model to the GPU if desired
if opt.gpuid >= 0 and opt.opencl == 0 then
    for k,v in pairs(protos1) do v:cuda() end
end
if opt.gpuid >= 0 and opt.opencl == 0 then
    for k,v in pairs(protos2) do v:cuda() end
end
if opt.gpuid >= 0 and opt.opencl == 0 then
    for k,v in pairs(protos3) do v:cuda() end
end

params, grad_params = model_utils.combine_all_parameters(protos1.rnn, protos2.rnn, protos3.rnn)

-- initialization
if do_random_init then
  params:uniform(-0.08, 0.08) -- small uniform numbers
  --params2:uniform(-0.08, 0.08)
  --params3:uniform(-0.08, 0.08)
end
--print(params)

print('number of parameters in the model1: ' .. params:nElement())

local JOINT_NUM = 3  --numbers of spatial joints

-- make a bunch of clones after flattening, as that reallocates memory
clones1 = {}
clones2 = {}
clones3 = {}
for name, proto in pairs(protos1) do
  print('cloning ' .. name)
  clones1[name] = model_utils.clone_many_times(proto, opt.seq_length*(JOINT_NUM-1), not proto.parameters)
end
for name, proto in pairs(protos2) do
  print('cloning ' .. name)
  clones2[name] = model_utils.clone_many_times(proto, opt.seq_length*(JOINT_NUM-1), not proto.parameters)
end
for name, proto in pairs(protos3) do
  print('cloning ' .. name)
  clones3[name] = model_utils.clone_many_times(proto, opt.seq_length, not proto.parameters)
end

-- preprocessing helper function
function prepro(x, y)  
    --x = x:transpose(1,2):contiguous() -- swap the axes for faster indexing
    -- y = y:transpose(1,2):contiguous() -- no need for y
    if opt.gpuid >= 0 and opt.opencl == 0 then -- ship the input arrays to GPU
        --x = x:float():cuda() -- have to convert to float because integers can't be cuda()'d
        y = y:float():cuda()
    end
    if opt.gpuid >= 0 and opt.opencl == 1 then -- ship the input arrays to GPU
        --x = x:cl()
        y = y:cl()
    end
    return x, y
end

-- do fwd/bwd and return loss, grad_params
init_state_global11 = clone_list(init_state1)
init_state_global12 = clone_list(init_state1)
init_state_global21 = clone_list(init_state2)
init_state_global22 = clone_list(init_state2)
init_state_global3 = clone_list(init_state3)

local current_training_batch_number = 0    --control the retrieve sample process

function eval_split(split_index, max_batches) 
  print('evaluating loss over split index ' .. split_index)
  local current_testing_batch_number = 0
  local loss = 0
  local rnn_state1 = {[0] = init_state_global11}
  table.insert(rnn_state1, init_state_global12)
  local rnn_state2 = {[0] = init_state_global21}
  table.insert(rnn_state2, init_state_global22)
  local rnn_state3 = {[0] = init_state_global3}
  for i = 1, torch.floor(test_size/opt.batch_size) do
    collectgarbage()
    current_testing_batch_number = current_testing_batch_number + 1
    local x, y = retrieve_batch(current_testing_batch_number,2)
    x, y = prepro(x, y)
------------------- forward pass -------------------
    local predictions1 = {}
    local predictions2 = {}
    local predictions3 = {}      

    local CurrRnnIndx1 = 0
    local CurrRnnIndx2 = 0
    local CurrRnnIndx3 = 0 
    for t = 1, opt.seq_length do
      CurrRnnIndx1 = CurrRnnIndx1 + 1
      CurrRnnIndx2 = CurrRnnIndx2 + 1
      CurrRnnIndx3 = CurrRnnIndx3 + 1
      clones1.rnn[CurrRnnIndx1]:evaluate() 
      clones1.rnn[CurrRnnIndx1+1]:evaluate()
      clones2.rnn[CurrRnnIndx2]:evaluate()
      clones2.rnn[CurrRnnIndx2+1]:evaluate()
      clones3.rnn[CurrRnnIndx3]:evaluate()

      --for rnn type1-------------------------------------
      local inputX11 = x[{{},{t},{4,5}}]:contiguous():view(opt.batch_size, input_size)
      local tempInput11 = {}
      table.insert(tempInput11, inputX11)
      --define preRnnIndex
      if t == 1 then
        preRnnIndex = 0
      else
        preRnnIndex = CurrRnnIndx1 - 2
      end
      for k, v in ipairs(rnn_state1[preRnnIndex]) do table.insert(tempInput11, v) end
      local lst11 = clones1.rnn[CurrRnnIndx1]:forward(tempInput11)
      rnn_state1[CurrRnnIndx1] = {}
      for index = 1, #init_state1 do table.insert(rnn_state1[CurrRnnIndx1], lst11[index]) end
      predictions1[CurrRnnIndx1] = lst11[#lst11]

      CurrRnnIndx1 = CurrRnnIndx1 + 1
      local inputX12 = x[{{},{t},{7,8}}]:contiguous():view(opt.batch_size, input_size)
      local tempInput12 = {}
      table.insert(tempInput12, inputX12)
      --define preRnnIndex
      if t == 1 then              
        preRnnIndex = 1
      else
        preRnnIndex = CurrRnnIndx1 - 2
      end
      for k, v in ipairs(rnn_state1[preRnnIndex]) do table.insert(tempInput12, v) end
      local lst12 = clones1.rnn[CurrRnnIndx1]:forward(tempInput12)
      rnn_state1[CurrRnnIndx1] = {}
      for index = 1, #init_state1 do table.insert(rnn_state1[CurrRnnIndx1], lst12[index]) end
      predictions1[CurrRnnIndx1] = lst12[#lst12]

      --for rnn type2--------------------------------------
      local inputX21 = x[{{},{t},{7,8}}]:contiguous():view(opt.batch_size, input_size)
      local tempInput21 = {}
      table.insert(tempInput21, inputX21)
      --define preRnnIndex
      if t == 1 then
        preRnnIndex = 0
      else
        preRnnIndex = CurrRnnIndx2 - 2
      end
      
      local control_gate = torch.zeros(opt.batch_size,opt.rnn_size):float():cuda()
      --local control_gate = torch.ones(opt.batch_size,opt.rnn_size):float():cuda()
      for k = 1, opt.batch_size do
        if x[{k,t,6}] < x[{k,t,9}] then
          control_gate[{{k},{}}] = torch.ones(1,opt.rnn_size):float():cuda()
        end
      end
      table.insert(tempInput21, control_gate)
      
      for k, v in ipairs(rnn_state2[preRnnIndex]) do table.insert(tempInput21, v) end  --temporal component
      for k, v in ipairs(rnn_state1[CurrRnnIndx1-1]) do table.insert(tempInput21, v) end  --spatial component
      local lst21 = clones2.rnn[CurrRnnIndx2]:forward(tempInput21)
      rnn_state2[CurrRnnIndx2] = {}
      for index = 1, #init_state2 do table.insert(rnn_state2[CurrRnnIndx2], lst21[index]) end
      predictions2[CurrRnnIndx2] = lst21[#lst21]

      CurrRnnIndx2 = CurrRnnIndx2 + 1
      local inputX22 = x[{{},{t},{4,5}}]:contiguous():view(opt.batch_size, input_size)
      local tempInput22 = {}
      table.insert(tempInput22, inputX22)
      --define preRnnIndex
      if t == 1 then              
        preRnnIndex = 1
      else
        preRnnIndex = CurrRnnIndx2 - 2
      end
      local control_gate = torch.zeros(opt.batch_size,opt.rnn_size):float():cuda()
      for k = 1, opt.batch_size do
        if x[{k,t,6}] > x[{k,t,9}] then
          control_gate[{{k},{}}] = torch.ones(1,opt.rnn_size):float():cuda()
        end
      end
      table.insert(tempInput22, control_gate)
    
      for k, v in ipairs(rnn_state2[preRnnIndex]) do table.insert(tempInput22, v) end  --temporal component
      for k, v in ipairs(rnn_state1[CurrRnnIndx1]) do table.insert(tempInput22, v) end  --spatial component
      local lst22 = clones2.rnn[CurrRnnIndx2]:forward(tempInput22)
      rnn_state2[CurrRnnIndx2] = {}
      for index = 1, #init_state2 do table.insert(rnn_state2[CurrRnnIndx2], lst22[index]) end
      predictions2[CurrRnnIndx2] = lst22[#lst22]

      --for rnn type3----------------------------------
      local inputX3 = x[{{},{t},{1,2}}]:contiguous():view(opt.batch_size, input_size)
      local tempInput3 = {}
      table.insert(tempInput3, inputX3)
      --define preRnnIndex
      preRnnIndex = CurrRnnIndx3 - 1
      
      local control_gate1 = torch.zeros(opt.batch_size, opt.rnn_size):float():cuda()
      local control_gate2 = torch.zeros(opt.batch_size, opt.rnn_size):float():cuda()
      --local control_gate1 = torch.ones(opt.batch_size, opt.rnn_size):float():cuda()
      --local control_gate2 = torch.ones(opt.batch_size, opt.rnn_size):float():cuda()
      for k = 1, opt.batch_size do
        if x[{k,t,6}] < x[{k,t,9}] then
          control_gate1[{{k},{}}] = torch.ones(1, opt.rnn_size):float():cuda()
        else
          control_gate2[{{k},{}}] = torch.ones(1, opt.rnn_size):float():cuda()
        end
      end
      table.insert(tempInput3, control_gate1)
      table.insert(tempInput3, control_gate2)
      
      for k, v in ipairs(rnn_state2[CurrRnnIndx2-1]) do table.insert(tempInput3, v) end --spatial1
      for k, v in ipairs(rnn_state2[CurrRnnIndx2]) do table.insert(tempInput3, v) end --spatial2
      for k, v in ipairs(rnn_state3[preRnnIndex]) do table.insert(tempInput3, v) end  --temporal
      local lst3 = clones3.rnn[CurrRnnIndx3]:forward(tempInput3)
      rnn_state3[CurrRnnIndx3] = {}
      for index = 1, #init_state3 do table.insert(rnn_state3[CurrRnnIndx3], lst3[index]) end
      predictions3[CurrRnnIndx3] = lst3[#lst3]
    end
    rnn_state1[0] = rnn_state1[1]
    rnn_state1[1] = rnn_state1[2]
    rnn_state2[0] = rnn_state2[1]
    rnn_state2[1] = rnn_state2[2]
    rnn_state3[0] = rnn_state3[1]
    loss = loss + clones3.criterion[CurrRnnIndx3]:forward(predictions3[CurrRnnIndx3], y[{{},{1}}])

  end
  return loss
end

function feval(x)-----------------------------------------------------------------------------------------------------------------------------------
  if x ~= params then
    params:copy(x) -- update parameters
  end
  grad_params:zero()

  ------------------ get minibatch -------------------
  local x, y = retrieve_batch(current_training_batch_number,1)
  x, y = prepro(x, y)
  ------------------- forward pass -------------------
  local rnn_state1 = {[0] = init_state_global11}
  table.insert(rnn_state1, init_state_global12)
  local rnn_state2 = {[0] = init_state_global21}
  table.insert(rnn_state2, init_state_global22)
  local rnn_state3 = {[0] = init_state_global3}
  local predictions1 = {}
  local predictions2 = {}
  local predictions3 = {}
  local loss = 0
  local preRnnIndex

  local CurrRnnIndx1 = 0
  local CurrRnnIndx2 = 0
  local CurrRnnIndx3 = 0 
  collectgarbage()
  for t = 1, opt.seq_length do
    CurrRnnIndx1 = CurrRnnIndx1 + 1
    CurrRnnIndx2 = CurrRnnIndx2 + 1
    CurrRnnIndx3 = CurrRnnIndx3 + 1
    clones1.rnn[CurrRnnIndx1]:training()
    clones1.rnn[CurrRnnIndx1+1]:training()
    clones2.rnn[CurrRnnIndx2]:training()
    clones2.rnn[CurrRnnIndx2+1]:training()
    clones3.rnn[CurrRnnIndx3]:training()

    --for rnn type1-------------------------------------
    local inputX11 = x[{{},{t},{4,5}}]:contiguous():view(opt.batch_size, input_size)
    local tempInput11 = {}
    table.insert(tempInput11, inputX11)
    --define preRnnIndex
    if t == 1 then
      preRnnIndex = 0
    else
      preRnnIndex = CurrRnnIndx1 - 2
    end
    for k, v in ipairs(rnn_state1[preRnnIndex]) do table.insert(tempInput11, v) end
    local lst11 = clones1.rnn[CurrRnnIndx1]:forward(tempInput11)
    rnn_state1[CurrRnnIndx1] = {}
    for index = 1, #init_state1 do table.insert(rnn_state1[CurrRnnIndx1], lst11[index]) end
    predictions1[CurrRnnIndx1] = lst11[#lst11]

    CurrRnnIndx1 = CurrRnnIndx1 + 1
    local inputX12 = x[{{},{t},{7,8}}]:contiguous():view(opt.batch_size, input_size)
    local tempInput12 = {}
    table.insert(tempInput12, inputX12)
    --define preRnnIndex
    if t == 1 then              
      preRnnIndex = 1
    else
      preRnnIndex = CurrRnnIndx1 - 2
    end
    for k, v in ipairs(rnn_state1[preRnnIndex]) do table.insert(tempInput12, v) end
    local lst12 = clones1.rnn[CurrRnnIndx1]:forward(tempInput12)
    rnn_state1[CurrRnnIndx1] = {}
    for index = 1, #init_state1 do table.insert(rnn_state1[CurrRnnIndx1], lst12[index]) end
    predictions1[CurrRnnIndx1] = lst12[#lst12]

    --for rnn type2--------------------------------------
    local inputX21 = x[{{},{t},{7,8}}]:contiguous():view(opt.batch_size, input_size)
    local tempInput21 = {}
    table.insert(tempInput21, inputX21)
    --define preRnnIndex
    if t == 1 then
      preRnnIndex = 0
    else
      preRnnIndex = CurrRnnIndx2 - 2
    end
    
    local control_gate = torch.zeros(opt.batch_size,opt.rnn_size):float():cuda()
    for k = 1, opt.batch_size do
      if x[{k,t,6}] < x[{k,t,9}] then
        control_gate[{{k},{}}] = torch.ones(1,opt.rnn_size):float():cuda()
      end
    end
    table.insert(tempInput21, control_gate)
    
    for k, v in ipairs(rnn_state2[preRnnIndex]) do table.insert(tempInput21, v) end  --temporal component
    for k, v in ipairs(rnn_state1[CurrRnnIndx1-1]) do table.insert(tempInput21, v) end  --spatial component
    local lst21 = clones2.rnn[CurrRnnIndx2]:forward(tempInput21)
    rnn_state2[CurrRnnIndx2] = {}
    for index = 1, #init_state2 do table.insert(rnn_state2[CurrRnnIndx2], lst21[index]) end
    predictions2[CurrRnnIndx2] = lst21[#lst21]

    CurrRnnIndx2 = CurrRnnIndx2 + 1
    local inputX22 = x[{{},{t},{4,5}}]:contiguous():view(opt.batch_size, input_size)
    local tempInput22 = {}
    table.insert(tempInput22, inputX22)
    --define preRnnIndex
    if t == 1 then              
      preRnnIndex = 1
    else
      preRnnIndex = CurrRnnIndx2 - 2
    end
    
    local control_gate = torch.zeros(opt.batch_size,opt.rnn_size):float():cuda()
    for k = 1, opt.batch_size do
      if x[{k,t,6}] > x[{k,t,9}] then
        control_gate[{{k},{}}] = torch.ones(1,opt.rnn_size):float():cuda()
      end
    end
    table.insert(tempInput22, control_gate)
    
    for k, v in ipairs(rnn_state2[preRnnIndex]) do table.insert(tempInput22, v) end  --temporal component
    for k, v in ipairs(rnn_state1[CurrRnnIndx1]) do table.insert(tempInput22, v) end  --spatial component
    local lst22 = clones2.rnn[CurrRnnIndx2]:forward(tempInput22)
    rnn_state2[CurrRnnIndx2] = {}
    for index = 1, #init_state2 do table.insert(rnn_state2[CurrRnnIndx2], lst22[index]) end
    predictions2[CurrRnnIndx2] = lst22[#lst22]

    --for rnn type3----------------------------------
    local inputX3 = x[{{},{t},{1,2}}]:contiguous():view(opt.batch_size, input_size)
    local tempInput3 = {}
    table.insert(tempInput3, inputX3)
    --define preRnnIndex
    preRnnIndex = CurrRnnIndx3 - 1
    
    local control_gate1 = torch.zeros(opt.batch_size, opt.rnn_size):float():cuda()
    local control_gate2 = torch.zeros(opt.batch_size, opt.rnn_size):float():cuda()
    for k = 1, opt.batch_size do
      if x[{k,t,6}] < x[{k,t,9}] then
        control_gate1[{{k},{}}] = torch.ones(1, opt.rnn_size):float():cuda()
      else
        control_gate2[{{k},{}}] = torch.ones(1, opt.rnn_size):float():cuda()
      end
    end
    table.insert(tempInput3, control_gate1)
    table.insert(tempInput3, control_gate2)
    
    for k, v in ipairs(rnn_state2[CurrRnnIndx2-1]) do table.insert(tempInput3, v) end --spatial1
    for k, v in ipairs(rnn_state2[CurrRnnIndx2]) do table.insert(tempInput3, v) end --spatial2
    for k, v in ipairs(rnn_state3[preRnnIndex]) do table.insert(tempInput3, v) end  --temporal
    local lst3 = clones3.rnn[CurrRnnIndx3]:forward(tempInput3)
    rnn_state3[CurrRnnIndx3] = {}
    for index = 1, #init_state3 do table.insert(rnn_state3[CurrRnnIndx3], lst3[index]) end
    predictions3[CurrRnnIndx3] = lst3[#lst3]
  end
  --loss = loss / (opt.seq_length * JOINT_NUM)
  loss = clones3.criterion[CurrRnnIndx3]:forward(predictions3[CurrRnnIndx3], y[{{},{1}}])

  ------------------ backward pass -------------------
  -- initialize gradient at time t to be zeros (there's no influence from future)
  local drnn_state1 = {}
  local drnn_state2 = {}
  local drnn_state3 = {}
  for i = 0, (opt.seq_length * (JOINT_NUM-1)) do
    drnn_state1[i] = clone_list(init_state1, true) -- true also zeros the clones
  end
  for i = 0, (opt.seq_length * (JOINT_NUM-1)) do
    drnn_state2[i] = clone_list(init_state2, true)
  end
  for i = 0, (opt.seq_length) do
    drnn_state3[i] = clone_list(init_state3, true)
  end

  local CurrRnnIndx1 = opt.seq_length * (JOINT_NUM-1)
  local CurrRnnIndx2 = opt.seq_length * (JOINT_NUM-1)
  local CurrRnnIndx3 = opt.seq_length
  collectgarbage()
  for t = opt.seq_length, 1, -1 do
    -- backprop through loss, and softmax/linear
    local doutput_t = clones3.criterion[CurrRnnIndx3]:backward(predictions3[CurrRnnIndx3], y[{{},{1}}])
    table.insert(drnn_state3[CurrRnnIndx3], doutput_t) -- drnn includes two part: 1) from t + 1, 2) from criterion
    assert (#(drnn_state3[CurrRnnIndx3]) == (#init_state3)+1)

    --for rnn type3----------------------------------
    local inputX3 = x[{{},{t},{1,2}}]:contiguous():view(opt.batch_size, input_size)

    local tempInput3 = {}
    table.insert(tempInput3, inputX3)
    --if preRnnIndex == 0 then random_list(init_state_global) end
    local control_gate1 = torch.zeros(opt.batch_size, opt.rnn_size):float():cuda()
    local control_gate2 = torch.zeros(opt.batch_size, opt.rnn_size):float():cuda()
    for k = 1, opt.batch_size do
      if x[{k,t,6}] < x[{k,t,9}] then
        control_gate1[{{k},{}}] = torch.ones(1, opt.rnn_size):float():cuda()
      else
        control_gate2[{{k},{}}] = torch.ones(1, opt.rnn_size):float():cuda()
      end
    end
    table.insert(tempInput3, control_gate1)
    table.insert(tempInput3, control_gate2)
    
    if t == 1 then              
      preRnnIndex = 0
    else
      preRnnIndex = CurrRnnIndx3-1
    end
    for k, v in ipairs(rnn_state2[CurrRnnIndx2-1]) do table.insert(tempInput3, v) end --spatial1
    for k, v in ipairs(rnn_state2[CurrRnnIndx2]) do table.insert(tempInput3, v) end --spatial2
    for k, v in ipairs(rnn_state3[preRnnIndex]) do table.insert(tempInput3, v) end
    local dlst3 = clones3.rnn[CurrRnnIndx3]:backward(tempInput3, drnn_state3[CurrRnnIndx3])

    for index = 1, #init_state3 do
      drnn_state3[preRnnIndex][index] = drnn_state3[preRnnIndex][index] + dlst3[index+7]                --temporal
    end
    
    for k = 1, opt.batch_size do
      if x[{k,t,6}] < x[{k,t,9}] then
        for index = 1, #init_state2 do
          drnn_state2[CurrRnnIndx2-1][index][k] = drnn_state2[CurrRnnIndx2-1][index][k] + dlst3[index+3+(#init_state3)][k]     --spatial
        end
      else
        for index = 1, #init_state2 do
          drnn_state2[CurrRnnIndx2][index][k] = drnn_state2[CurrRnnIndx2][index][k] + dlst3[index+3+(#init_state3)+(#init_state2)][k]    --spatial
        end
      end
    end

    --for rnn type2----------------------------------
    local doutput_t = clones2.criterion[CurrRnnIndx2]:backward(predictions2[CurrRnnIndx2], y[{{},{2}}])
    table.insert(drnn_state2[CurrRnnIndx2], doutput_t) -- drnn includes two part: 1) from t + 1, 2) from criterion
    assert (#(drnn_state2[CurrRnnIndx2]) == (#init_state2)+1)
    
    local inputX22 = x[{{},{t},{4,5}}]:contiguous():view(opt.batch_size, input_size)

    local tempInput22 = {}
    table.insert(tempInput22, inputX22)
    --if preRnnIndex == 0 then random_list(init_state_global) end

    if t == 1 then              
      preRnnIndex = 1
    else
      preRnnIndex = CurrRnnIndx2-2
    end
    
    local control_gate = torch.zeros(opt.batch_size,opt.rnn_size):float():cuda()
    for k = 1, opt.batch_size do
      if x[{k,t,6}] > x[{k,t,9}] then
        control_gate[{{k},{}}] = torch.ones(1,opt.rnn_size):float():cuda()
      end
    end
    table.insert(tempInput22, control_gate)
    
    for k, v in ipairs(rnn_state2[preRnnIndex]) do table.insert(tempInput22, v) end  --temporal
    for k, v in ipairs(rnn_state1[CurrRnnIndx1]) do table.insert(tempInput22, v) end --spatial1
    local dlst22 = clones2.rnn[CurrRnnIndx2]:backward(tempInput22, drnn_state2[CurrRnnIndx2])
    
    for index = 1, #init_state2 do    ---perform the best
      drnn_state2[preRnnIndex][index] = drnn_state2[preRnnIndex][index] + dlst22[index+2]   --temporal
    end    
    
    for k = 1, opt.batch_size do
      if x[{k,t,6}] > x[{k,t,9}] then
        for index = 1, #init_state1 do
          drnn_state1[CurrRnnIndx1][index][k] = drnn_state1[CurrRnnIndx1][index][k] + dlst22[index+2+(#init_state2)][k]      --spatial
        end
      end
    end
    
    -----------------------------
    local doutput_t = clones2.criterion[CurrRnnIndx2-1]:backward(predictions2[CurrRnnIndx2-1], y[{{},{3}}])
    table.insert(drnn_state2[CurrRnnIndx2-1], doutput_t) -- drnn includes two part: 1) from t + 1, 2) from criterion
    assert (#(drnn_state2[CurrRnnIndx2-1]) == (#init_state2)+1)
    
    local inputX21 = x[{{},{t},{7,8}}]:contiguous():view(opt.batch_size, input_size)

    local tempInput21 = {}
    table.insert(tempInput21, inputX21)

    if t == 1 then              
      preRnnIndex = 0
    else
      preRnnIndex = CurrRnnIndx2-1-2
    end
    
    local control_gate = torch.zeros(opt.batch_size,opt.rnn_size):float():cuda()
    for k = 1, opt.batch_size do
      if x[{k,t,6}] < x[{k,t,9}] then
        control_gate[{{k},{}}] = torch.ones(1,opt.rnn_size):float():cuda()
      end
    end
    table.insert(tempInput21, control_gate)
    
    for k, v in ipairs(rnn_state2[preRnnIndex]) do table.insert(tempInput21, v) end  --temporal
    for k, v in ipairs(rnn_state1[CurrRnnIndx1-1]) do table.insert(tempInput21, v) end --spatial1
    local dlst21 = clones2.rnn[CurrRnnIndx2-1]:backward(tempInput21, drnn_state2[CurrRnnIndx2-1])
    
    for index = 1, #init_state2 do
      drnn_state2[preRnnIndex][index] = drnn_state2[preRnnIndex][index] + dlst21[index+2]    --templaral
    end
    
    for k = 1, opt.batch_size do
      if x[{k,t,6}] < x[{k,t,9}] then
        for index = 1, #init_state1 do
          drnn_state1[CurrRnnIndx1-1][index][k] = drnn_state1[CurrRnnIndx1-1][index][k] + dlst21[index+2+(#init_state2)][k]     --spatial
        end
      end
    end    

    --for rnn type1----------------------------------
    local doutput_t = clones1.criterion[CurrRnnIndx1]:backward(predictions1[CurrRnnIndx1], y[{{},{3}}])
    table.insert(drnn_state1[CurrRnnIndx1], doutput_t) -- drnn includes two part: 1) from t + 1, 2) from criterion
    assert (#(drnn_state1[CurrRnnIndx1]) == (#init_state1)+1)
    local inputX12 = x[{{},{t},{7,8}}]:contiguous():view(opt.batch_size, input_size)

    local tempInput12 = {}
    table.insert(tempInput12, inputX12)
    --if preRnnIndex == 0 then random_list(init_state_global) end
    if t == 1 then              
      preRnnIndex = 1
    else
      preRnnIndex = CurrRnnIndx1-2
    end
    for k, v in ipairs(rnn_state1[preRnnIndex]) do table.insert(tempInput12, v) end  --temporal
    local dlst12 = clones1.rnn[CurrRnnIndx1]:backward(tempInput12, drnn_state1[CurrRnnIndx1])
    
    for index = 1, #init_state1 do
      drnn_state1[preRnnIndex][index] = drnn_state1[preRnnIndex][index] + dlst12[index+1]          --temporal
    end
    
    ---------------------------
    local doutput_t = clones1.criterion[CurrRnnIndx1-1]:backward(predictions1[CurrRnnIndx1-1], y[{{},{2}}])
    table.insert(drnn_state1[CurrRnnIndx1-1], doutput_t) -- drnn includes two part: 1) from t + 1, 2) from criterion
    assert (#(drnn_state1[CurrRnnIndx1-1]) == (#init_state1)+1)
    local inputX11 = x[{{},{t},{4,5}}]:contiguous():view(opt.batch_size, input_size)

    local tempInput11 = {}
    table.insert(tempInput11, inputX11)
    --if preRnnIndex == 0 then random_list(init_state_global) end
    if t == 1 then              
      preRnnIndex = 0
    else
      preRnnIndex = CurrRnnIndx1-1-2
    end
    for k, v in ipairs(rnn_state1[preRnnIndex]) do table.insert(tempInput11, v) end  --temporal
    local dlst11 = clones1.rnn[CurrRnnIndx1-1]:backward(tempInput11, drnn_state1[CurrRnnIndx1-1])

    for index = 1, #init_state1 do
      drnn_state1[preRnnIndex][index] = drnn_state1[preRnnIndex][index] + dlst11[index+1]     --temporal
    end

    CurrRnnIndx3 = CurrRnnIndx3 - 1
    CurrRnnIndx2 = CurrRnnIndx2 - 2
    CurrRnnIndx1 = CurrRnnIndx1 - 2
  end

------------------------ misc ----------------------
-- transfer final state to initial state (BPTT)
  init_state_global11 = clone_list(rnn_state1[1]) --
  init_state_global12 = clone_list(rnn_state1[2]) --
  init_state_global21 = clone_list(rnn_state2[1]) --
  init_state_global22 = clone_list(rnn_state2[2]) --
  init_state_global3 = clone_list(rnn_state3[1]) --
  grad_params:clamp(-opt.grad_clip, opt.grad_clip)
  return loss, grad_params
end

--retrieve batch in training set
function retrieve_batch(batch_number, num)  --num == 1 for training set, 2 for testing set

  local startindex = 1 + (batch_number - 1) * opt.batch_size
  local endindex = batch_number * opt.batch_size

  if num == 1 then
    x_batches = trainX[{{startindex, endindex}, {}, {}}]
    y_batches = trainy[{{startindex, endindex}, {}}]
  elseif num == 2 then
    x_batches = testX[{{startindex, endindex}, {}, {}}]
    y_batches = testy[{{startindex, endindex}, {}}]
  end

  return x_batches, y_batches -- valid_samples_count
end

-- -----------------------------------------------------start optimization here-------------------------------------------------------
train_losses = {}
train_losses_epoch = {}
val_losses = {}
local optim_state = {learningRate = opt.learning_rate, alpha = opt.decay_rate}
local iterations = opt.max_epochs * torch.floor(train_size/opt.batch_size)
local loss0 = nil
local lastepoch
random_list(init_state_global11)
random_list(init_state_global12)
random_list(init_state_global21)
random_list(init_state_global22)
random_list(init_state_global3)

for i = (opt.startingiter or 1), (iterations + 1) do
  local epoch = i / torch.floor(train_size/opt.batch_size)
  local lastepoch = (i - 1) / torch.floor(train_size/opt.batch_size)

  if torch.floor(epoch) > torch.floor(lastepoch) then -- it's a new epoch!
    --random_list(init_state_global)
    current_training_batch_number = 0
    table.insert(train_losses_epoch, train_losses[#train_losses])
    local val_loss = eval_split(2) -- 2 = validation
    table.insert(val_losses, val_loss)
  end

  if i <= iterations then
    local timer = torch.Timer()
    current_training_batch_number = current_training_batch_number + 1
    local _, loss = optim.rmsprop(feval, params, optim_state)
    --local _, loss = optim.adam(feval, params, optim_state)
        if opt.accurate_gpu_timing == 1 and opt.gpuid >= 0 then
			cutorch.synchronize()
		end
    local time = timer:time().real

    local train_loss = loss[1] -- the loss is inside a list, pop it
    train_losses[i] = train_loss

    -- exponential learning rate decay
    if i % torch.floor(train_size/opt.batch_size) == 0 and opt.learning_rate_decay < 1 then
      if epoch >= opt.learning_rate_decay_after then
        local decay_factor = opt.learning_rate_decay
        optim_state.learningRate = optim_state.learningRate * decay_factor -- decay it
        print('current learning rate ' .. optim_state.learningRate)
      end
    end

    if i % opt.print_every == 0 then
      print(string.format("%d/%d (epoch %.3f), train_loss = %6.8f, grad/param norm = %6.4e, time/batch = %.4fs", i, iterations, epoch, train_loss, grad_params:norm() / params:norm(), time))
    end

    if i % 10 == 0 then collectgarbage() end

    -- handle early stopping if things are going really bad
    if loss[1] ~= loss[1] then
      print('loss is NaN.')
      break -- halt
    end
    if loss0 == nil then loss0 = loss[1] end
    if loss[1] > loss0 * 3 then
      print('loss is exploding, aborting.')
      break -- halt
    end
  end
end

----------------------------------------------------start test here-----------------------------------------------------
local current_testing_batch_number = 0
pred_results = torch.CudaTensor(test_size,1)
local loss = 0
local rnn_state1 = {[0] = init_state_global11}
table.insert(rnn_state1, init_state_global12)
local rnn_state2 = {[0] = init_state_global21}
table.insert(rnn_state2, init_state_global22)
local rnn_state3 = {[0] = init_state_global3}
--random_list(init_state_global)
for i = 1, torch.floor(test_size/opt.batch_size) do
  collectgarbage()
  current_testing_batch_number = current_testing_batch_number + 1
  local x, y = retrieve_batch(current_testing_batch_number,2)
  x, y = prepro(x, y)
------------------- forward pass -------------------
  local predictions1 = {}
  local predictions2 = {}
  local predictions3 = {}        

  local CurrRnnIndx1 = 0
  local CurrRnnIndx2 = 0
  local CurrRnnIndx3 = 0 
  for t = 1, opt.seq_length do
    CurrRnnIndx1 = CurrRnnIndx1 + 1
    CurrRnnIndx2 = CurrRnnIndx2 + 1
    CurrRnnIndx3 = CurrRnnIndx3 + 1
    clones1.rnn[CurrRnnIndx1]:evaluate() 
    clones1.rnn[CurrRnnIndx1+1]:evaluate()
    clones2.rnn[CurrRnnIndx2]:evaluate()
    clones2.rnn[CurrRnnIndx2+1]:evaluate()
    clones3.rnn[CurrRnnIndx3]:evaluate()
    
--for rnn type1-------------------------------------
    local inputX11 = x[{{},{t},{4,5}}]:contiguous():view(opt.batch_size, input_size)
    local tempInput11 = {}
    table.insert(tempInput11, inputX11)
    --define preRnnIndex
    if t == 1 then
      preRnnIndex = 0
    else
      preRnnIndex = CurrRnnIndx1 - 2
    end
    for k, v in ipairs(rnn_state1[preRnnIndex]) do table.insert(tempInput11, v) end
    local lst11 = clones1.rnn[CurrRnnIndx1]:forward(tempInput11)
    rnn_state1[CurrRnnIndx1] = {}
    for index = 1, #init_state1 do table.insert(rnn_state1[CurrRnnIndx1], lst11[index]) end
    predictions1[CurrRnnIndx1] = lst11[#lst11]

    CurrRnnIndx1 = CurrRnnIndx1 + 1
    local inputX12 = x[{{},{t},{7,8}}]:contiguous():view(opt.batch_size, input_size)
    local tempInput12 = {}
    table.insert(tempInput12, inputX12)
    if t == 1 then              
      preRnnIndex = 1
    else
      preRnnIndex = CurrRnnIndx1 - 2
    end
    for k, v in ipairs(rnn_state1[preRnnIndex]) do table.insert(tempInput12, v) end
    local lst12 = clones1.rnn[CurrRnnIndx1]:forward(tempInput12)
    rnn_state1[CurrRnnIndx1] = {}
    for index = 1, #init_state1 do table.insert(rnn_state1[CurrRnnIndx1], lst12[index]) end
    predictions1[CurrRnnIndx1] = lst12[#lst12]

    --for rnn type2--------------------------------------
    local inputX21 = x[{{},{t},{7,8}}]:contiguous():view(opt.batch_size, input_size)
    local tempInput21 = {}
    table.insert(tempInput21, inputX21)
    --define preRnnIndex
    if t == 1 then
      preRnnIndex = 0
    else
      preRnnIndex = CurrRnnIndx2 - 2
    end
    
    local control_gate = torch.zeros(opt.batch_size,opt.rnn_size):float():cuda()
    --local control_gate = torch.ones(opt.batch_size,opt.rnn_size):float():cuda()
    for k = 1, opt.batch_size do
      if x[{k,t,6}] < x[{k,t,9}] then
        control_gate[{{k},{}}] = torch.ones(1,opt.rnn_size):float():cuda()
      end
    end
    table.insert(tempInput21, control_gate)
    
    for k, v in ipairs(rnn_state2[preRnnIndex]) do table.insert(tempInput21, v) end  --temporal component
    for k, v in ipairs(rnn_state1[CurrRnnIndx1-1]) do table.insert(tempInput21, v) end  --spatial component
    local lst21 = clones2.rnn[CurrRnnIndx2]:forward(tempInput21)
    rnn_state2[CurrRnnIndx2] = {}
    for index = 1, #init_state2 do table.insert(rnn_state2[CurrRnnIndx2], lst21[index]) end
    predictions2[CurrRnnIndx2] = lst21[#lst21]

    CurrRnnIndx2 = CurrRnnIndx2 + 1
    local inputX22 = x[{{},{t},{4,5}}]:contiguous():view(opt.batch_size, input_size)
    local tempInput22 = {}
    table.insert(tempInput22, inputX22)
    --define preRnnIndex
    if t == 1 then              
      preRnnIndex = 1
    else
      preRnnIndex = CurrRnnIndx2 - 2
    end
    
    local control_gate = torch.zeros(opt.batch_size,opt.rnn_size):float():cuda()
    for k = 1, opt.batch_size do
      if x[{k,t,6}] > x[{k,t,9}] then
        control_gate[{{k},{}}] = torch.ones(1,opt.rnn_size):float():cuda()
      end
    end
    table.insert(tempInput22, control_gate)
    
    for k, v in ipairs(rnn_state2[preRnnIndex]) do table.insert(tempInput22, v) end  --temporal component
    for k, v in ipairs(rnn_state1[CurrRnnIndx1]) do table.insert(tempInput22, v) end  --spatial component
    local lst22 = clones2.rnn[CurrRnnIndx2]:forward(tempInput22)
    rnn_state2[CurrRnnIndx2] = {}
    for index = 1, #init_state2 do table.insert(rnn_state2[CurrRnnIndx2], lst22[index]) end
    predictions2[CurrRnnIndx2] = lst22[#lst22]

    --for rnn type3----------------------------------
    local inputX3 = x[{{},{t},{1,2}}]:contiguous():view(opt.batch_size, input_size)
    local tempInput3 = {}
    table.insert(tempInput3, inputX3)
    preRnnIndex = CurrRnnIndx3 - 1
    
    local control_gate1 = torch.zeros(opt.batch_size, opt.rnn_size):float():cuda()
    local control_gate2 = torch.zeros(opt.batch_size, opt.rnn_size):float():cuda()

    for k = 1, opt.batch_size do
      if x[{k,t,6}] < x[{k,t,9}] then
        control_gate1[{{k},{}}] = torch.ones(1, opt.rnn_size):float():cuda()
      else
        control_gate2[{{k},{}}] = torch.ones(1, opt.rnn_size):float():cuda()
      end
    end
    table.insert(tempInput3, control_gate1)
    table.insert(tempInput3, control_gate2)
    
    for k, v in ipairs(rnn_state2[CurrRnnIndx2-1]) do table.insert(tempInput3, v) end --spatial1
    for k, v in ipairs(rnn_state2[CurrRnnIndx2]) do table.insert(tempInput3, v) end --spatial2
    for k, v in ipairs(rnn_state3[preRnnIndex]) do table.insert(tempInput3, v) end  --temporal
    local lst3 = clones3.rnn[CurrRnnIndx3]:forward(tempInput3)
    rnn_state3[CurrRnnIndx3] = {}
    for index = 1, #init_state3 do table.insert(rnn_state3[CurrRnnIndx3], lst3[index]) end
    predictions3[CurrRnnIndx3] = lst3[#lst3]
  end
  rnn_state1[0] = rnn_state1[1]
  rnn_state1[1] = rnn_state1[2]
  rnn_state2[0] = rnn_state2[1]
  rnn_state2[1] = rnn_state2[2]
  rnn_state3[0] = rnn_state3[1]
  loss = loss + clones3.criterion[CurrRnnIndx3]:forward(predictions3[CurrRnnIndx3], y[{{},{1}}])

  local startindex = 1 + (current_testing_batch_number - 1) * opt.batch_size
  local endindex = current_testing_batch_number * opt.batch_size
  pred_results[{{startindex,endindex},{}}] = predictions3[CurrRnnIndx3]
end

print(loss)
local testy = testy*std_v + mean_v
local pred_results = pred_results*std_v + mean_v

--write the result in csv file
subtensor = torch.cat(pred_results[{{1,test_size},{}}], testy[{{1,test_size},{1}}], 2)
local out = assert(io.open("./pred_results_stlstm.csv", "w")) -- open a file for serialization
splitter = ","
for i=1,subtensor:size(1) do
    for j=1,subtensor:size(2) do
        out:write(subtensor[i][j])
        if j == subtensor:size(2) then
            out:write("\n")
        else
            out:write(splitter)
        end
    end
end

out:close()
