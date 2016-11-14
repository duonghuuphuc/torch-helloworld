require 'torch'
require 'nn'

-- MODEL
mlp = nn.Sequential();  -- make a multi-layer perceptron
input = 4
output = 2
hiddenLayer1 = 4
hiddenLayer2 = 4

mlp:add(nn.Linear(input, hiddenLayer1))
mlp:add(nn.Tanh())
mlp:add(nn.Linear(hiddenLayer1, hiddenLayer2))
mlp:add(nn.Tanh())
mlp:add(nn.Linear(hiddenLayer2, output))
mlp:add(nn.LogSoftMax())

---- Print the mlp to check the defined model (Optional)
print(mlp)

---- Test model with random data (Optional)
preTest = mlp:forward(torch.randn(1,4))
print(preTest)



-- TRAIN
criterion = nn.ClassNLLCriterion()
trainer = nn.StochasticGradient(mlp, criterion)
trainer.learningRate = 0.1



-- DATA
---- Processing the input data in CSV format
function string:splitAtCommas()
  local sep, output = ",", {}
  local pattern = string.format("([^%s]+)", sep)
  self:gsub(pattern, function(c) output[#output+1] = c end)
  return output
end

function loadData(dataFile)
  local dataset = {}
  local i = 1
  for line in io.lines(dataFile) do
    local values = line:splitAtCommas()
    local y = torch.Tensor(1)
    y[1] = values[#values]	-- the last number in line is class
    values[#values] = nil
    local x = torch.Tensor(values)	-- all other numbers are input
    dataset[i] = {x, y}
    i = i + 1
  end
  function dataset:size() return (i - 1) end -- the requirement mentioned
  return dataset
end

---- Load the dataset
dataset = loadData("train.csv")

---- Training model with given dataset
trainer:train(dataset)