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

function loadData(filePath)
  local dataset = {}
  local i = 1
  for line in io.lines(filePath) do
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



-- PREDICTION AND EVALUATION
function argmax(v)
  local max = torch.max(v)
  for i = 1, v:size(1) do
    if v[i] == max then
      return i
    end
  end
end

function evaluation(filePath)
  local total = 0
  local positive = 0

  for line in io.lines(filePath) do
    local values = line:splitAtCommas()
    local y = torch.Tensor(1)
    y[1] = values[#values]
    values[#values] = nil
    local x = torch.Tensor(values)
    local prediction = argmax(mlp:forward(x))
    if math.floor(prediction) == math.floor(y[1]) then
      positive = positive + 1
    end
    total = total + 1
  end

  return (positive / total) * 100
end

---- Read the testset and compute the accuracy
accuracy = evaluation("test.csv")
print("Accuracy(%) is " .. accuracy)



--[[
-- Print the weight matrix
print("Weights of saved model: ")
print(mlp:get(1)) -- Get the first module of our model, i.e. nn.Linear(4 -> 4)
print(mlp:get(1).weight)  -- Get the weight matrix of that layer


-- Save the trained model
torch.save("file.th", mlp)

-- Load the saved model
mlp2 = torch.load("file.th")
print(mlp2:get(1).weight)
--]]
