#!/usr/bin/env th
--[[
this script reads a large csv file and converts it to torch format.
Easier and simpler to modify than other options. Courtesy
http://blog.aicry.com/torch7-reading-csv-into-tensor/
--]]

require 'torch'
require './torchConfig'

-- Split string
function string:split(sep)
  local sep, fields = sep, {}
  local pattern = string.format("([^%s]+)", sep)
  self:gsub(pattern, function(substr) fields[#fields + 1] = substr end)
  return fields
end

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Options')
cmd:option('-ipfilepath','./proc_data/datafile.csv','filename of the csv file processed in python')
cmd:option('-opfilepath','./torch_data/datafile.csv','filename of the output file')
cmd:text()

-- parse input params
local opt = cmd:parse(arg)

local filePath = opt.ipfilepath

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

-- Serialize tensor
local outputFilePath = opt.opfilepath
torch.save(outputFilePath, data)
