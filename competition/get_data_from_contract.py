import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dutils.dataset import Dataset
from dutils.neural_network import NeuralNetwork
from web3 import Web3, HTTPProvider, IPCProvider
from matplotlib import pyplot as plt
import numpy as np

# 设置http超时时间为180s，不然程序会超时
import socket
timeout = 180
socket.setdefaulttimeout(timeout)

w_scale = 1000 # Scale up weights by 1000x
b_scale = 1000 # Scale up biases by 1000x

def scale_packed_data(data, scale):
    # Scale data and convert it to an integer
    return list(map(lambda x: int(x*scale), data))

print("Connecting to geth...\n")
web3 = Web3(HTTPProvider('http://localhost:8545'))
# web3 = Web3(HTTPProvider('http://93.85.92.250:8545'))
# 用web3.isConnected()判断是否连上了geth的8545的RPC端口
if web3.isConnected():
    print("connected!\n")
else:
    print("not connected!\n")

print("Connected to the geth node!\n")

abi = [{"constant":True,"inputs":[],"name":"init1_block_height","outputs":[{"name":"","type":"uint256"}],"payable":False,"stateMutability":"view","type":"function"},{"constant":False,"inputs":[],"name":"init2","outputs":[],"payable":False,"stateMutability":"nonpayable","type":"function"},{"constant":False,"inputs":[{"name":"submission_index","type":"uint256"}],"name":"evaluate_model","outputs":[],"payable":False,"stateMutability":"nonpayable","type":"function"},{"constant":True,"inputs":[{"name":"submission_index","type":"uint256"},{"name":"data","type":"int256[3][]"}],"name":"model_accuracy","outputs":[{"name":"","type":"int256"}],"payable":False,"stateMutability":"view","type":"function"},{"constant":True,"inputs":[],"name":"get_training_index","outputs":[{"name":"","type":"uint256[16]"}],"payable":False,"stateMutability":"view","type":"function"},{"constant":True,"inputs":[],"name":"evaluation_stage_block_size","outputs":[{"name":"","type":"uint256"}],"payable":False,"stateMutability":"view","type":"function"},{"constant":True,"inputs":[{"name":"","type":"uint256"},{"name":"","type":"uint256"}],"name":"test_data","outputs":[{"name":"","type":"int256"}],"payable":False,"stateMutability":"view","type":"function"},{"constant":True,"inputs":[],"name":"get_testing_index","outputs":[{"name":"","type":"uint256[4]"}],"payable":False,"stateMutability":"view","type":"function"},{"constant":False,"inputs":[{"name":"_test_data_groups","type":"int256[]"},{"name":"_test_data_group_nonces","type":"int256"}],"name":"reveal_test_data","outputs":[],"payable":False,"stateMutability":"nonpayable","type":"function"},{"constant":True,"inputs":[{"name":"paymentAddress","type":"address"},{"name":"num_neurons_input_layer","type":"uint256"},{"name":"num_neurons_output_layer","type":"uint256"},{"name":"num_neurons_hidden_layer","type":"uint256[]"},{"name":"weights","type":"int256[]"},{"name":"biases","type":"int256[]"}],"name":"get_submission_id","outputs":[{"name":"","type":"uint256"}],"payable":False,"stateMutability":"view","type":"function"},{"constant":True,"inputs":[],"name":"best_submission_index","outputs":[{"name":"","type":"uint256"}],"payable":False,"stateMutability":"view","type":"function"},{"constant":True,"inputs":[],"name":"use_test_data","outputs":[{"name":"","type":"bool"}],"payable":False,"stateMutability":"view","type":"function"},{"constant":False,"inputs":[{"name":"_train_data_groups","type":"int256[]"},{"name":"_train_data_group_nonces","type":"int256"}],"name":"init3","outputs":[],"payable":False,"stateMutability":"nonpayable","type":"function"},{"constant":True,"inputs":[{"name":"l_nn","type":"uint256[]"},{"name":"input_layer","type":"int256[]"},{"name":"hidden_layers","type":"int256[]"},{"name":"output_layer","type":"int256[]"},{"name":"weights","type":"int256[]"},{"name":"biases","type":"int256[]"}],"name":"forward_pass2","outputs":[{"name":"","type":"int256[]"}],"payable":False,"stateMutability":"pure","type":"function"},{"constant":False,"inputs":[{"name":"_hashed_data_groups","type":"bytes32[20]"},{"name":"accuracy_criteria","type":"int256"},{"name":"organizer_refund_address","type":"address"}],"name":"init1","outputs":[],"payable":False,"stateMutability":"nonpayable","type":"function"},{"constant":True,"inputs":[],"name":"organizer","outputs":[{"name":"","type":"address"}],"payable":False,"stateMutability":"view","type":"function"},{"constant":True,"inputs":[],"name":"init_level","outputs":[{"name":"","type":"uint256"}],"payable":False,"stateMutability":"view","type":"function"},{"constant":True,"inputs":[{"name":"","type":"uint256"}],"name":"testing_partition","outputs":[{"name":"","type":"uint256"}],"payable":False,"stateMutability":"view","type":"function"},{"constant":True,"inputs":[],"name":"get_train_data_length","outputs":[{"name":"","type":"uint256"}],"payable":False,"stateMutability":"view","type":"function"},{"constant":True,"inputs":[],"name":"best_submission_accuracy","outputs":[{"name":"","type":"int256"}],"payable":False,"stateMutability":"view","type":"function"},{"constant":False,"inputs":[],"name":"finalize_contract","outputs":[],"payable":False,"stateMutability":"nonpayable","type":"function"},{"constant":True,"inputs":[],"name":"contract_terminated","outputs":[{"name":"","type":"bool"}],"payable":False,"stateMutability":"view","type":"function"},{"constant":True,"inputs":[],"name":"init3_block_height","outputs":[{"name":"","type":"uint256"}],"payable":False,"stateMutability":"view","type":"function"},{"constant":True,"inputs":[],"name":"get_submission_queue_length","outputs":[{"name":"","type":"uint256"}],"payable":False,"stateMutability":"view","type":"function"},{"constant":False,"inputs":[{"name":"payment_address","type":"address"},{"name":"num_neurons_input_layer","type":"uint256"},{"name":"num_neurons_output_layer","type":"uint256"},{"name":"num_neurons_hidden_layer","type":"uint256[]"},{"name":"weights","type":"int256[]"},{"name":"biases","type":"int256[]"}],"name":"submit_model","outputs":[],"payable":False,"stateMutability":"nonpayable","type":"function"},{"constant":True,"inputs":[{"name":"","type":"uint256"}],"name":"training_partition","outputs":[{"name":"","type":"uint256"}],"payable":False,"stateMutability":"view","type":"function"},{"constant":True,"inputs":[],"name":"reveal_test_data_groups_block_size","outputs":[{"name":"","type":"uint256"}],"payable":False,"stateMutability":"view","type":"function"},{"constant":False,"inputs":[],"name":"cancel_contract","outputs":[],"payable":False,"stateMutability":"nonpayable","type":"function"},{"constant":True,"inputs":[],"name":"model_accuracy_criteria","outputs":[{"name":"","type":"int256"}],"payable":False,"stateMutability":"view","type":"function"},{"constant":True,"inputs":[{"name":"","type":"uint256"},{"name":"","type":"uint256"}],"name":"train_data","outputs":[{"name":"","type":"int256"}],"payable":False,"stateMutability":"view","type":"function"},{"constant":True,"inputs":[],"name":"get_test_data_length","outputs":[{"name":"","type":"uint256"}],"payable":False,"stateMutability":"view","type":"function"},{"constant":True,"inputs":[],"name":"submission_stage_block_size","outputs":[{"name":"","type":"uint256"}],"payable":False,"stateMutability":"view","type":"function"},{"inputs":[],"payable":False,"stateMutability":"nonpayable","type":"constructor"},{"payable":True,"stateMutability":"payable","type":"fallback"}]

contract_tx = "0x9A0991fc223dFFE420e08f15b88a593a3b8D44B8"

# Get contract instance
danku = web3.eth.contract(abi, contract_tx)

print("Downloading training data from the contract...\n")
# Get training data
contract_train_data_length = danku.call().get_train_data_length()
print("train_data_length:", contract_train_data_length)
contract_train_data = []
# 第i行数据
for i in range(contract_train_data_length):
    for j in range(3):
        # 两维特征一维标签
        contract_train_data.append(danku.call().train_data(i,j))
ds = Dataset()
# dps: data point size
ds.dps = 3
# 依据dps,unpacke data
contract_train_data = ds.unpack_data(contract_train_data)
print("Download finished!\n")
print("Contract training data:\n" + str(contract_train_data) + "\n")

# Visualize the training data
print("Visualizing training data...\n")
scatter_x = np.array(list(map(lambda x: x[1:2][0], contract_train_data)))
scatter_y = np.array(list(map(lambda x: x[:1][0], contract_train_data)))
group = np.array(list(map(lambda x: x[2:3][0], contract_train_data)))
cdict = {0: "blue", 1: "red"}

names = []
names.append("Democrat")
names.append("Republican")

fig, ax = plt.subplots()
for g in np.unique(group):
    ix = np.where(group == g)
    ax.scatter(scatter_x[ix], scatter_y[ix], c = cdict[g], label = names[g], s = 4)
ax.legend()
plt.title("Training data points")
plt.show()
