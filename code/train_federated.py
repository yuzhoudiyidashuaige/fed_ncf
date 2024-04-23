import math
import os

import torch
from train_single import NCFTrainer
from dataloader import MovielensDatasetLoader
import random
from tqdm import tqdm
from server_model import ServerNeuralCollaborativeFiltering
import copy

import numpy as np
import sklearn.metrics.pairwise as smp

class Utils:
	def __init__(self, num_clients, local_path="./models/local_items/", server_path="./models/central/"):
		self.epoch = 0
		self.num_clients = num_clients
		#改成local文件夹
		self.local_path = "./models/local/"
		self.server_path = server_path
		self.memory = None
		self.user_number = 200
		self.negativeList, self.test_positive1item_list = self.generate_negetive_list()  # 生成用于测试的99个负样例 每个用户一个list
		# 混入99个负样例中的一个正样例 每个用户1个样例
		self.combine_list = copy.deepcopy(self.negativeList)
		for l1, l2 in zip(self.combine_list, self.test_positive1item_list):
			l1 = l1.extend([l2])

	def load_pytorch_client_model(self, path):
		return torch.jit.load(path)

	def get_user_models(self, loader):
		models = []
		for client_id in range(self.num_clients):
			models.append({'model':loader(self.local_path+"dp"+str(client_id)+".pt")})
		return models

	def get_previous_federated_model(self):
		self.epoch += 1
		return torch.jit.load(self.server_path+"server"+str(self.epoch-1)+".pt")

	def save_federated_model(self, model):
		torch.jit.save(model, self.server_path+"server"+str(self.epoch)+".pt")

	def global_model_eval(self, model, N):
		model.eval()
		# x, y = x.int(), y.float()
		# x, y = x.to(self.device), y.to(self.device)
		right = 0
		ndcg_list = []
		with torch.no_grad():
			for i in range(0, 200):
				x = copy.deepcopy(self.combine_list[i])
				x = np.array(x)
				actual_index = 99
				# actual_index = np.where(np.all( x == self.test_positive1item_list[i], axis=1))
				# actual_index = np.where(x == self.test_positive1item_list[i])
				x = torch.tensor(x)
				# 一开始，我们创建一个和t形状相同，全部填充为0的张量
				n_tensor = torch.zeros_like(x)
				# 然后我们将n填充到特定的位置，例如每个小张量的第一个元素位置
				n_tensor[:, 0] = i
				x = x - n_tensor
				x = x.int()
				x = x.to("cpu")
				y = model(x)
				y = y.cpu().numpy()
				mask = np.zeros_like(y)
				mask[y > 0] = 1
				y = y * mask
				y = y.reshape(-1)
				predict_topk_indexes = np.argsort(y)[-N:][::-1]  # 预测的topk的index值
				# actual_index = np.where(y == self.test_positive1item_list[i])
				if actual_index in predict_topk_indexes:
					right = right + 1
				ndcg = 0
				for i in range(len(predict_topk_indexes)):
					item = predict_topk_indexes[i]
					if item == 99:
						ndcg = math.log(2) / math.log(i + 2)
						break
				ndcg_list.append(ndcg)
			hit_ratio = right / self.user_number
			ndcg_avg = np.array(ndcg_list).mean()

		return hit_ratio, ndcg_avg

	def generate_negetive_list(self):
		negativeList = []
		positive1itemlist = []
		with open("./Data/ml-1m.test.negative", "r") as f:
			line = f.readline()
			count = 0;
			while line != None and line != "":
				if count < 0:
					count = count + 1
					line = f.readline()
					continue
				elif 0 <= count <= self.user_number-1:
					arr = line.split("\t")
					negatives = []
					positive1item = arr[0]
					for x in arr[1:]:
						negatives.append([int(count), int(x)])
					negativeList.append(negatives)
					positive1itemlist.append(positive1item)
					line = f.readline()
					count = count + 1
				else:
					break;
		trans_positive1itemlist = [list(eval(item)) for item in positive1itemlist]  # 将字符串格式的对子改成list
		return negativeList, trans_positive1itemlist

def federate(utils):
	client_models = utils.get_user_models(utils.load_pytorch_client_model)
	server_model = utils.get_previous_federated_model()
	if len(client_models) == 0:
		utils.save_federated_model(server_model)
		return
	n = len(client_models)
	server_old_dict = copy.deepcopy(server_model.state_dict())
	clients_grads = []
	for i in range(0, len(client_models)):
		client_grad=copy.deepcopy(client_models[i]['model'].state_dict())
		for k in server_old_dict.keys():
			client_grad[k] -= server_old_dict[k]
		clients_grads.append(client_grad)

	# print(clients_grads[0])
	grad_len = np.array(clients_grads[0].get('output_logits.weight').data.cpu().numpy().shape).prod()
	grads = np.zeros((n, grad_len))
	if utils.memory is None:
		utils.memory = np.zeros((n, grad_len))

	for i in range(len(clients_grads)):
		grads[i] = np.reshape(clients_grads[i].get('output_logits.weight').data.cpu().numpy(), (grad_len))
	utils.memory += grads

	# wv = foolsgold(utils.memory) # 等待替换为memory
	wv=np.zeros(n)
	wv.fill(1)
	print("wv: ",wv)
	server_new_dict = copy.deepcopy(server_old_dict)
	for i in range(0, len(client_models)):
		client_dict = copy.deepcopy(client_models[i]['model'].state_dict())
		for k in client_dict.keys():
			server_new_dict[k] += wv[i]*client_dict[k]

	for k in server_new_dict.keys():
		server_new_dict[k] = server_new_dict[k] / n
	server_model.load_state_dict(server_new_dict)

	hit_ratio, ndcg = utils.global_model_eval(server_model,10)
	print("global_hit_ratio:", hit_ratio)
	print("global_ndcg:", ndcg)
	utils.save_federated_model(server_model)

def foolsgold(clients_grads):
	n_clients = clients_grads.shape[0]
	modified_clients_grads = [vec if np.any(vec) else np.full_like(vec, 0.001) for vec in clients_grads]
	# 计算这个新矩阵的余弦相似度
	cs = smp.cosine_similarity(modified_clients_grads) - np.eye(n_clients)
	maxcs = np.max(cs, axis=1)
	# pardoning
	for i in range(n_clients):
		for j in range(n_clients):
			if i == j:
				continue
			if maxcs[i] < maxcs[j]:
				cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]
	wv = 1 - (np.max(cs, axis=1))
	wv[wv > 1] = 1
	wv[wv < 0] = 0

	# Rescale so that max value is wv
	wv = wv / np.max(wv)
	wv[(wv == 1)] = .99

	# Logit function
	wv = (np.log(wv / (1 - wv)) + 0.5)
	wv[(np.isinf(wv) + wv > 1)] = 1
	wv[(wv < 0)] = 0

	return wv

class FederatedNCF:
	def __init__(self, ui_matrix, num_clients=50, user_per_client_range=[5, 10], mode="ncf", aggregation_epochs=50, local_epochs=10, batch_size=128, latent_dim=32, seed=0):
		random.seed(seed)
		self.ui_matrix = ui_matrix[:200]
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		self.num_clients = num_clients
		self.latent_dim = latent_dim
		self.user_per_client_range = user_per_client_range
		self.mode = mode
		self.aggregation_epochs = aggregation_epochs
		self.local_epochs = local_epochs
		self.batch_size = batch_size
		self.clients = self.generate_clients()
		self.ncf_optimizers = [torch.optim.Adam(client.ncf.parameters(), lr=1e-3) for client in self.clients]
		self.utils = Utils(self.num_clients)
		# self.generate_testmodel()

	def generate_clients(self):
		start_index = 0
		clients = []
		for i in range(self.num_clients):
			# users = random.randint(self.user_per_client_range[0], self.user_per_client_range[1])
			# 改成固定的10个
			users = 20
			# 创建一个和ui_matrix大小相同的全0数组
			ui_matrix = np.zeros_like(self.ui_matrix)
			# 将ui_matrix的指定切片复制到新的数组
			ui_matrix[start_index:start_index + users] = self.ui_matrix[start_index:start_index + users]
			# if i==0 or i==1:
			# 	poison_matrix=copy.deepcopy(self.ui_matrix[start_index:start_index+users])
			# 	# 将矩阵转换为numpy数组
			# 	poison_matrix = np.array(poison_matrix)
			#
			# 	# 将所有元素填充为5
			# 	# poison_matrix.fill(5)
			# 	# 将所有元素+1 5分的变成0分
			# 	poison_matrix = (poison_matrix + 1) % 6
			# 	clients.append(NCFTrainer(poison_matrix, epochs=self.local_epochs,
			# 							  batch_size=self.batch_size))
			# else:
			# 	clients.append(NCFTrainer(self.ui_matrix[start_index:start_index+users], epochs=self.local_epochs, batch_size=self.batch_size))
			clients.append(NCFTrainer(ui_matrix,start_user_id=start_index,epochs=self.local_epochs, batch_size=self.batch_size))
			start_index += users
		return clients

	# def generate_testmodel(self):
	# 	user_num=self.ui_matrix.shape(0)
	# 	start_index = random.randint(0,user_num-51)
	# 	testmodel = NCFTrainer(self.ui_matrix[start_index:start_index + 50], epochs=self.local_epochs,
	# 			   batch_size=self.batch_size)
	# 	ncf_optimizer = torch.optim.Adam(testmodel.ncf.parameters(), lr=1e-3)
	# 	for i in range (200):
	# 		testmodel.train(self.ncf_optimizer)
	#
	# 	model = torch.jit.script(testmodel.ncf.to(torch.device("cpu")))
	# 	os.makedirs("./models/test_model", exist_ok=True)
	# 	torch.jit.save(model, "./models/test_model/testmodel.pt")


	def single_round(self, epoch=0, first_time=False):
		single_round_results = {key:[] for key in ["num_users", "loss", "hit_ratio@10", "ndcg@10"]}
		bar = tqdm(enumerate(self.clients), total=self.num_clients)
		os.makedirs("./models/local", exist_ok=True)
		for client_id, client in bar:
			results = client.train(self.ncf_optimizers[client_id])
			for k,i in results.items():
				single_round_results[k].append(i)
			printing_single_round = {"epoch": epoch}
			printing_single_round.update({k:round(sum(i)/len(i), 4) for k,i in single_round_results.items()})
			model = torch.jit.script(client.ncf.to(torch.device("cpu")))
			torch.jit.save(model, "./models/local/dp"+str(client_id)+".pt")
			bar.set_description(str(printing_single_round))
		bar.close()

	def extract_item_models(self):
		os.makedirs("./models/local_items", exist_ok=True)
		for client_id in range(self.num_clients):
			model = torch.jit.load("./models/local/dp"+str(client_id)+".pt")
			item_model = ServerNeuralCollaborativeFiltering(item_num=self.ui_matrix.shape[1], predictive_factor=self.latent_dim)
			item_model.set_weights(model)
			item_model = torch.jit.script(item_model.to(torch.device("cpu")))
			torch.jit.save(item_model, "./models/local_items/dp"+str(client_id)+".pt")

	def train(self):
		first_time = True
		server_model = ServerNeuralCollaborativeFiltering(user_num=self.ui_matrix.shape[0],item_num=self.ui_matrix.shape[1], predictive_factor=self.latent_dim)
		server_model = torch.jit.script(server_model.to(torch.device("cpu")))
		# 确保目录存在
		os.makedirs("./models/central", exist_ok=True)
		torch.jit.save(server_model, "./models/central/server"+str(0)+".pt")
		for epoch in range(self.aggregation_epochs):
			server_model = torch.jit.load("./models/central/server"+str(epoch)+".pt", map_location=self.device)
			_ = [client.ncf.to(self.device) for client in self.clients]
			_ = [client.ncf.load_server_weights(server_model) for client in self.clients]
			self.single_round(epoch=epoch, first_time=first_time)
			first_time = False
			# self.extract_item_models()
			federate(self.utils)

if __name__ == '__main__':
	dataloader = MovielensDatasetLoader()
	fncf = FederatedNCF(dataloader.ratings, num_clients=10, user_per_client_range=[10, 20], mode="ncf", aggregation_epochs=50, local_epochs=10, batch_size=128)
	fncf.train()
	mlp_user_embeddings = torch.nn.Embedding(num_embeddings=10, embedding_dim=5)
	mlp_item_embeddings = torch.nn.Embedding(num_embeddings=10, embedding_dim=5)
	print(torch.cat(mlp_item_embeddings,mlp_user_embeddings,dim=1))