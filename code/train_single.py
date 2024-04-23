import copy
import math

import torch
from dataloader import MovielensDatasetLoader
from model import NeuralCollaborativeFiltering
import numpy as np
from tqdm import tqdm
from metrics import compute_metrics
import pandas as pd


# 一个batch里面四分之一正，四分之三负
class MatrixLoader:
	def __init__(self, ui_matrix, default=None, seed=0):
		np.random.seed(seed)
		self.ui_matrix = ui_matrix
		self.positives = np.argwhere(self.ui_matrix!=0)
		self.negatives = np.argwhere(self.ui_matrix==0)
		if default is None:
			self.default = np.array([[0, 0]]), np.array([0])
		else:
			self.default = default

	def delete_indexes(self, indexes, arr="pos"):
		if arr=="pos":
			self.positives = np.delete(self.positives, indexes, 0)
		else:
			self.negatives = np.delete(self.negatives, indexes, 0)

	def get_batch(self, batch_size):
		if self.positives.shape[0]<batch_size//4 or self.negatives.shape[0]<batch_size-batch_size//4:
			return torch.tensor(self.default[0]), torch.tensor(self.default[1])
		try:
			pos_indexes = np.random.choice(self.positives.shape[0], batch_size//4)
			neg_indexes = np.random.choice(self.negatives.shape[0], batch_size - batch_size//4)
			pos = self.positives[pos_indexes]
			neg = self.negatives[neg_indexes]
			self.delete_indexes(pos_indexes, "pos")
			self.delete_indexes(neg_indexes, "neg")
			batch = np.concatenate((pos, neg), axis=0)
			if batch.shape[0]!=batch_size:
				return torch.tensor(self.default[0]), torch.tensor(self.default[1]).float()
			np.random.shuffle(batch)
			y = np.array([self.ui_matrix[i][j] for i,j in batch])
			#batch 表示一个位置，类似于一堆i,j坐标，y表示对应的i,j坐标的值
			return torch.tensor(batch), torch.tensor(y).float()
		except:
			return torch.tensor(self.default[0]), torch.tensor(self.default[1]).float()

class NCFTrainer:
	def __init__(self, ui_matrix, start_user_id, epochs, batch_size, latent_dim=32, device=None):
		self.ui_matrix = ui_matrix
		self.start_user_id = 0
		self.last_user_id = self.start_user_id + self.ui_matrix.shape[0]-1
		self.epochs = epochs
		self.latent_dim = latent_dim
		self.batch_size = batch_size
		self.loader = None
		self.initialize_loader()
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		self.ncf = NeuralCollaborativeFiltering(self.ui_matrix.shape[0], self.ui_matrix.shape[1], self.latent_dim).to(self.device)

		self.negativeList,self.test_positive1item_list = self.generate_negetive_list() #生成用于测试的99个负样例 每个用户一个list
		# 混入99个负样例中的一个正样例 每个用户1个样例
		self.combine_list = copy.deepcopy(self.negativeList)
		for l1, l2 in zip(self.combine_list , self.test_positive1item_list):
			l1=l1.extend([l2])
		# self.combine_list = self.negativeList+self.test_positive1item_list #100个测试sample
		# print(self.negativeList)
		# print(self.test_positive1item_list)


	def generate_negetive_list(self):
		negativeList = []
		positive1itemlist=[]
		with open("./Data/ml-1m.test.negative", "r") as f:
			line = f.readline()
			count = 0;
			while line != None and line != "":
				if count<self.start_user_id:
					count=count+1
					line = f.readline()
					continue
				elif self.start_user_id <= count <= self.last_user_id:
					arr = line.split("\t")
					negatives = []
					positive1item = arr[0]
					for x in arr[1:]:
						negatives.append([int(count),int(x)])
					negativeList.append(negatives)
					positive1itemlist.append(positive1item)
					line = f.readline()
					count=count+1
				else:
					break;
		trans_positive1itemlist = [list(eval(item)) for item in positive1itemlist] #将字符串格式的对子改成list
		return negativeList,trans_positive1itemlist

	def initialize_loader(self):
		self.loader = MatrixLoader(self.ui_matrix)

	def train_batch(self, x, y, optimizer):
		optimizer.zero_grad()
		y_ = self.ncf(x)
		mask = (y>0).float()
		loss = torch.nn.functional.mse_loss(y_.view(-1)*mask, y)
		loss.backward()
		optimizer.step()
		# print(self.ncf.output_logits.weight)
		# print(self.ncf)
		return loss.item(), y_.detach()

	#还需改
	def model_eval(self,N):
		self.ncf.eval()
		# x, y = x.int(), y.float()
		# x, y = x.to(self.device), y.to(self.device)
		right=0
		ndcg_list = []
		with torch.no_grad():
			for i in range(self.start_user_id,self.last_user_id+1):
				x=copy.deepcopy(self.combine_list[i-self.start_user_id])
				x = np.array(x)
				actual_index = 99
				#actual_index = np.where(np.all( x == self.test_positive1item_list[i], axis=1))
				#actual_index = np.where(x == self.test_positive1item_list[i])
				x = torch.tensor(x)
				# 一开始，我们创建一个和t形状相同，全部填充为0的张量
				n_tensor = torch.zeros_like(x)
				# 然后我们将n填充到特定的位置，例如每个小张量的第一个元素位置
				n_tensor[:, 0] = i
				x=x-n_tensor
				x=x.int()
				x=x.to(self.device)
				y = self.ncf(x)
				y= y.cpu().numpy()
				mask = np.zeros_like(y)
				mask[y > 0] = 1
				y=y*mask
				y=y.reshape(-1)
				predict_topk_indexes = np.argsort(y)[-N:][::-1] #预测的topk的index值
				# actual_index = np.where(y == self.test_positive1item_list[i])
				if actual_index in predict_topk_indexes:
					right=right+1
				ndcg = 0
				for i in range(len(predict_topk_indexes)):
					item = predict_topk_indexes[i]
					if item == 99:
						ndcg = math.log(2) / math.log(i + 2)
						break
				ndcg_list.append(ndcg)
			hit_ratio = right/self.ui_matrix.shape[0]
			ndcg_avg = np.array(ndcg_list).mean()

		return hit_ratio,ndcg_avg

	def train_model(self, optimizer, epochs=None, print_num=10):
		epoch = 0
		progress = {"epoch": [], "loss": [], "hit_ratio@10": [], "ndcg@10": []}
		running_loss, running_hr, running_ndcg = 0, 0, 0
		prev_running_loss, prev_running_hr, prev_running_ndcg = 0, 0, 0
		if epochs is None:
			epochs = self.epochs
		steps, prev_steps, prev_epoch = 0, 0, 0
		while epoch<epochs:
			x, y = self.loader.get_batch(self.batch_size)
			if x.shape[0]<self.batch_size:
				prev_running_loss, prev_running_hr, prev_running_ndcg = running_loss, running_hr, running_ndcg
				running_loss = 0
				running_hr = 0
				running_ndcg = 0
				prev_steps = steps
				steps = 0
				epoch += 1
				self.initialize_loader()
				x, y = self.loader.get_batch(self.batch_size)
			x, y = x.int(), y.float()
			x, y = x.to(self.device), y.to(self.device)
			loss, y_ =	self.train_batch(x, y, optimizer)
			hr, ndcg = compute_metrics(y.cpu().numpy(), y_.cpu().numpy())
			running_loss += loss
			running_hr += hr
			running_ndcg += ndcg
			if epoch!=0 and steps==0:
				results = {"epoch": prev_epoch, "loss": prev_running_loss/(prev_steps+1), "hit_ratio@10": prev_running_hr/(prev_steps+1), "ndcg@10": prev_running_ndcg/(prev_steps+1)}
			else:
				results = {"epoch": prev_epoch, "loss": running_loss/(steps+1), "hit_ratio@10": running_hr/(steps+1), "ndcg@10": running_ndcg/(steps+1)}
			steps += 1
			if prev_epoch!=epoch:
				progress["epoch"].append(results["epoch"])
				progress["loss"].append(results["loss"])
				progress["hit_ratio@10"].append(results["hit_ratio@10"])
				progress["ndcg@10"].append(results["ndcg@10"])
				prev_epoch+=1

		hit_ratio,ndcg=self.model_eval(10)
		print("hit_ratio:",hit_ratio)
		print("ndcg:",ndcg)
		r_results = {"num_users": self.ui_matrix.shape[0]}
		r_results.update({i:results[i] for i in ["loss", "hit_ratio@10", "ndcg@10"]})
		return r_results, progress

	def train(self, ncf_optimizer, return_progress=False):
		self.ncf.join_output_weights()
		results, progress = self.train_model(ncf_optimizer)
		if return_progress:
			return results, progress
		else:
			return results

if __name__ == '__main__':
	dataloader = MovielensDatasetLoader()
	trainer = NCFTrainer(dataloader.ratings[:50], epochs=20, batch_size=128)
	ncf_optimizer = torch.optim.Adam(trainer.ncf.parameters(), lr=1e-3)
	_, progress = trainer.train(ncf_optimizer, return_progress=True)