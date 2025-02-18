import pandas as pd
import torch
import torch.nn as nn
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

def binary_cross_entropy_loss(y_pred, y_true):
  loss = -torch.mean(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))
  return loss

ratings = pd.read_csv("ratings.csv")

# filtering out rare users -- less than 5 appearances
user_counts = ratings["userId"].value_counts()
filtered_ratings = ratings[ratings["userId"].isin(user_counts[user_counts >= 5].index)]

user = filtered_ratings["userId"]
movie = filtered_ratings["movieId"]
ratings = filtered_ratings["rating"]

user = torch.tensor(user)
movie = torch.tensor(movie)
ratings = torch.tensor(ratings, dtype=torch.float)
labels = (ratings >= 3).float()

unique_user_ids = torch.unique(user)
unique_movie_ids = torch.unique(movie)

# map ids to contiguous range (so there aren't gaps after filtering)
mapped_user_ids = torch.searchsorted(unique_user_ids, user)
mapped_movie_ids = torch.searchsorted(unique_movie_ids, movie)

num_users = len(unique_user_ids)
num_movies = len(unique_movie_ids)
embedding_dim = 64

user_embeddings = nn.Embedding(num_users, embedding_dim)
movie_embeddings = nn.Embedding(num_movies, embedding_dim)

loss_fn = binary_cross_entropy_loss

optimizer = torch.optim.Adam(list(user_embeddings.parameters()) + 
                            list(movie_embeddings.parameters()), lr=0.01)

user_emb = user_embeddings(mapped_user_ids)
movie_emb = movie_embeddings(mapped_movie_ids)

num_epochs = 25
all_preds = []
all_labels = []

for epoch in range(num_epochs):
  # need to zero the gradients in the optimizer so we don't
  # use the gradients from previous iterations
  optimizer.zero_grad()  

  user_emb = user_embeddings(mapped_user_ids)
  movie_emb = movie_embeddings(mapped_movie_ids)

  interaction = torch.sum(user_emb * movie_emb, dim=1) / embedding_dim
  output = torch.sigmoid(interaction)

  loss = loss_fn(output, labels)
  loss.backward() 
  optimizer.step()

  print('epoch {} loss {}'.format(epoch+1, loss.item()))

  all_preds.extend(output.detach().cpu().numpy()) 
  all_labels.extend(labels.detach().cpu().numpy())

# calculate the fpr and tpr for all thresholds of the classification
fpr, tpr, threshold = metrics.roc_curve(all_labels, all_preds)
roc_auc = metrics.auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()