import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.figure_factory as ff
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import auc, roc_curve
import plotly.express as px
from sklearn.metrics import confusion_matrix
import torch.nn as nn
import torch.nn.functional as F
import dgl


class Results:
   def __init__(self):
      pass

   def accuracy(self, true, preds):
      return accuracy_score(true, preds)

   def calculate_metrics(self, y_test, y_prob, y_pred, model_name):
      auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
      accuracy = accuracy_score(y_test, y_pred)
      precision = precision_score(y_test, y_pred, average='macro')
      recall = recall_score(y_test, y_pred, average='macro')
      f1 = f1_score(y_test, y_pred, average='macro')

      results = {
            'Model': model_name,
            'AUC': auc,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
      }

      return pd.Series(results)

   def print_confusion_matrix(self, y_test, y_pred, labels):
      cm = confusion_matrix(y_test, y_pred, labels=labels)

      # Plot the confusion matrix using Plotly
      fig = ff.create_annotated_heatmap(
         z=cm,
         x=labels,
         y=labels,
         colorscale='blues',
         annotation_text=cm,
         showscale=True
      )

      fig.update_layout(
         title='Confusion Matrix',
         xaxis=dict(title='Predicted Label'),
         yaxis=dict(title='True Label'),
      )

      fig.show()


   def plot_roc_curve(self, y_true, y_prob, model_name, title="ROC Curve"):
      # Convert the true labels to one-hot encoding
      n_classes = len(set(y_true))

      if n_classes < 2:
         raise ValueError("Number of classes should be at least 2 for ROC curve.")

      # Calculate micro-average ROC curve and AUC
      fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1], pos_label=1)
      roc_auc = auc(fpr, tpr)

      # Plot the ROC curve
      fig = go.Figure()
      fig.add_trace(go.Scatter(x=fpr, y=tpr,
                              mode='lines',
                              name=f'{model_name} (AUC = {roc_auc:.2f})'))

      fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                              mode='lines',
                              line=dict(color='navy', width=2, dash='dash'),
                              showlegend=False))

      fig.update_layout(
         title=title,
         xaxis=dict(title='False Positive Rate'),
         yaxis=dict(title='True Positive Rate'),
         width=800,
         height=600
      )

      fig.show()

   def plot_roc_curves(self, y_true, y_probs, model_names, title="ROC-AUC Curves"):
      fig = go.Figure()

      for idx, model_name in enumerate(model_names):
         fpr, tpr, _ = roc_curve(y_true, y_probs[idx])
         roc_auc = auc(fpr, tpr)

         fig.add_trace(go.Scatter(x=fpr, y=tpr,
                     mode='lines',
                     name=f'{model_name} (AUC = {roc_auc:.2f})'))

      fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                     mode='lines',
                     line=dict(color='navy', width=2, dash='dash'),
                     showlegend=False))

      fig.update_layout(
         title=title,
         xaxis=dict(title='False Positive Rate'),
         yaxis=dict(title='True Positive Rate'),
         width=800,
         height=600
      )
      fig.show()

   def plot_hist(self, true_labels, predicted_labels):
      # Check if the lengths of true_labels and predicted_labels are the same
      if len(true_labels) != len(predicted_labels):
         raise ValueError("Lengths of true_labels and predicted_labels must be the same.")

      # Identify correct predictions
      correct_predictions = (true_labels == predicted_labels)

      # Count the occurrences of each correct label
      unique_labels, counts = np.unique(true_labels[correct_predictions], return_counts=True)
      unique_labels = [str(num) for num in unique_labels]

      # Create a Plotly bar chart
      fig = px.bar(x=unique_labels, y=counts, labels={'x': 'Categorical Labels', 'y': 'Count'},
                  title='Histogram of Correct Categorical Labels')

      # Show the plot
      fig.show()

   def plot_3D_scatter(self, x, y, z, x_label, y_label, z_label, color_lambda=None):
      fig = go.Figure()

      if color_lambda is None:
         colors = z
      else:
         colors = color_lambda

      fig.add_trace(
         go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker=dict(
                  size=3,
                  color=colors,
                  opacity=0.8
            )
         )
      )

      # set layout and show plot
      fig.update_layout(
         height=800,
         width=800,
         scene=dict(
            xaxis_title=x_label,
            yaxis_title=y_label,
            zaxis_title=z_label
         )
      )

      fig.show()


class Data_Handler:
   def __init__(self, df):
      self.df = df

   def train_test_split(self, target_col, cols2exclude = None, test_size=0.2):
      if cols2exclude is None:
         cols2exclude = target_col
      else:
         cols2exclude.append(target_col)
      return train_test_split(self.df[self.df.columns.difference(cols2exclude)], self.df[target_col],
                                                          test_size=test_size, shuffle=False)

   def qcut_data(self, target_col, num_cuts):
      target_cuts = pd.qcut(self.df[target_col], num_cuts, labels=False)
      return target_cuts

class LR:
   def __init__(self):
      self.model = LogisticRegression()
      self.results = Results()

   def train(self, X_train, y_train):
      self.model.fit(X_train, y_train)

   def predict(self, X_test):
      preds = self.model.predict(X_test)
      return preds

   def predict_prob(self, X_test):
      probs = self.model.predict_proba(X_test)
      return probs

class GB:
   def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42):
      self.model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=random_state)
      self.results = Results()

   def train(self, X_train, y_train):
      self.model.fit(X_train, y_train)

   def predict(self, X_test):
      preds = self.model.predict(X_test)
      return preds

   def predict_prob(self, X_test):
      probs = self.model.predict_proba(X_test)
      return probs

class MLP:
   def __init__(self, hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=0.0001):
      self.model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, alpha=alpha)
      self.results = Results()

   def train(self, X_train, y_train):
      self.model.fit(X_train, y_train)

   def predict(self, X_test):
      preds = self.model.predict(X_test)
      return preds

   def predict_prob(self, X_test):
      probs = self.model.predict_proba(X_test)
      return probs


# class GCN(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(GCN, self).__init__()
#         self.layer1 = dgl.nn.GraphConv(input_dim, hidden_dim)
#         self.layer2 = dgl.nn.GraphConv(hidden_dim, hidden_dim)
#         self.fc1 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, output_dim)

#     def forward(self, g, features):
#         x = F.relu(self.layer1(g, features))
#         x = F.relu(self.layer2(g, x))
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x
