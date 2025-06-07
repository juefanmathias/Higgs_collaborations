def feature_correlations(data):
    pass


def systematics_dependence(data):
    pass


def minimal_dependent_features(data):
    return data.columns

import os
from tensorflow.keras.models import model_from_json
import pickle
 
# Define the subdirectory
subfolder_path = os.path.join("Best_NN_model")
 
# Load the model structure (JSON)
with open(os.path.join(subfolder_path, "model.json"), "r") as json_file:
    model_json = json_file.read()
model = model_from_json(model_json)
 
# Load the model weights (H5)
model.load_weights(os.path.join(subfolder_path, "model.h5"))
 
# Load preprocessing (PKL)
with open(os.path.join(subfolder_path, "preprocessing.pkl"), "rb") as pkl_file:
    preprocessing = pickle.load(pkl_file)
 
import torch
from torch.autograd import grad
 
# Assume `model` is your trained neural network and `data_loader` provides your dataset.
 
def compute_feature_importance(model, data_loader):
    model.eval()  # Set the model to evaluation mode
    feature_importance = None
   
    for inputs, _ in data_loader:
        inputs = inputs.requires_grad_()  # Enable gradient computation for inputs
        outputs = model(inputs)  # Forward pass
       
        # Assuming single-output model, take output of interest
        output_of_interest = outputs[:, 0].sum()
       
        # Compute gradients w.r.t. inputs
        grads = grad(output_of_interest, inputs, create_graph=False, retain_graph=False)[0]
       
        # Aggregate gradients
        if feature_importance is None:
            feature_importance = grads.abs().mean(dim=0)  # Start accumulation
        else:
            feature_importance += grads.abs().mean(dim=0)
   
    return feature_importance / len(data_loader)  # Normalize across dataset
 
# Example usage
# importance = compute_feature_importance(model, data_loader)
# print("Feature Importance: ", importance)