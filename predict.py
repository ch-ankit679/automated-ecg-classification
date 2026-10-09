import torch
import numpy as np
from model import load_model
from utils.preprocessing import preprocess_ecg

# Load model once
model = load_model()

CLASS_NAMES = [
    "Normal",
    "Atrial Premature",
    "PVC",
    "Fusion (V+N)",
    "Fusion (Paced)"
]

def predict_ecg(file=None, data_array=None):
    # Determine if we are predicting from a file (clean) or array (noisy)
    if data_array is None:
        data = preprocess_ecg(file)
    else:
        data = data_array

    # Convert to tensor
    data_tensor = torch.tensor(data, dtype=torch.float32)

    # Prediction and XAI Extraction
    with torch.no_grad():
        output = model(data_tensor)
        
        # Safely extract attention weights from inside the RNNAttentionModel
        x = model.conv1(data_tensor)
        x = model.conv2(x)
        x_out, hid_states = model.rnn_layer(x)
        x_cat = torch.cat([hid_states[0], hid_states[1]], dim=0).transpose(0, 1)
        attention_weights = torch.tanh(model.attn(x_cat))

    # Move to CPU before numpy
    probs = output.detach().cpu().numpy()[0] # Get 1D array of probs
    
    # Format attention weights for visualization
    attn_1d = np.abs(attention_weights.cpu().numpy()).flatten()

    pred_index = probs.argmax()
    confidence = probs.max()
    predicted_label = CLASS_NAMES[pred_index]

    return predicted_label, confidence, probs, attn_1d, data