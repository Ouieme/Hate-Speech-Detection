from flask import Flask, request, render_template
import torch
from preprocess import preprocess_text

# Define the model (ensure the architecture matches your trained model)
class HateSpeechClassifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(HateSpeechClassifier, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Set model parameters (match your trained model)
input_dim = 100  # Should match the embedding dimension
hidden_dim = 150  # Ensure this matches your trained model
output_dim = 2

# Load the model
model = HateSpeechClassifier(input_dim, hidden_dim, output_dim)
model.load_state_dict(torch.load("hate_speech_model.pth"))
model.eval()  # Set the model to evaluation mode

# Initialize Flask app
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get user input from the form
        user_input = request.form["text"]
        
        # Preprocess the input text
        input_tensor = preprocess_text(user_input)  # Preprocessing step
        
        # Ensure the tensor is the correct shape
        input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension

        # Pass the input tensor to the model
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        
        # Decode the prediction
        label = "Hate Speech" if predicted.item() == 1 else "Not Hate Speech"
        
        return render_template("index.html", prediction=label)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
