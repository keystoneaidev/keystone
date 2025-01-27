import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, List

class RecommendationModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(RecommendationModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.softmax(x)

class AIRecommender:
    def __init__(self):
        self.model = RecommendationModel(input_size=15, hidden_size=30, output_size=5)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.trained = False
    
    def generate_recommendation(self, input_features: List[float]) -> Dict[str, Any]:
        if not self.trained:
            return {"error": "Model is not trained"}
        
        with torch.no_grad():
            input_tensor = torch.tensor([input_features], dtype=torch.float32)
            output = self.model(input_tensor)
            recommendation_index = torch.argmax(output, dim=1).item()
            return {"recommendation": self._decode_recommendation(recommendation_index), "confidence": output[0][recommendation_index].item()}
    
    def train_model(self, training_data: List[List[float]], labels: List[int], epochs: int = 100):
        if len(training_data) != len(labels):
            raise ValueError("Mismatch between training data and labels.")
        
        dataset = torch.tensor(training_data, dtype=torch.float32)
        label_tensor = torch.tensor(labels, dtype=torch.long)
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            outputs = self.model(dataset)
            loss = self.criterion(outputs, label_tensor)
            loss.backward()
            self.optimizer.step()
            if epoch % 10 == 0:
                print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}")
        
        self.trained = True
    
    def _decode_recommendation(self, index: int) -> str:
        recommendations = [
            "Increase token utility through governance.",
            "Optimize smart contract for gas efficiency.",
            "Integrate multi-chain compatibility.",
            "Enhance security measures.",
            "Focus on community engagement."
        ]
        return recommendations[index]

if __name__ == "__main__":
    recommender = AIRecommender()
    training_data = [[random.uniform(0, 1) for _ in range(15)] for _ in range(100)]
    labels = [random.randint(0, 4) for _ in range(100)]
    recommender.train_model(training_data, labels)
    
    test_features = [random.uniform(0, 1) for _ in range(15)]
    result = recommender.generate_recommendation(test_features)
    print(json.dumps(result, indent=2))

