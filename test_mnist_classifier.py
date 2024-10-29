import unittest
import torch
from model import CNN  # Import your CNN model class from MnistClassifier.py

class TestMnistClassifier(unittest.TestCase):
    def setUp(self):
        """Initialize the model for testing"""
        self.model = CNN()
    
    def test_model_initialization(self):
        """Test if the model initializes properly"""
        self.assertIsInstance(self.model, CNN)

    def test_forward_pass_output_shape(self):
        """Test if the model's forward pass produces the correct output shape"""
        # Create a dummy input tensor with shape [batch_size, channels, height, width]
        # For MNIST, we use 1 channel and 28x28 images, batch_size = 1 for testing
        dummy_input = torch.randn(1, 1, 28, 28)
        output = self.model(dummy_input)
        
        # The output should have shape [batch_size, 10] for 10 classes
        self.assertEqual(output.shape, (1, 10))

    def test_prediction_values(self):
        """Check if the model output contains valid probabilities (after softmax)"""
        dummy_input = torch.randn(1, 1, 28, 28)
        output = self.model(dummy_input)
        
        # Check that the output does not contain NaNs or infinite values
        self.assertFalse(torch.isnan(output).any().item(), "Output contains NaN values")
        self.assertFalse(torch.isinf(output).any().item(), "Output contains infinite values")

if __name__ == "__main__":
    unittest.main()
