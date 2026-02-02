"""This is untested
This file is kept around because there is a rudimentary inclusion of the RL portion of the workflow
"""

import torch
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from queue import Queue
from threading import Thread
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMAgent:
    def __init__(self, model_name, device, role="speaker"):
        """
        Initialize an LLM agent with specific role and device assignment
        
        Args:
            model_name (str): HuggingFace model identifier
            device (str): GPU device identifier (e.g., 'cuda:0')
            role (str): Either 'speaker' or 'validator'
        """
        self.device = device
        self.role = role
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.learning_rate = 0.0001 if role == "speaker" else 0
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate) if role == "speaker" else None
        
    def generate_response(self, prompt):
        """Generate a response from the model"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_length=200,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.eos_token_id
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def update_weights(self, feedback_score):
        """Simple RL update for the speaker model"""
        if self.role != "speaker" or self.optimizer is None:
            return
            
        # Convert feedback into a loss signal
        loss = -torch.tensor(feedback_score, device=self.device)
        loss.requires_grad_(True)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class DualLLMSystem:
    def __init__(self, speaker_model, validator_model):
        """
        Initialize the dual LLM system with two separate GPU devices
        
        Args:
            speaker_model (str): Model name for the speaker
            validator_model (str): Model name for the validator
        """
        # Initialize multiprocessing
        mp.set_start_method('spawn', force=True)
        
        # Assign different GPUs
        self.speaker = LLMAgent(speaker_model, 'cuda:0', role="speaker")
        self.validator = LLMAgent(validator_model, 'cuda:1', role="validator")
        
        # Communication queues
        self.speaker_to_validator = Queue()
        self.validator_to_speaker = Queue()
        
        # Validation history
        self.validation_history = []
        
    def validator_prompt_template(self, statement):
        """Template for validator's system prompt"""
        return f"""Please verify the following statement and respond with either 'CORRECT' or 'INCORRECT' 
        followed by a brief explanation:
        
        Statement: {statement}
        """
    
    def run_conversation(self, initial_prompt, max_iterations=5):
        """
        Run the conversation between the two models
        
        Args:
            initial_prompt (str): Starting prompt for the speaker
            max_iterations (int): Maximum number of conversation turns
        """
        current_statement = self.speaker.generate_response(initial_prompt)
        
        for i in range(max_iterations):
            logger.info(f"Iteration {i+1}")
            logger.info(f"Speaker: {current_statement}")
            
            # Get validator's response
            validator_prompt = self.validator_prompt_template(current_statement)
            validation = self.validator.generate_response(validator_prompt)
            logger.info(f"Validator: {validation}")
            
            # Parse validation result
            is_correct = validation.strip().upper().startswith("CORRECT")
            self.validation_history.append(is_correct)
            
            # Update speaker's weights based on validation
            feedback_score = 1.0 if is_correct else -0.1
            self.speaker.update_weights(feedback_score)
            
            # Check if we've reached consistency
            if len(self.validation_history) >= 3 and all(self.validation_history[-3:]):
                logger.info("Reached consistent correct statements. Stopping.")
                break
            
            # Generate new statement if needed
            if not is_correct:
                current_statement = self.speaker.generate_response(initial_prompt)
            
        return self.validation_history

def main():
    # Example usage
    system = DualLLMSystem(
        speaker_model="gpt2",  # Replace with your preferred models
        validator_model="gpt2-medium"
    )
    
    initial_prompt = "Generate a factual statement about quantum physics."
    history = system.run_conversation(initial_prompt)
    
    logger.info(f"Conversation completed with validation history: {history}")

if __name__ == "__main__":
    main()