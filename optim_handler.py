from typing import Dict, Any, List, Optional
import torch
import torch.optim as optim

from song import Song
from loss_handler import LossHandler
from members import Member, PolyphonicMember, MonophonicMember


class OptimHandler:
    """Handles optimization of member weights using gradient descent."""
    
    # Default optimization hyperparameters
    DEFAULT_LR = 0.01
    DEFAULT_BETAS = (0.9, 0.999)
    DEFAULT_WEIGHT_DECAY = 0.0
    
    def __init__(
        self,
        song: Song,
        loss_handler: LossHandler,
    ):
        self.song = song
        self.loss_handler = loss_handler
        self.steps = 0
        self.optimizer: Optional[optim.Optimizer] = None
        
        # Initialize optimizer
        self._initialize_optimizer()
    
    def _initialize_optimizer(self):
        """Create optimizer with parameters from all members."""
        params = []
        
        for member in self.song.members:
            # Get member-specific learning rate or use default
            lr = member.hp.get('lr', self.DEFAULT_LR)
            betas = member.hp.get('betas', self.DEFAULT_BETAS)
            weight_decay = member.hp.get('weight_decay', self.DEFAULT_WEIGHT_DECAY)
            
            # Add member's weights as a parameter group
            params.append({
                'params': [member.weights],
                'lr': lr,
                'betas': betas,
                'weight_decay': weight_decay,
                'name': member.name,  # For reference
            })
        
        # Use Adam optimizer - good default for most cases
        self.optimizer = optim.Adam(params)
    
    def do_steps(self, desired_steps: int) -> List[Dict[str, Any]]:
        """
        Run optimization for a specified number of steps.
        
        Args:
            desired_steps: Number of optimization steps to run
            
        Returns:
            List of loss dictionaries for each step
        """
        if self.optimizer is None:
            raise RuntimeError("Optimizer not initialized")
        
        step_history = []
        
        for _ in range(desired_steps):
            self.optimizer.zero_grad()
            
            # Calculate loss
            total_loss, loss_dict = self.loss_handler.calculate_loss()
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(
                [member.weights for member in self.song.members],
                max_norm=1.0
            )
            
            # Optimizer step
            self.optimizer.step()
            
            # Store loss info
            step_info = {
                'step': self.steps,
                **loss_dict
            }
            step_history.append(step_info)
            
            self.steps += 1
        
        return step_history
    
    def get_optimizer_state(self) -> Dict[str, Any]:
        """Get current optimizer state for inspection."""
        if self.optimizer is None:
            return {}
        
        state = {
            'steps': self.steps,
            'param_groups': []
        }
        
        for group in self.optimizer.param_groups:
            state['param_groups'].append({
                'name': group.get('name', 'unknown'),
                'lr': group['lr'],
                'betas': group['betas'],
                'weight_decay': group['weight_decay'],
            })
        
        return state
    
    def set_learning_rate(self, member_name: str, new_lr: float):
        """Update learning rate for a specific member."""
        if self.optimizer is None:
            return
        
        for group in self.optimizer.param_groups:
            if group.get('name') == member_name:
                group['lr'] = new_lr
                break
