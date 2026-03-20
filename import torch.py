import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. THALAMUS (Sensory Gatekeeper) ---
# Learns to compress 4K raw data into meaningful 512-dim tensors
class ThalamusNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_encoder = nn.Conv2d(3, 16, kernel_size=3, stride=2) # Sees
        self.audio_encoder = nn.Linear(128, 64)                         # Hears
        self.tactile_encoder = nn.Linear(10, 32)                        # Touches
        self.integration = nn.Linear(16*31*31 + 64 + 32, 512)           # Unifies

    def forward(self, vis, aud, tac):
        v = F.relu(self.vision_encoder(vis)).view(vis.size(0), -1)
        a = F.relu(self.audio_encoder(aud))
        t = F.relu(self.tactile_encoder(tac))
        # Signals become a unified "Thought Tensor"
        return self.integration(torch.cat((v, a, t), dim=1))

# --- 2. LIMBIC SYSTEM (Valence & Reward) ---
# Assigns 'Pain' or 'Pleasure' to sensory patterns
class LimbicSystem(nn.Module):
    def __init__(self):
        super().__init__()
        self.valence_head = nn.Linear(512, 1) # Outputs -1.0 to 1.0

    def forward(self, thought_tensor):
        return torch.tanh(self.valence_head(thought_tensor))

# --- 3. NEOCORTEX (Reasoning & Planning) ---
# A small Transformer block representing the "thinking" part
class Neocortex(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.memory_gate = nn.Linear(512, 512)

    def forward(self, thought_tensor):
        # Adds 'Attention' to the thought
        x = thought_tensor.unsqueeze(0)
        thought = self.layer(x)
        return thought.squeeze(0)

# --- 4. CEREBELLUM (Motor Reflexes) ---
# Turns abstract plans into physical joint torques
class Cerebellum(nn.Module):
    def __init__(self):
        super().__init__()
        self.motor_map = nn.Linear(512, 12) # 12 DOFs for the baby's arms/hands

    def forward(self, thought):
        return torch.tanh(self.motor_map(thought))

# --- 5. THE ORCHESTRATOR (The Nervous System) ---
class DigitalBaby:
    def __init__(self):
        # All modules start with RANDOM weights (Tabula Rasa)
        self.thalamus = ThalamusNet()
        self.limbic = LimbicSystem()
        self.neocortex = Neocortex()
        self.cerebellum = Cerebellum()
        
    def pulse(self, vision, audio, tactile):
        # Step A: Thalamus filters and unifies
        thought_vector = self.thalamus(vision, audio, tactile)
        
        # Step B: Limbic checks for "Pain" (Valence)
        valence = self.limbic(thought_vector)
        
        # Step C: Neocortex reasons and decides
        decision = self.neocortex(thought_vector)
        
        # Step D: Cerebellum executes the movement
        motor_action = self.cerebellum(decision)
        
        return motor_action, valence

# --- INITIALIZATION TEST ---
baby = DigitalBaby()
print("Baby Brain initialized with random weights.")

# Fake simulation data (1 Image, 1 Audio stream, 1 Tactile sensor)
fake_vis = torch.randn(1, 3, 64, 64)
fake_aud = torch.randn(1, 128)
fake_tac = torch.randn(1, 10)

action, feeling = baby.pulse(fake_vis, fake_aud, fake_tac)
print(f"Action Output (Joint Torques): {action.shape}")
print(f"Internal Feeling (Pain/Pleasure): {feeling.item():.4f}")