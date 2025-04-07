import gym
from gym import spaces
import numpy as np
import mss
import cv2
import time
import os
import pyautogui
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
import re
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Adjust path as needed
except ImportError:
    TESSERACT_AVAILABLE = False
    print("Pytesseract not available. OCR-based rewards will be disabled.")

#########################################
# Feature Extraction and Image Processing
#########################################
def extract_track_features(frame):
    """
    Extract important features from the track image
    
    Args:
        frame: The raw screen capture image
        
    Returns:
        processed_frame: Image with highlighted track boundaries and features
        track_direction: Estimated track direction (angle)
        distance_to_edges: Array of distances to track edges
    """
    # Convert to HSV color space for better track detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    
    # Define color ranges for track detection (customize for your game)
    # For Trackmania, this might detect track boundaries, grass, etc.
    lower_track = np.array([0, 0, 100])  # Light track color
    upper_track = np.array([180, 30, 255])
    
    lower_grass = np.array([35, 50, 50])  # Green grass
    upper_grass = np.array([85, 255, 255])
    
    # Create masks
    track_mask = cv2.inRange(hsv, lower_track, upper_track)
    grass_mask = cv2.inRange(hsv, lower_grass, upper_grass)
    
    # Combine masks (track is good, grass is bad)
    combined_mask = cv2.bitwise_and(track_mask, cv2.bitwise_not(grass_mask))
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours of the track
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a visualization image
    visualization = frame.copy()
    
    # Track direction and distance to edges
    track_direction = 0
    distance_to_edges = np.zeros(8)  # 8 directions
    
    if contours:
        # Draw the largest contour (the track)
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(visualization, [largest_contour], 0, (0, 255, 0), 2)
        
        # Find the center of the image (where the car would be)
        height, width = frame.shape[:2]
        center_x, center_y = width // 2, height // 2
        
        # Calculate distances to track edges in 8 directions
        angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        for i, angle in enumerate(angles):
            # Create a ray from the center
            max_dist = min(width, height)
            for dist in range(10, max_dist, 5):
                x = int(center_x + dist * np.cos(angle))
                y = int(center_y + dist * np.sin(angle))
                
                # Check if we've hit the edge of the image
                if x < 0 or y < 0 or x >= width or y >= height:
                    distance_to_edges[i] = max_dist
                    break
                
                # Check if we've hit the track boundary
                if cleaned_mask[y, x] == 0:  # Outside the track
                    distance_to_edges[i] = dist
                    # Draw the ray
                    cv2.line(visualization, (center_x, center_y), (x, y), (0, 0, 255), 1)
                    break
                    
                if dist == max_dist - 5:
                    distance_to_edges[i] = max_dist
        
        # Estimate track direction based on the distances
        # The idea is that the track likely continues in the direction with the largest distance
        max_dist_idx = np.argmax(distance_to_edges)
        track_direction = angles[max_dist_idx]
        
        # Draw track direction
        dir_x = int(center_x + 50 * np.cos(track_direction))
        dir_y = int(center_y + 50 * np.sin(track_direction))
        cv2.arrowedLine(visualization, (center_x, center_y), (dir_x, dir_y), (255, 0, 0), 2)
        
    return visualization, track_direction, distance_to_edges


#########################################
# Advanced Reward System
#########################################
class RewardSystem:
    def __init__(self):
        # Previous position tracking for velocity calculation
        self.prev_position = None
        self.position_history = []
        self.velocity_history = []
        
        # Lap time tracking
        self.start_time = None
        self.prev_lap_time = float('inf')
        self.best_lap_time = float('inf')
        self.current_lap_time = 0
        
        # Checkpoint tracking
        self.checkpoints_passed = 0
        self.last_checkpoint_time = 0
        
        # Track progression - percentage of track completed
        self.track_progression = 0
        self.previous_progression = 0
        
        # Performance metrics
        self.smoothness_factor = 0  # How smooth the driving is
        self.centerline_deviation = 0  # How well the car stays on racing line
    
    def extract_time_from_screen(self, image, time_region=(50, 50, 200, 100)):
        """Extract the lap time from the game screen using OCR"""
        if not TESSERACT_AVAILABLE:
            return None
            
        # Crop to the region where the timer is displayed
        time_img = image[time_region[1]:time_region[3], time_region[0]:time_region[2]]
        
        # Preprocess for better OCR
        gray = cv2.cvtColor(time_img, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        
        # Perform OCR
        text = pytesseract.image_to_string(thresh, config='--psm 7 -c tessedit_char_whitelist=0123456789:.')
        
        # Parse time
        time_match = re.search(r'(\d+):(\d+\.\d+)', text)
        if time_match:
            minutes = int(time_match.group(1))
            seconds = float(time_match.group(2))
            return minutes * 60 + seconds
        
        return None
    
    def estimate_position(self, features):
        """Estimate the car's position based on extracted features"""
        # This would use the track features to determine where on the track the car is
        # In a real implementation, you might use checkpoints or track landmarks
        
        # For this example, we'll use a simplified position estimate
        # based on the track direction and distances to edges
        track_direction = features[1]
        distances = features[2]
        
        # Calculate a position vector (very simplified)
        position = np.array([
            np.cos(track_direction) * np.mean(distances),
            np.sin(track_direction) * np.mean(distances)
        ])
        
        return position
    
    def calculate_velocity(self, position):
        """Calculate velocity based on position changes"""
        if self.prev_position is not None:
            velocity = position - self.prev_position
            self.velocity_history.append(np.linalg.norm(velocity))
            
            # Keep history limited
            if len(self.velocity_history) > 10:
                self.velocity_history.pop(0)
        
        self.prev_position = position
        
        # Return current velocity magnitude
        if self.velocity_history:
            return np.mean(self.velocity_history)
        return 0
    
    def detect_checkpoints(self, image):
        """Detect if the car has passed a checkpoint"""
        # In a real implementation, this would use computer vision to detect checkpoint markers
        # For this example, we'll use a simplified time-based approach
        
        current_time = time.time()
        # Assume a new checkpoint every 10 seconds (very simplified)
        if current_time - self.last_checkpoint_time > 10:
            self.checkpoints_passed += 1
            self.last_checkpoint_time = current_time
            return True
        
        return False
    
    def estimate_progression(self, checkpoint_count, total_checkpoints=10):
        """Estimate how far along the track the car has progressed"""
        self.previous_progression = self.track_progression
        self.track_progression = min(1.0, checkpoint_count / total_checkpoints)
        
        # Calculate progression improvement
        progression_delta = self.track_progression - self.previous_progression
        
        return progression_delta
    
    def calculate_smoothness(self, steering_history):
        """Calculate how smooth the driving is based on steering inputs"""
        if len(steering_history) < 2:
            return 0
            
        # Calculate steering changes
        steering_changes = np.abs(np.diff(steering_history))
        
        # Lower values mean smoother driving
        smoothness = 1.0 - min(1.0, np.mean(steering_changes) * 5)
        
        return smoothness
    
    def calculate_centerline_deviation(self, distances):
        """Calculate how well the car stays on the ideal racing line"""
        # In a real implementation, this would compare the car's position
        # to a pre-defined racing line
        
        # For this example, we'll use a simplified approach based on
        # the symmetry of distances to left and right edges
        left_distances = distances[0:4]
        right_distances = distances[4:8]
        
        # Calculate asymmetry - lower is better
        left_mean = np.mean(left_distances)
        right_mean = np.mean(right_distances)
        
        if left_mean + right_mean > 0:
            deviation = np.abs(left_mean - right_mean) / (left_mean + right_mean)
        else:
            deviation = 1.0
            
        # Convert to a 0-1 score where 1 is perfect
        centerline_score = 1.0 - min(1.0, deviation)
        
        return centerline_score
    
    def calculate_reward(self, image, features, action_history):
        """Calculate the comprehensive reward based on multiple factors"""
        reward = 0
        
        # 1. Extract lap time (if visible)
        lap_time = self.extract_time_from_screen(image)
        if lap_time is not None:
            self.current_lap_time = lap_time
            
            # Check if lap completed (timer reset)
            if self.current_lap_time < self.prev_lap_time - 10:
                # Completed a lap - big reward for lap completion
                reward += 100
                
                # Additional reward for improving lap time
                if self.prev_lap_time != float('inf'):
                    time_improvement = self.prev_lap_time - self.current_lap_time
                    reward += max(0, time_improvement * 50)  # 50 points per second improvement
                
                # Update lap time tracking
                if self.current_lap_time < self.best_lap_time:
                    self.best_lap_time = self.current_lap_time
                
                self.prev_lap_time = self.current_lap_time
        
        # 2. Position and velocity rewards
        position = self.estimate_position(features)
        velocity = self.calculate_velocity(position)
        
        # Reward for speed
        reward += velocity * 0.1
        
        # 3. Checkpoint and progression rewards
        if self.detect_checkpoints(image):
            reward += 10  # Reward for reaching a checkpoint
        
        progression_delta = self.estimate_progression(self.checkpoints_passed)
        reward += progression_delta * 50  # Reward for making progress on the track
        
        # 4. Driving quality rewards
        if len(action_history) > 5:
            steering_history = [a[0] for a in action_history[-10:]]  # Get last 10 steering actions
            
            # Calculate driving smoothness (penalize erratic steering)
            self.smoothness_factor = self.calculate_smoothness(steering_history)
            reward += self.smoothness_factor * 2
            
            # Calculate centerline adherence
            self.centerline_deviation = self.calculate_centerline_deviation(features[2])
            reward += self.centerline_deviation * 2
        
        # 5. Penalties
        # Detect if car is off-track or crashed (simplified)
        min_distance = np.min(features[2])
        if min_distance < 10:  # Very close to edge
            reward -= 5  # Penalty for getting too close to edge
        
        # Stuck penalty is handled by the environment's step function
        
        return reward


#########################################
# Custom Trackmania Environment
#########################################
class TrackmaniaEnv(gym.Env):
    """Custom Environment for Trackmania that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, screen_region=(0, 40, 1024, 768), frame_skip=4, grayscale=True, 
                 save_video=False, video_dir="./videos/"):
        super(TrackmaniaEnv, self).__init__()
        
        # Define action and observation space
        # Actions: [steering, acceleration, brake]
        # steering: -1 (full left) to 1 (full right)
        # acceleration: 0 (no gas) to 1 (full gas)
        # brake: 0 (no brake) to 1 (full brake)
        self.action_space = spaces.Box(
            low=np.array([-1, 0, 0]), 
            high=np.array([1, 1, 1]), 
            dtype=np.float32
        )
        
        # Screen capture setup
        self.screen_region = screen_region
        self.sct = mss.mss()
        self.grayscale = grayscale
        self.frame_skip = frame_skip
        
        # Image size after preprocessing
        self.width = 84
        self.height = 84
        img_shape = (self.height, self.width, 1 if grayscale else 3)
        self.observation_space = spaces.Box(low=0, high=255, shape=img_shape, dtype=np.uint8)
        
        # State tracking
        self.current_time = 0
        self.start_time = 0
        self.last_checkpoint_time = 0
        self.episode_start_time = 0
        self.last_reward = 0
        self.total_reward = 0
        self.steps = 0
        self.done = False
        self.episode_number = 0
        
        # Crash detection
        self.last_positions = []
        self.position_history_size = 10
        self.stuck_frames_threshold = 30
        self.stuck_frames = 0
        
        # Game control keys
        self.reset_key = 'backspace'
        self.restart_key = 'delete'
        
        # Action history for reward calculation
        self.action_history = []
        
        # Advanced reward system
        self.reward_system = RewardSystem()
        
        # Video recording
        self.save_video = save_video
        self.video_dir = video_dir
        self.video_writer = None
        
        if self.save_video:
            os.makedirs(self.video_dir, exist_ok=True)
    
    def _get_observation(self):
        """Capture screen and preprocess the image"""
        monitor = {"top": self.screen_region[0], 
                   "left": self.screen_region[1], 
                   "width": self.screen_region[2], 
                   "height": self.screen_region[3]}
        screenshot = self.sct.grab(monitor)
        img = np.array(screenshot)
        
        # Convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        
        # Extract track features
        processed_img, track_direction, distances = extract_track_features(img)
        
        # Save the original and processed images for video if enabled
        if self.save_video and self.video_writer is not None:
            # Resize for consistent video size
            frame_to_save = cv2.resize(processed_img, (640, 480))
            self.video_writer.write(cv2.cvtColor(frame_to_save, cv2.COLOR_RGB2BGR))
        
        # Save the features for reward calculation
        self.current_features = (processed_img, track_direction, distances)
        
        # Resize and preprocess for the neural network
        if self.grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = np.expand_dims(img, axis=2)
        
        img = cv2.resize(img, (self.width, self.height))
        
        return img
    
    def _take_action(self, action):
        """Convert normalized actions to keyboard/controller inputs"""
        steering, acceleration, brake = action
        
        # Store action for reward calculation
        self.action_history.append(action)
        if len(self.action_history) > 20:
            self.action_history.pop(0)
        
        # Apply steering
        if steering < -0.5:
            pyautogui.keyDown('left')
            pyautogui.keyUp('right')
        elif steering > 0.5:
            pyautogui.keyDown('right')
            pyautogui.keyUp('left')
        else:
            pyautogui.keyUp('left')
            pyautogui.keyUp('right')
        
        # Apply acceleration
        if acceleration > 0.5:
            pyautogui.keyDown('up')
        else:
            pyautogui.keyUp('up')
        
        # Apply brake
        if brake > 0.5:
            pyautogui.keyDown('down')
        else:
            pyautogui.keyUp('down')
    
    def _calculate_reward(self):
        """Calculate reward based on game state"""
        # Use the advanced reward system
        reward = self.reward_system.calculate_reward(
            self.current_features[0],  # Original image
            self.current_features,     # Track features
            self.action_history        # Recent actions
        )
        
        # Add a small base reward for staying on track
        reward += 0.1
        
        # Detect if car is stuck by checking if position hasn't changed
        current_img = self._get_observation()
        self.last_positions.append(current_img)
        if len(self.last_positions) > self.position_history_size:
            self.last_positions.pop(0)
        
        # Check if the car is stuck (similar frames over time)
        if len(self.last_positions) == self.position_history_size:
            first_frame = self.last_positions[0]
            latest_frame = self.last_positions[-1]
            diff = cv2.absdiff(first_frame, latest_frame)
            non_zero_count = np.count_nonzero(diff)
            
            if non_zero_count < 100:  # Threshold for "not moving"
                self.stuck_frames += 1
                if self.stuck_frames > self.stuck_frames_threshold:
                    reward -= 10  # Penalty for getting stuck
                    self.done = True
            else:
                self.stuck_frames = 0
        
        return reward
    
    def step(self, action):
        """Execute one time step within the environment"""
        self.steps += 1
        
        # Skip frames to speed up training
        for _ in range(self.frame_skip):
            if not self.done:
                self._take_action(action)
                time.sleep(0.01)  # Small delay to allow game to respond
        
        # Get new observation
        observation = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward()
        self.total_reward += reward
        self.last_reward = reward
        
        # Check if episode is done
        # In a full implementation, detect race completion or timeout
        if self.steps > 1000:  # Limit episode length
            self.done = True
        
        info = {
            'steps': self.steps,
            'total_reward': self.total_reward,
            'stuck_frames': self.stuck_frames,
            'checkpoints': self.reward_system.checkpoints_passed,
            'lap_time': self.reward_system.current_lap_time,
            'best_lap': self.reward_system.best_lap_time
        }
        
        return observation, reward, self.done, info
    
    def reset(self):
        """Reset the state of the environment to an initial state"""
        # Close video writer if it exists
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        
        # Create a new video writer if enabled
        if self.save_video:
            video_path = os.path.join(self.video_dir, f"episode_{self.episode_number}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(video_path, fourcc, 20.0, (640, 480))
        
        # Reset the game by pressing the reset key
        pyautogui.press(self.reset_key)
        time.sleep(1)  # Wait for the game to reset
        
        # Reset internal state
        self.steps = 0
        self.done = False
        self.total_reward = 0
        self.last_reward = 0
        self.episode_start_time = time.time()
        self.last_checkpoint_time = time.time()
        self.stuck_frames = 0
        self.last_positions = []
        self.action_history = []
        self.episode_number += 1
        
        # Reset the reward system
        self.reward_system = RewardSystem()
        
        # Return initial observation
        return self._get_observation()
    
    def render(self, mode='human'):
        """Render the environment to the screen"""
        # We don't need to render separately as we're capturing the game screen
        pass
    
    def close(self):
        """Close any open resources"""
        # Release all keys
        pyautogui.keyUp('up')
        pyautogui.keyUp('down')
        pyautogui.keyUp('left')
        pyautogui.keyUp('right')
        self.sct.close()
        
        # Close video writer if it exists
        if self.video_writer is not None:
            self.video_writer.release()

# Helper function to create a vectorized environment
def make_env(video=False):
    def _init():
        env = TrackmaniaEnv(save_video=video)
        return env
    return _init

# Create vectorized environment (required for some RL algorithms)
def create_trackmania_env(video=False):
    return DummyVecEnv([make_env(video)])


#########################################
# Training Callbacks and Visualization
#########################################
class TrackmaniaCallback(BaseCallback):
    def __init__(self, check_freq=1000, log_dir="./logs/", verbose=1):
        super(TrackmaniaCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.best_mean_reward = -np.inf
        self.rewards = []
        self.lap_times = []
        self.checkpoints = []
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
    def _on_step(self) -> bool:
        # Log reward
        if len(self.model.ep_info_buffer) > 0 and len(self.model.ep_info_buffer[0]) > 0:
            ep_reward = self.model.ep_info_buffer[-1]["r"]
            ep_len = self.model.ep_info_buffer[-1]["l"]
            
            # Get additional info from the environment
            env = self.training_env.envs[0]
            info = env.get_attr("info")[0]
            
            # Store current episode data
            self.rewards.append(ep_reward)
            
            # Store lap time if available, otherwise use episode length
            if 'lap_time' in info and info['lap_time'] > 0:
                self.lap_times.append(info['lap_time'])
            else:
                self.lap_times.append(ep_len)
                
            # Store checkpoint count
            if 'checkpoints' in info:
                self.checkpoints.append(info['checkpoints'])
            else:
                self.checkpoints.append(0)
            
            # Check if we have a new best model (based on mean reward)
            if len(self.rewards) > 10:  # Need enough episodes for meaningful average
                mean_reward = np.mean(self.rewards[-10:])
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    if self.verbose > 0:
                        print(f"Saving new best model with mean reward: {mean_reward:.2f}")
                    self.model.save(os.path.join(self.log_dir, "best_model"))
            
            # Log progress every check_freq steps
            if self.n_calls % self.check_freq == 0:
                if len(self.rewards) > 0:
                    mean_reward = np.mean(self.rewards[-10:] if len(self.rewards) >= 10 else self.rewards)
                    mean_lap_time = np.mean(self.lap_times[-10:] if len(self.lap_times) >= 10 else self.lap_times)
                    mean_checkpoints = np.mean(self.checkpoints[-10:] if len(self.checkpoints) >= 10 else self.checkpoints)
                    
                    print(f"Steps: {self.n_calls}")
                    print(f"Mean reward: {mean_reward:.2f}")
                    print(f"Mean lap time: {mean_lap_time:.2f}")
                    print(f"Mean checkpoints: {mean_checkpoints:.2f}")
                    print("----------------------------")
                
                # Save rewards and lap times to file
                np.save(os.path.join(self.log_dir, "rewards.npy"), np.array(self.rewards))
                np.save(os.path.join(self.log_dir, "lap_times.npy"), np.array(self.lap_times))
                np.save(os.path.join(self.log_dir, "checkpoints.npy"), np.array(self.checkpoints))
                
                # Plot training progress every few checkpoints
                if self.n_calls % (self.check_freq * 5) == 0:
                    self.plot_training_progress()
        
        return True
    
    def plot_training_progress(self):
        """Generate plots to visualize training progress"""
        # Create a figure with multiple subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
        
        # Plot rewards
        episodes = np.arange(1, len(self.rewards) + 1)
        ax1.plot(episodes, self.rewards, 'b-')
        ax1.set_title('Reward per Episode')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.grid(True)
        
        # Calculate and plot moving average for rewards
        window_size = 10
        if len(self.rewards) >= window_size:
            moving_avg = np.convolve(self.rewards, np.ones(window_size)/window_size, mode='valid')
            ax1.plot(np.arange(window_size, len(self.rewards) + 1), moving_avg, 'r-', 
                    label=f'{window_size}-episode Moving Average')
            ax1.legend()
        
        # Plot lap times
        if self.lap_times:
            ax2.plot(episodes[:len(self.lap_times)], self.lap_times, 'g-')
            ax2.set_title('Lap Time per Episode')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Lap Time (seconds or steps)')
            ax2.grid(True)
            
            # Calculate and plot moving average for lap times
            if len(self.lap_times) >= window_size:
                moving_avg = np.convolve(self.lap_times, np.ones(window_size)/window_size, mode='valid')
                ax2.plot(np.arange(window_size, len(self.lap_times) + 1), moving_avg, 'r-', 
                        label=f'{window_size}-episode Moving Average')
                ax2.legend()
        
        # Plot checkpoints
        if self.checkpoints:
            ax3.plot(episodes[:len(self.checkpoints)], self.checkpoints, 'm-')
            ax3.set_title('Checkpoints Reached per Episode')
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Checkpoints')
            ax3.grid(True)
            
            # Calculate and plot moving average for checkpoints
            if len(self.checkpoints) >= window_size:
                moving_avg = np.convolve(self.checkpoints, np.ones(window_size)/window_size, mode='valid')
                ax3.plot(np.arange(window_size, len(self.checkpoints) + 1), moving_avg, 'r-', 
                        label=f'{window_size}-episode Moving Average')
                ax3.legend()
        
        # Make sure x-axis is integer (episode numbers)
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax3.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'training_progress.png'))
        plt.close()

def plot_training_results(log_dir="./logs/"):
    """Generate comprehensive plots after training is complete"""
    # Load the saved data
    rewards = np.load(os.path.join(log_dir, "rewards.npy"))
    
    try:
        lap_times = np.load(os.path.join(log_dir, "lap_times.npy"))
        checkpoints = np.load(os.path.join(log_dir, "checkpoints.npy"))
    except FileNotFoundError:
        lap_times = []
        checkpoints = []

    # Create a figure with multiple subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))

    # Plot rewards
    episodes = np.arange(1, len(rewards) + 1)
    ax1.plot(episodes, rewards, 'b-')
    ax1.set_title('Reward per Episode')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.grid(True)

    # Calculate and plot moving average for rewards
    window_size = 10
    if len(rewards) >= window_size:
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        ax1.plot(np.arange(window_size, len(rewards) + 1), moving_avg, 'r-', label=f'{window_size}-episode Moving Average')
        ax1.legend()

    # Plot lap times
    if lap_times:
        ax2.plot(episodes[:len(lap_times)], lap_times, 'g-')
        ax2.set_title('Lap Time per Episode')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Lap Time (seconds or steps)')
        ax2.grid(True)

        # Calculate and plot moving average for lap times
        if len(lap_times) >= window_size:
            moving_avg = np.convolve(lap_times, np.ones(window_size)/window_size, mode='valid')
            ax2.plot(np.arange(window_size, len(lap_times) + 1), moving_avg, 'r-', label=f'{window_size}-episode Moving Average')
            ax2.legend()

    # Plot checkpoints
    if checkpoints:
        ax3.plot(episodes[:len(checkpoints)], checkpoints, 'm-')
        ax3.set_title('Checkpoints Reached per Episode')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Checkpoints')
        ax3.grid(True)

        # Calculate and plot moving average for checkpoints
        if len(checkpoints) >= window_size:
            moving_avg = np.convolve(checkpoints, np.ones(window_size)/window_size, mode='valid')
            ax3.plot(np.arange(window_size, len(checkpoints) + 1), moving_avg, 'r-', label=f'{window_size}-episode Moving Average')
            ax3.legend()

    # Make sure x-axis is integer (episode numbers)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax3.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'final_training_results.png'))
    plt.close()

# Main training routine
def train_trackmania_agent():
    log_dir = "./logs/"
    os.makedirs(log_dir, exist_ok=True)

    # Configure logger
    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])

    # Create environment
    env = create_trackmania_env(video=False)

    # Create PPO model
    model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=log_dir)
    model.set_logger(new_logger)

    # Create callbacks
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=log_dir, name_prefix='ppo_trackmania')
    trackmania_callback = TrackmaniaCallback(check_freq=1000, log_dir=log_dir)

    # Train the agent
    model.learn(total_timesteps=1000000, callback=[checkpoint_callback, trackmania_callback])

    # Save the final model
    model.save(os.path.join(log_dir, "final_model"))

    # Plot training results
    plot_training_results(log_dir)

# Run the training
if __name__ == "__main__":
    train_trackmania_agent()