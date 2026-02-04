import gradio as gr
import gymnasium as gym
import numpy as np
import time
import os
import imageio.v2 as imageio # Required for saving frames as GIF

# Define the UCU Hostel/Campus Locations (R, G, Y, B from the Taxi-v3 map)
LOCATIONS = {
    0: 'Sabiti Hostel (R)',
    1: 'Niambi Hostel (G)',
    2: 'UCU Campus (Y)',
    3: 'Maxine Hostel (B)'
}

ACTION_MAP = {
    0: 'Move South', 1: 'Move North', 2: 'Move East',
    3: 'Move West', 4: 'Pickup Student', 5: 'Dropoff Student'
}

Q_TABLE_PATH = "eriduride_q_table.npy"
Q_TABLE = None

def load_q_table():
    global Q_TABLE
    if Q_TABLE is None:
        if os.path.exists(Q_TABLE_PATH):
            Q_TABLE = np.load(Q_TABLE_PATH)
            print("Q-table loaded successfully.")
        else:
            raise FileNotFoundError(f"Q-table not found at {Q_TABLE_PATH}. Please run train_model.py first.")
    return Q_TABLE

def run_simulation_gradio_visual(pickup_location_name, destination_location_name):
    """
    Runs one simulation episode with a user-specified start and end point
    and saves the frames as a temporary GIF for display in Gradio.
    """
    q_table = load_q_table()
    
    # Map names back to indices used by Gymnasium
    pickup_idx = [k for k, v in LOCATIONS.items() if v == pickup_location_name][0]
    dest_idx = [k for k, v in LOCATIONS.items() if v == destination_location_name][0]

    # Initialize environment in RGB_ARRAY mode
    env = gym.make("Taxi-v3", render_mode="rgb_array")

    # Manually set the initial state using the encode function
    initial_state = env.unwrapped.encode(0, 0, pickup_idx, dest_idx) 
    state, info = env.reset()
    env.unwrapped.s = initial_state 
    state = initial_state
    
    done = False
    total_money = 0
    step_count = 0
    frames = [] # To store RGB array frames

    while not done and step_count < 200: # Add step limit as safety
        # Select the action with the highest Q-value (optimal policy)
        action = np.argmax(q_table[state, :])
        
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        total_money += reward
        state = next_state
        step_count += 1
        
        # Capture the RGB frame
        frame = env.render()
        frames.append(frame)

    env.close()

    # Save frames as a temporary GIF file
    gif_path = f"simulation_output_{int(time.time())}.gif"
    imageio.mimsave(gif_path, frames, fps=5) # 5 frames per second playback speed

    summary = f"\n--- Simulation Complete ---\n" \
              f"Finished in {step_count} steps.\n" \
              f"Total Money Earned: {total_money}\n" \
              f"Status: {'Dropped off successfully' if terminated else 'Simulation stopped early (max steps reached/truncated)'}"
    
    # Return the path to the GIF and the summary stats
    return gif_path, total_money, step_count, summary

#  Gradio Interface Setup 

# Dropdowns for user inputs
pickup_input = gr.Dropdown(choices=list(LOCATIONS.values()), label="Select Pickup Hostel/Location", value='Sabiti Hostel (R)')
destination_input = gr.Dropdown(choices=list(LOCATIONS.values()), label="Select Destination Hostel/Campus", value='UCU Campus (Y)')

# Output components change:
output_video = gr.Image(label="Simulation Visualization", type="filepath") # Use Image with type="filepath" for GIF playback
output_money = gr.Number(label="Total Money Earned")
output_steps = gr.Number(label="Total Steps Taken")
output_summary = gr.Textbox(label="Run Summary", lines=5)


# Create the Gradio Interface
demo = gr.Interface(
    fn=run_simulation_gradio_visual, # Use the new visual function
    inputs=[pickup_input, destination_input],
    outputs=[output_video, output_money, output_steps, output_summary],
    title="EriduRide UCU Taxi Model Deployment 🚕",
    description="Run the trained Q-learning agent for a specific pickup and destination point. A visualization of the trip will appear below."
)
if __name__ == "__main__":
    # Ensure the Q table is loaded before launching the app
    load_q_table() 
    demo.launch()
