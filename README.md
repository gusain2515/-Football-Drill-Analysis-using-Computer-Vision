# Football Drill Analysis using Computer Vision

This project provides a Python-based solution for analyzing football drill videos. It uses state-of-the-art computer vision models to dynamically track footballs, identify the player, and determine which ball is actively being used in the drill (the "action ball").

![Sample Output Animation](https://i.imgur.com/aBcDeFg.gif)
*(Sample animation demonstrating the final output)*

---

## Features

- **Multi-Object Tracking**: Detects and tracks multiple footballs simultaneously across video frames.
- **Player Pose Estimation**: Identifies the player's skeleton and tracks the position of their feet.
- **Dynamic Action Ball Identification**: Intelligently determines the "action ball" by its proximity to the player's feet, providing a robust analysis of the drill.
- **Clear Visualization**: Overlays tracking data onto the output video, with stationary balls in green boxes and the action ball marked with a red dot.
- **High Accuracy**: Utilizes the powerful YOLOv8m model to ensure detection of small and distant objects.

---

## Methodology

The core of this project is a hybrid approach that combines two powerful AI models to understand the scene comprehensively:

1.  **Ball Detection & Tracking (YOLOv8 + ByteTrack)**: The system first uses **YOLOv8m**, a state-of-the-art object detection model, to locate all footballs in a frame. It then feeds these detections into the **ByteTrack** algorithm, which assigns a persistent ID to each ball and tracks it through occlusions and fast movements.

2.  **Player Interaction (Mediapipe Pose)**: To understand which ball is in use, the system runs Google's **Mediapipe Pose** model. This lightweight model maps the player's skeleton and provides real-time coordinates for their body parts. We specifically use the coordinates of the player's heels.

3.  **Classification by Proximity**: The "action ball" is determined by finding the tracked ball that is closest to the player's heel coordinates. This is a highly robust method that directly links player interaction to the ball, overcoming the limitations of motion-only analysis.

---

## Setup and Installation

### Prerequisites

- Python 3.8 or higher
- `pip` (Python package installer)

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/football-cv-analysis.git](https://github.com/your-username/football-cv-analysis.git)
    cd football-cv-analysis
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```


---

## Usage

1.  Place your input videos (e.g., `test1.mp4`, `test2.mp4`) in the root directory of the project.

2.  Run the main Python script from your terminal:
    ```bash
    python football_media.py
    ```

3.  The script will process the videos specified in the `if __name__ == "__main__":` block and generate the output files in the same directory.

---

## Output

The script will produce new video files (e.g., `output_mediapipe_1.mp4`). These videos will be identical to the originals but with the following visual overlays:

-   **Stationary Balls**: Marked with a **green bounding box**.
-   **Action Ball**: Marked with a **solid red dot** and the text "Action Ball".
-   **Player Pose**: A faint skeleton will be drawn over the player to visualize the pose detection.

---

## Dependencies

This project relies on the following major Python libraries:

-   `opencv-python`: For video processing and drawing.
-   `ultralytics`: For the YOLOv8 object detection and ByteTrack models.
-   `mediapipe`: For the Pose Estimation model.
-   `numpy`: For numerical operations.


