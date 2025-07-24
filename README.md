## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-link>
    cd <your-repo-name>
    ```
2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install torch pandas matplotlib jupyter
    ```

## How to Run

1.  **Place the dataset:**
    Make sure the `toy_tts_parallel_data.csv` file is in the root directory of the project.

2.  **Train the model:**
    Run the main training script from the terminal.
    ```bash
    python src/train.py
    ```
    This will start the training process and save the trained model as `tts_model.pth` once completed.

3.  **See a demonstration:**
    Open and run the `demo_notebook.ipynb` in Jupyter to see how the model performs and to visualize the generated audio.
    ```bash
    jupyter notebook demo_notebook.ipynb
    ```
