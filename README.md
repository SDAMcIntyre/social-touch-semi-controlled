-----

# Semi-Controlled Social Touch Experiment

A Python application to run semi-controlled social touch experiments for microneurography and psychophysics studies.

## ✨ Features

  * **Two Experiment Modes**: Tailored workflows for **Microneurography** and **Psychophysics**.
  * **Hardware Integration**: Triggers Kinect camera recordings and sends synchronization signals (TTL for neural recording equipment, LED for video).
  * **Stimulus Control**: Manages fixed or randomized stimulus sequences.
  * **Operator & Participant Cues**: Provides visual and audio cues for stimulus delivery and timing (via a metronome).
  * **Data Integrity**: Logs all experimental parameters and saves data incrementally after each stimulus presentation, allowing for safe cancellation.
  * **Automated File Naming**: Generates organized filenames with participant, unit, date, and time information.

-----

## 📋 Prerequisites

Before you begin, ensure you have the following installed on your system:

  * **Conda**: An installation of [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to manage the Python environment.
  * **Git**: For cloning the repository.
  * **MKVToolNix**: Required for post-processing video files to remove the Kinect's IR track. You can download it from the [official website](https://mkvtoolnix.download/).

-----

## ⚙️ Installation

Follow these steps to set up your local development environment.

### 1\. Clone the Repository

First, clone the project repository from GitHub to your local machine.

```bash
git clone https://github.com/your-username/social-touch-semi-controlled.git
cd social-touch-semi-controlled
```

### 2\. Create and Activate the Conda Environment

This command creates an isolated Conda environment named `social-touch-env` with the required Python version.

```bash
# Create the conda environment
conda create --name social-touch-env python=3.10 -y

# Activate the environment
conda activate social-touch-env
```

Your command prompt should now be prefixed with `(social-touch-env)`.

### 3\. Install Dependencies

Install the required Python packages using `pip` within the active Conda environment.

```bash
# Ensure pip is up-to-date
python -m pip install --upgrade pip

# Install dependencies from requirements.txt
pip install -r requirements.txt

# Install the project package in editable mode
pip install -e .
```

> **Note**: Editable mode (`-e`) allows you to make changes to the source code, and the changes will be reflected immediately without needing to reinstall the package.

-----

## 🚀 Usage

Run experiments from the command line. The main script requires specifying the experiment type and participant details.

### Microneurography Experiment

For microneurography, you need to provide a participant code, a unit name/number, and the number of repeats.

```bash
python -m your_package_name.main --type microneuro --participant P01 --unit U01 --repeats 10
```

### Psychophysics Experiment

For psychophysics, the stimulus sequence is randomized, and a unit name is not required. The application will prompt for participant responses after each stimulus.

```bash
python -m your_package_name.main --type psychophys --participant P02 --repeats 20
```

> Replace `your_package_name` with the actual name of your source code directory.

-----

## 📹 Data Post-Processing

The Kinect camera records an infrared (IR) video track that may need to be removed for analysis. The following command uses `mkvmerge` (from MKVToolNix) to process all `.mkv` files in a directory and create copies without the IR track.

1.  Open Command Prompt **as an administrator**.

2.  Navigate to the directory containing your video files:

    ```cmd
    cd path\to\your\video\data
    ```

3.  Run the following command. It will create new files prefixed with `NoIR_`.

    ```cmd
    FOR /F "delims=*" %A IN ('dir /b *.mkv') DO "C:\Program Files\MKVToolNix\mkvmerge.exe" -o "NoIR_%A" -d !2 --compression -1:none "%A"
    ```

    > **Note**: The path to `mkvmerge.exe` may vary depending on your installation location.